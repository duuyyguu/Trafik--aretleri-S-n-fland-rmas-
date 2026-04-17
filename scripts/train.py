from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data import DataSpec, build_loaders
from src.modeling import ModelSpec, accuracy_top1, build_model
from src.utils import (
    RunInfo,
    atomic_symlink_or_copy_latest,
    ensure_dir,
    get_device,
    now_utc_compact,
    save_checkpoint,
    save_run_metadata,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gtsrb", choices=["gtsrb"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    p.add_argument("--min-delta", type=float, default=1e-4, help="Min improvement for early stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    run_root = ensure_dir("runs")
    run_dir = ensure_dir(run_root / f"{args.dataset}_{args.model}_{now_utc_compact()}")

    train_loader, test_loader, num_classes = build_loaders(
        DataSpec(
            dataset=args.dataset,
            data_dir=Path(args.data_dir),
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    model = build_model(num_classes=num_classes, spec=ModelSpec(args.model)).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    info = RunInfo(
        dataset=args.dataset,
        model=args.model,
        num_classes=num_classes,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        created_utc=now_utc_compact(),
    )
    save_run_metadata(run_dir, info)

    best_acc = -1.0
    best_path = Path(run_dir) / "best.pt"
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            acc = accuracy_top1(logits.detach(), y)
            running_loss += loss.item()
            running_acc += acc
            n_batches += 1
            lr_now = opt.param_groups[0]["lr"]
            pbar.set_postfix(loss=running_loss / n_batches, acc=running_acc / n_batches, lr=lr_now)

        model.eval()
        test_acc = 0.0
        test_batches = 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="eval"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                test_acc += accuracy_top1(logits, y)
                test_batches += 1
        test_acc /= max(1, test_batches)

        if scheduler is not None:
            scheduler.step()

        ckpt = {
            "model": args.model,
            "num_classes": num_classes,
            "image_size": args.image_size,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "test_acc": test_acc,
        }
        save_checkpoint(Path(run_dir) / "last.pt", ckpt)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(best_path, ckpt)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, n_batches),
            "train_acc": running_acc / max(1, n_batches),
            "test_acc": test_acc,
            "best_acc": best_acc,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        (Path(run_dir) / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(
            f"epoch={epoch} train_acc={epoch_row['train_acc']:.4f} "
            f"test_acc={test_acc:.4f} best={best_acc:.4f} "
            f"no_improve={epochs_no_improve}/{args.patience}"
        )

        if epochs_no_improve >= args.patience:
            print("early stopping triggered")
            break

    latest = Path("runs") / "latest.pt"
    atomic_symlink_or_copy_latest(Path(run_dir), best_path, latest)
    print(f"saved best checkpoint: {best_path}")
    print(f"latest checkpoint: {latest}")


if __name__ == "__main__":
    main()

