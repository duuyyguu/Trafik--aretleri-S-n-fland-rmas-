from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.data import DataSpec, build_loaders
from src.modeling import ModelSpec, build_model
from src.utils import ensure_dir, get_device, load_checkpoint, now_utc_compact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gtsrb", choices=["gtsrb"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ckpt", default="runs/latest.pt")
    p.add_argument("--model", default=None, help="Override model name stored in checkpoint")
    p.add_argument("--image-size", type=int, default=None, help="Override image size stored in checkpoint")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default=None, help="Output directory for eval artifacts")
    p.add_argument("--max-labels", type=int, default=25, help="Max labels to annotate on confusion matrix")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model_name = args.model or ckpt.get("model", "resnet18")
    image_size = args.image_size or int(ckpt.get("image_size", 64))
    num_classes = int(ckpt["num_classes"])

    out_dir = ensure_dir(
        args.out_dir
        or Path("runs") / f"eval_{Path(args.ckpt).stem}_{now_utc_compact()}"
    )

    _, test_loader, _ = build_loaders(
        DataSpec(
            dataset=args.dataset,
            data_dir=Path(args.data_dir),
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    model = build_model(num_classes=num_classes, spec=ModelSpec(model_name))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="eval"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy={acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("confusion_matrix_shape:", cm.shape)

    metrics = {
        "ckpt": str(args.ckpt),
        "model": model_name,
        "image_size": image_size,
        "num_classes": num_classes,
        "accuracy": float(acc),
        "confusion_matrix_shape": list(cm.shape),
    }

    (Path(out_dir) / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Confusion matrix visualization (avoid unreadable 43x43 labels by default)
    plt.figure(figsize=(12, 10))
    annotate = num_classes <= args.max_labels
    sns.heatmap(
        cm,
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=annotate,
        yticklabels=annotate,
    )
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = Path(out_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"saved: {Path(out_dir) / 'metrics.json'}")
    print(f"saved: {cm_path}")


if __name__ == "__main__":
    main()

