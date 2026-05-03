from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, help="Path to a run history.json file")
    p.add_argument("--out-dir", default=None, help="Directory for generated plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"Empty history file: {history_path}")

    out_dir = ensure_dir(args.out_dir or history_path.parent)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_acc"] for row in history], marker="o", label="train_acc")
    plt.plot(epochs, [row["val_acc"] for row in history], marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_path = Path(out_dir) / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = Path(out_dir) / "loss_curve.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    print(f"saved: {acc_path}")
    print(f"saved: {loss_path}")


if __name__ == "__main__":
    main()

