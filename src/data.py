from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataSpec:
    dataset: str
    data_dir: Path
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 0


def _common_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_gtsrb_loaders(spec: DataSpec) -> Tuple[DataLoader, DataLoader, int]:
    root = spec.data_dir / "gtsrb"
    train_ds = datasets.GTSRB(
        root=str(root),
        split="train",
        download=True,
        transform=_common_transforms(spec.image_size, train=True),
    )
    test_ds = datasets.GTSRB(
        root=str(root),
        split="test",
        download=True,
        transform=_common_transforms(spec.image_size, train=False),
    )
    # Val split: %80 train, %20 val
    val_size = int(0.2 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=spec.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=spec.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # Torchvision versions differ: GTSRB may not expose `classes`.
    if hasattr(train_ds, "classes"):
        num_classes = len(train_ds.classes)  # type: ignore[attr-defined]
    elif hasattr(train_ds, "_labels"):
        labels = getattr(train_ds, "_labels")
        num_classes = int(max(labels)) + 1
    elif hasattr(train_ds, "targets"):
        targets = getattr(train_ds, "targets")
        num_classes = int(max(targets)) + 1
    elif hasattr(train_ds, "_samples"):
        samples = getattr(train_ds, "_samples")
        num_classes = int(max(lbl for _, lbl in samples)) + 1
    else:
        # Official GTSRB has 43 classes.
        num_classes = 43

    return train_loader, test_loader, val_loader,num_classes


def build_loaders(spec: DataSpec) -> Tuple[DataLoader, DataLoader, int]:
    ds = spec.dataset.lower()
    if ds == "gtsrb":
        return build_gtsrb_loaders(spec)
    raise ValueError(f"Unknown dataset: {spec.dataset}")

