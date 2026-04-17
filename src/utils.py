from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def now_utc_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class RunInfo:
    dataset: str
    model: str
    num_classes: int
    image_size: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    created_utc: str


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location or "cpu")


def save_run_metadata(run_dir: str | Path, info: RunInfo) -> None:
    run_dir = ensure_dir(run_dir)
    write_json(run_dir / "run.json", asdict(info))


def atomic_symlink_or_copy_latest(run_dir: Path, ckpt_path: Path, latest_path: Path) -> None:
    # Windows symlink may require admin; fall back to copy.
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        os.symlink(str(ckpt_path), str(latest_path))
    except Exception:
        latest_path.write_bytes(ckpt_path.read_bytes())

