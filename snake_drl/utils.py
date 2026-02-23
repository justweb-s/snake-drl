from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_run_dir(base_dir: str | os.PathLike, name: str | None = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def json_dump(path: str | os.PathLike, data: Any) -> None:
    def _default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)
