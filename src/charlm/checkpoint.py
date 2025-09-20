"""Checkpoint utilities."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .config import CheckpointConfig, TrainingConfig


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._recent: deque[Path] = deque(maxlen=cfg.keep_last)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        train_cfg: TrainingConfig,
        step: int,
        best_metric: Optional[float] = None,
    ) -> Path:
        path = self.root / f"step-{step}.pt"
        obj = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": asdict(train_cfg),
            "step": step,
            "best_metric": best_metric,
        }
        torch.save(obj, path)
        self._recent.append(path)
        self._prune()
        return path

    def _prune(self) -> None:
        if self.cfg.keep_last <= 0:
            return
        while len(self._recent) > self.cfg.keep_last:
            oldest = self._recent.popleft()
            if oldest.exists():
                oldest.unlink()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


__all__ = ["CheckpointManager", "load_checkpoint"]
