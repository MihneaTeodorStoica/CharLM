"""Learning rate schedules."""
from __future__ import annotations

import math

from .config import SchedulerConfig


def cosine_with_warmup(step: int, cfg: SchedulerConfig, base_lr: float) -> float:
    if step < cfg.warmup_steps:
        if cfg.warmup_steps == 0:
            return base_lr
        return base_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr + (base_lr - cfg.min_lr) * cosine


__all__ = ["cosine_with_warmup"]
