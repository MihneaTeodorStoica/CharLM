"""Optimizer helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch.optim import AdamW

from .config import OptimizerConfig


def _split_decay(model: torch.nn.Module) -> Tuple[Iterable[torch.nn.Parameter], Iterable[torch.nn.Parameter]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        elif param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    decay, no_decay = _split_decay(model)
    fused = cfg.fused if cfg.fused is not None else torch.cuda.is_available()
    optim = AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=fused,
    )
    return optim


__all__ = ["build_optimizer"]
