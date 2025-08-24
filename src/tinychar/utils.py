import math
import os
from dataclasses import asdict

import torch
from torch.optim import AdamW


def cosine_lr(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def setup_optimizer(model: torch.nn.Module, cfg) -> AdamW:
    fused = hasattr(torch.optim, "AdamW") and torch.cuda.is_available()
    return AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, fused=fused)


def bits_per_byte(loss: float) -> float:
    return loss / math.log(2)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer, cfg, step: int, best: bool = False) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best": best,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(obj, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
