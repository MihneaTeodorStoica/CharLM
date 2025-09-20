"""Miscellaneous utilities."""
from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bits_per_byte(loss: float) -> float:
    return loss / math.log(2)


def select_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - macOS only
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(device: torch.device, precision: str) -> torch.dtype:
    if precision == "fp32" or device.type == "cpu":
        return torch.float32
    if precision == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        raise ValueError("bfloat16 requested but not supported on this device")
    if precision == "fp16":
        if device.type == "cuda":
            return torch.float16
        raise ValueError("float16 requested but CUDA is unavailable")
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


__all__ = [
    "set_seed",
    "bits_per_byte",
    "select_device",
    "resolve_precision",
    "get_world_size",
]
