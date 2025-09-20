"""Core building blocks for CharLM."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float) -> None:
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.up = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_cached(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, device)
        item = self._cache.get(key)
        if item is not None:
            return item
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self._cache[key] = (cos, sin)
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2) + offset
        cos, sin = self._get_cached(seq_len, q.device)
        cos = cos[:, :, offset:seq_len, :]
        sin = sin[:, :, offset:seq_len, :]
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + x_rot * sin


__all__ = ["RMSNorm", "SwiGLU", "RotaryEmbedding"]
