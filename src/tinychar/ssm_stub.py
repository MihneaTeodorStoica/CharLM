"""Placeholder for future state-space model blocks."""

import torch
import torch.nn as nn


class SSMBlock(nn.Module):
    """A simple linear layer acting as a placeholder."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.linear(x)
