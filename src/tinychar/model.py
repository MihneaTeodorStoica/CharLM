from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.backends.cuda import SDPBackend, sdp_kernel
except Exception:  # pragma: no cover - CPU builds
    SDPBackend = None
    sdp_kernel = None

from .configs import TrainConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float) -> None:
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        x_up, x_gate = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(x_gate) * x_up)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if self._cos_cached is not None and self._cos_cached.size(1) >= seq_len and self._cos_cached.device == device:
            return
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self._cos_cached, self._sin_cached = cos, sin

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + x_rot * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2) + offset
        self._build_cache(seq_len, q.device)
        cos = self._cos_cached[:, :, offset:seq_len, :]
        sin = self._sin_cached[:, :, offset:seq_len, :]
        return self.apply(q, cos, sin), self.apply(k, cos, sin)


class Attention(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.head_dim * self.n_kv_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.head_dim * self.n_kv_head, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_base)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, offset)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        ctx = nullcontext()
        if sdp_kernel is not None and q.is_cuda:  # pragma: no cover - GPU only
            ctx = sdp_kernel(SDPBackend.FLASH_ATTENTION)
        with ctx:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_cache


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, cache = self.attn(self.ln1(x), cache, offset)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, cache


class TinyCharModel(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.size()
        tok = self.tok_emb(idx)
        if cache is None:
            cache = [None] * len(self.blocks)
            offset = 0
        else:
            offset = cache[0][0].size(2) if cache[0] is not None else 0
        for i, block in enumerate(self.blocks):
            tok, cache[i] = block(tok, cache[i], offset)
        tok = self.ln_f(tok)
        logits = self.lm_head(tok)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, cache

    @torch.no_grad()
    def generate(
        self,
        prompt: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device
        idx = torch.tensor(prompt, dtype=torch.long, device=device)[None, :]
        out = idx
        if idx.size(1) == 0:
            logits = torch.zeros((1, 1, self.cfg.vocab_size), device=device)
            cache = [None] * len(self.blocks)
        else:
            logits, _, cache = self(idx)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                v, ix = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, ix, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            out = torch.cat([out, next_id], dim=1)
            logits, _, cache = self(next_id, cache=cache)
        return out[0].tolist()
