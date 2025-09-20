"""Model definition for CharLM."""
from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - CPU builds lack flash kernels
    from torch.backends.cuda import SDPBackend, sdp_kernel
except Exception:  # pragma: no cover - disable flash attention path
    SDPBackend = None
    sdp_kernel = None

from .config import ModelConfig
from .layers import RMSNorm, RotaryEmbedding, SwiGLU

CacheEntry = Optional[Tuple[torch.Tensor, torch.Tensor]]
CacheList = List[CacheEntry]


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base)

    def forward(
        self,
        x: torch.Tensor,
        cache: CacheEntry,
        offset: int,
        max_cache_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, CacheEntry]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, offset)
        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        if max_cache_tokens is not None and k.size(2) > max_cache_tokens:
            k = k[:, :, -max_cache_tokens:, :]
            v = v[:, :, -max_cache_tokens:, :]
        new_cache: CacheEntry = (k, v)
        if self.n_kv_head != self.n_head:
            k_for_attn = k.repeat_interleave(self.n_rep, dim=1)
            v_for_attn = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_for_attn = k
            v_for_attn = v
        ctx = nullcontext()
        if sdp_kernel is not None and q.is_cuda:
            ctx = sdp_kernel(SDPBackend.FLASH_ATTENTION)
        with ctx:
            y = F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: CacheEntry,
        offset: int,
        max_cache_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, CacheEntry]:
        attn_out, new_cache = self.attn(self.ln1(x), cache, offset, max_cache_tokens)
        x = x + self.resid_dropout(attn_out)
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))
        return x, new_cache


class CharLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[CacheList] = None,
        max_cache_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], CacheList]:
        if idx.dim() != 2:
            raise ValueError("idx must be [batch, time]")
        B, T = idx.shape
        tok = self.tok_emb(idx)
        if cache is None:
            cache = [None] * len(self.blocks)
            offset = 0
        else:
            first = cache[0]
            offset = first[0].size(2) if first is not None else 0
        new_cache: CacheList = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            tok, new_cache[i] = block(tok, cache[i], offset, max_cache_tokens)
        tok = self.ln_f(tok)
        logits = self.lm_head(tok)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_cache

    @torch.no_grad()
    def generate(
        self,
        prompt: Sequence[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
    ) -> List[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        device = next(self.parameters()).device
        cur = torch.tensor(list(prompt), dtype=torch.long, device=device)[None, :]
        self.eval()
        cache: Optional[CacheList] = None
        if cur.numel() > 0:
            logits, _, cache = self(cur, max_cache_tokens=max_context_tokens)
        else:
            logits = torch.zeros((1, 1, self.cfg.vocab_size), device=device)
            cache = [None] * len(self.blocks)
        generated = cur
        for _ in range(max_new_tokens):
            logits_step = logits[:, -1, :] / temperature
            if top_k:
                values, indices = torch.topk(logits_step, top_k)
                probs = torch.zeros_like(logits_step).scatter_(1, indices, F.softmax(values, dim=-1))
            else:
                probs = F.softmax(logits_step, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            logits, _, cache = self(next_id, cache=cache, max_cache_tokens=max_context_tokens)
        return generated[0].tolist()


__all__ = ["CharLM"]
