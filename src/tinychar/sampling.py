from typing import List, Optional

import torch
import torch.nn.functional as F


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    v, ix = torch.topk(logits, k)
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(1, ix, v)
    return out


def sample(model, prompt: List[int], max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> List[int]:
    device = next(model.parameters()).device
    idx = torch.tensor(prompt, dtype=torch.long, device=device)[None, :]
    logits, _, cache = model(idx)
    out = idx
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :] / temperature
        if top_k is not None and top_k > 0:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        out = torch.cat([out, next_id], dim=1)
        logits, _, cache = model(next_id, cache=cache)
    return out[0].tolist()
