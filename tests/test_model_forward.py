import torch

from charlm.config import ModelConfig
from charlm.model import CharLM


def test_forward_and_cache_trimming():
    cfg = ModelConfig(
        d_model=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        seq_len=16,
        dropout=0.0,
    )
    model = CharLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, loss, cache = model(x, x)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert loss is not None and loss.item() > 0
    next_tokens = torch.randint(0, cfg.vocab_size, (2, 4))
    _, _, cache = model(next_tokens, cache=cache, max_cache_tokens=8)
    assert cache[0][0].shape[2] <= 8


def test_generate_length_and_prefix():
    cfg = ModelConfig(
        d_model=32,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        seq_len=8,
        dropout=0.0,
    )
    model = CharLM(cfg)
    prompt = [1, 2, 3]
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=4, max_context_tokens=8)
    assert out[: len(prompt)] == prompt
    assert len(out) == len(prompt) + 5
