import torch

from tinychar.configs import TrainConfig
from tinychar.model import TinyCharModel


def test_generate_empty_prompt():
    cfg = TrainConfig(d_model=32, n_layer=2, n_head=4, n_kv_head=2, seq_len=16)
    model = TinyCharModel(cfg)
    torch.manual_seed(0)
    out = model.generate([], max_new_tokens=10)
    assert len(out) == 10


def test_generate_non_empty_prompt():
    cfg = TrainConfig(d_model=32, n_layer=2, n_head=4, n_kv_head=2, seq_len=16)
    model = TinyCharModel(cfg)
    prompt = [1, 2, 3]
    torch.manual_seed(0)
    out = model.generate(prompt, max_new_tokens=2)
    assert out[: len(prompt)] == prompt
    assert len(out) == len(prompt) + 2
