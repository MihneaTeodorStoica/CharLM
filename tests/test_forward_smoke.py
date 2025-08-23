import torch

from tinychar.configs import TrainConfig
from tinychar.model import TinyCharModel


def test_forward_smoke():
    cfg = TrainConfig(d_model=32, n_layer=2, n_head=4, n_kv_head=2, seq_len=16)
    model = TinyCharModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, loss, _ = model(x, x)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert loss.item() > 0
    torch.manual_seed(0)
    logits1, _, _ = model(x)
    torch.manual_seed(0)
    logits2, _, _ = model(x)
    assert torch.allclose(logits1, logits2)
