import numpy as np

from tinychar.configs import TrainConfig
from tinychar.model import TinyCharModel
from tinychar.dataset import PackedDataset
from eval_bpb import eval_bpb


def test_eval_bpb(tmp_path):
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    bin_path = tmp_path / "val.bin"
    data.tofile(bin_path)
    idx_path = tmp_path / "val.idx"
    np.array([0, 3, 6], dtype=np.int64).tofile(idx_path)
    cfg = TrainConfig(d_model=32, n_layer=2, n_head=4, n_kv_head=2, seq_len=4)
    model = TinyCharModel(cfg)
    ds = PackedDataset(str(bin_path), str(idx_path), seq_len=4)
    bpb = eval_bpb(model, ds, batch_size=1)
    assert bpb > 0
