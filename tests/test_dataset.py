import numpy as np

from tinychar.dataset import PackedDataset


def test_dataset_separator(tmp_path):
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    bin_path = tmp_path / "train.bin"
    data.tofile(bin_path)
    idx = np.array([0, 3, 5], dtype=np.int64)
    idx_path = tmp_path / "train.idx"
    idx.tofile(idx_path)
    ds = PackedDataset(str(bin_path), str(idx_path), seq_len=4, separator=0, seed=0)
    x, y = ds[0]
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert 0 in x.numpy() or 0 in y.numpy()
