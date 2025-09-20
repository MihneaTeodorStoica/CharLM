import numpy as np
import torch

from charlm.data import PackedDataset


def test_packed_dataset_shapes(tmp_path):
    values = np.arange(1, 11, dtype=np.uint8)
    bin_path = tmp_path / "train.bin"
    values.tofile(bin_path)
    idx_path = tmp_path / "train.idx"
    np.array([0, 5, 10], dtype=np.int64).tofile(idx_path)

    dataset = PackedDataset(str(bin_path), str(idx_path), seq_len=6, separator=0, seed=123)
    x, y = dataset[0]
    assert x.shape == (6,)
    assert y.shape == (6,)
    # ensure separator appears due to packing multiple docs
    assert (x == 0).any().item() or (y == 0).any().item()


def test_dataset_reproducibility(tmp_path):
    values = np.arange(1, 7, dtype=np.uint8)
    bin_path = tmp_path / "train.bin"
    values.tofile(bin_path)
    idx_path = tmp_path / "train.idx"
    np.array([0, 3, 6], dtype=np.int64).tofile(idx_path)

    ds_a = PackedDataset(str(bin_path), str(idx_path), seq_len=4, seed=7)
    ds_b = PackedDataset(str(bin_path), str(idx_path), seq_len=4, seed=7)
    sample_a = torch.stack([ds_a[i][0] for i in range(3)])
    sample_b = torch.stack([ds_b[i][0] for i in range(3)])
    assert torch.equal(sample_a, sample_b)
