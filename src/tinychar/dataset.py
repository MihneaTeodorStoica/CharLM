import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class PackedDataset(Dataset):
    """Memmapped byte-level dataset with document packing."""

    def __init__(self, bin_path: str, idx_path: str, seq_len: int, separator: int = 0, seed: int = 0) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.idx = np.fromfile(idx_path, dtype=np.int64)
        self.seq_len = seq_len
        self.sep = np.uint8(separator)
        self.rng = np.random.default_rng(seed)
        self.n_docs = len(self.idx) - 1

    def __len__(self) -> int:
        return max(1, len(self.data) // self.seq_len)

    def __getitem__(self, _):
        out = np.empty(self.seq_len + 1, dtype=np.uint8)
        filled = 0
        while filled < self.seq_len + 1:
            doc_i = int(self.rng.integers(0, self.n_docs))
            start = self.idx[doc_i]
            end = self.idx[doc_i + 1]
            doc = self.data[start:end]
            take = min(len(doc), self.seq_len + 1 - filled)
            out[filled : filled + take] = doc[:take]
            filled += take
            if filled < self.seq_len + 1:
                out[filled] = self.sep
                filled += 1
        x = torch.from_numpy(out[:-1].astype(np.int64))
        y = torch.from_numpy(out[1:].astype(np.int64))
        return x, y


def create_dataloader(bin_path: str, idx_path: str, seq_len: int, batch_size: int, num_workers: int = 2) -> DataLoader:
    ds = PackedDataset(bin_path, idx_path, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
