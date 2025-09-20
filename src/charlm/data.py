"""Data loading utilities."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


class PackedDataset(Dataset):
    """Byte-level memmapped dataset with document packing."""

    def __init__(
        self,
        bin_path: str,
        idx_path: str,
        seq_len: int,
        separator: int = 0,
        seed: int = 0,
    ) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.idx = np.fromfile(idx_path, dtype=np.int64)
        if self.idx.ndim != 1 or len(self.idx) < 2:
            raise ValueError("idx file must contain at least two entries")
        self.seq_len = seq_len
        self.separator = np.uint8(separator)
        self.seed = seed
        self.n_docs = len(self.idx) - 1
        self._rng = np.random.default_rng(seed)

    def _rng_for_worker(self) -> np.random.Generator:
        info = get_worker_info()
        if info is None:
            return self._rng
        # Each worker gets a deterministic, independent stream.
        seed = self.seed + info.id
        if not hasattr(info.dataset, "_worker_rngs"):
            info.dataset._worker_rngs = {}
        cache = info.dataset._worker_rngs
        if seed not in cache:
            cache[seed] = np.random.default_rng(seed)
        return cache[seed]

    def __len__(self) -> int:
        return max(1, len(self.data) // self.seq_len)

    def __getitem__(self, _: int):
        rng = self._rng_for_worker()
        out = np.empty(self.seq_len + 1, dtype=np.uint8)
        filled = 0
        while filled < self.seq_len + 1:
            doc_idx = int(rng.integers(0, self.n_docs))
            start = self.idx[doc_idx]
            end = self.idx[doc_idx + 1]
            chunk = self.data[start:end]
            take = min(len(chunk), self.seq_len + 1 - filled)
            out[filled : filled + take] = chunk[:take]
            filled += take
            if filled < self.seq_len + 1:
                out[filled] = self.separator
                filled += 1
        x = torch.from_numpy(out[:-1].astype(np.int64))
        y = torch.from_numpy(out[1:].astype(np.int64))
        return x, y


def create_dataloader(
    bin_path: str,
    idx_path: str,
    seq_len: int,
    micro_batch_size: int,
    *,
    separator: int = 0,
    seed: int = 0,
    num_workers: int = 2,
) -> DataLoader:
    dataset = PackedDataset(bin_path, idx_path, seq_len, separator=separator, seed=seed)

    def _init_worker(worker_id: int) -> None:  # pragma: no cover - torch invokes in worker
        torch.manual_seed(seed + worker_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_init_worker,
        persistent_workers=num_workers > 0,
    )


__all__ = ["PackedDataset", "create_dataloader"]
