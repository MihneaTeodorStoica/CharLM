#!/usr/bin/env python
"""Create a tiny public-domain dataset for quick prototyping.

Downloads the Tiny Shakespeare corpus (~1MB) and converts it into
memmapped binary files expected by ``PackedDataset``.
"""
import pathlib
import urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    text = urllib.request.urlopen(URL, timeout=30).read()
    data = np.frombuffer(text, dtype=np.uint8)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    out_dir = pathlib.Path("data/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    train.tofile(out_dir / "train.bin")
    np.array([0, len(train)], dtype=np.int64).tofile(out_dir / "train.idx")
    val.tofile(out_dir / "val.bin")
    np.array([0, len(val)], dtype=np.int64).tofile(out_dir / "val.idx")
    print("Sample dataset written to", out_dir)


if __name__ == "__main__":
    main()
