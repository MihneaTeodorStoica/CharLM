import argparse
import math

import torch
from torch.utils.data import DataLoader

from tinychar.configs import TrainConfig
from tinychar.dataset import PackedDataset
from tinychar.model import TinyCharModel


def eval_bpb(model: TinyCharModel, ds: PackedDataset, batch_size: int = 32) -> float:
    loader = DataLoader(ds, batch_size=batch_size)
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            _, loss, _ = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    return total_loss / total_tokens / math.log(2)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute bpb")
    p.add_argument("--checkpoint", default="checkpoints/last.pt")
    p.add_argument("--data", default="data/out/val.bin")
    p.add_argument("--index", default="data/out/val.idx")
    p.add_argument("--seq_len", type=int, default=1024)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    model = TinyCharModel(cfg)
    model.load_state_dict(ckpt["model"])
    ds = PackedDataset(args.data, args.index, args.seq_len)
    bpb = eval_bpb(model, ds)
    print(f"bpb {bpb:.4f}")


if __name__ == "__main__":
    main()
