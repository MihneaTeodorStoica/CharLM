"""Compute bits-per-byte on a validation corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from charlm.config import ModelConfig, TrainingConfig
from charlm.data import create_dataloader
from charlm.model import CharLM
from charlm.utils import bits_per_byte, select_device


def load_model(checkpoint: str, device: torch.device) -> CharLM:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config")
    if not isinstance(cfg_dict, dict):
        raise ValueError("checkpoint missing configuration")
    train_cfg = TrainingConfig.from_mapping(cfg_dict)
    model = CharLM(train_cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def eval_bpb(model: CharLM, dataloader) -> float:
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            _, loss, _ = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    if total_tokens == 0:
        raise ValueError("validation loader produced zero tokens")
    return bits_per_byte(total_loss / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bits-per-byte")
    parser.add_argument("--checkpoint", default="checkpoints/last.pt")
    parser.add_argument("--data", default="data/out/val.bin")
    parser.add_argument("--index", default="data/out/val.idx")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    loader = create_dataloader(
        args.data,
        args.index,
        seq_len=args.seq_len,
        micro_batch_size=args.batch,
        seed=0,
        num_workers=0,
    )
    score = eval_bpb(model, loader)
    print(f"bpb {score:.4f}")


if __name__ == "__main__":
    main()
