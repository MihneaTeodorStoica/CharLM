"""CLI for sampling from CharLM."""
from __future__ import annotations

import argparse

import torch

from charlm.config import TrainingConfig
from charlm.model import CharLM
from charlm.utils import select_device


def load_model(checkpoint: str, device: torch.device) -> CharLM:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg_dict = ckpt.get("config")
    if not isinstance(cfg_dict, dict):
        raise ValueError("checkpoint missing configuration")
    cfg = TrainingConfig.from_mapping(cfg_dict)
    model = CharLM(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with CharLM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-context", type=int, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    ids = list(args.prompt.encode("utf-8", errors="ignore"))
    generated = model.generate(
        ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k or None,
        max_context_tokens=args.max_context,
    )
    text = bytes(generated).decode("utf-8", errors="ignore")
    print(text)


if __name__ == "__main__":
    main()
