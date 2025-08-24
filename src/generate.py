import argparse
import torch

from tinychar.configs import TrainConfig
from tinychar.model import TinyCharModel


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    model = TinyCharModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids = [ord(c) for c in args.prompt.encode("utf-8", errors="ignore")]
    out = model.generate(ids, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.top_k or None)
    text = bytes(out).decode("utf-8", errors="ignore")
    print(text)


if __name__ == "__main__":
    main()
