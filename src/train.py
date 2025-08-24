import argparse
import os

import torch
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

from tinychar.configs import TrainConfig
from tinychar.dataset import create_dataloader
from tinychar.model import TinyCharModel
from tinychar.utils import cosine_lr, load_checkpoint, save_checkpoint, set_seed, setup_optimizer


def load_config(path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if path:
        with open(path) as f:
            user = yaml.safe_load(f)
        for k, v in user.items():
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyChar")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/out/train.bin")
    parser.add_argument("--index", type=str, default="data/out/train.idx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg.ckpt_path = args.checkpoint
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = TinyCharModel(cfg).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore
    optimizer = setup_optimizer(model, cfg)
    scaler = GradScaler(enabled=dtype == torch.float16)

    start_step = 0
    if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        ckpt = load_checkpoint(cfg.ckpt_path, model, optimizer)
        start_step = ckpt.get("step", 0)

    loader = create_dataloader(args.data, args.index, cfg.seq_len, cfg.micro_bsz)
    loader_iter = iter(loader)
    grad_accum = cfg.batch_size_tokens // (cfg.micro_bsz * cfg.seq_len)

    for step in tqdm(range(start_step, cfg.max_steps), dynamic_ncols=True):
        lr = cosine_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        model.train()
        optimizer.zero_grad()
        for _ in range(grad_accum):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            with autocast(dtype=dtype):
                _, loss, _ = model(x, y)
            scaler.scale(loss / grad_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_interval == 0:
            print(f"step {step} loss {loss.item():.4f} lr {lr:.5f}")
        if step % cfg.save_interval == 0 and step > 0:
            save_checkpoint(f"checkpoints/step-{step}.pt", model, optimizer, cfg, step)

    save_checkpoint("checkpoints/last.pt", model, optimizer, cfg, cfg.max_steps)


if __name__ == "__main__":
    main()
