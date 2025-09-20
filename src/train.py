"""Entry point for training CharLM."""
from __future__ import annotations

import argparse

from charlm.config import TrainingConfig, load_training_config
from charlm.trainer import Trainer


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.max_steps is not None:
        cfg.scheduler.max_steps = args.max_steps
    if args.warmup_steps is not None:
        cfg.scheduler.warmup_steps = args.warmup_steps
    if args.batch_size_tokens is not None:
        cfg.batch_size_tokens = args.batch_size_tokens
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    if args.precision is not None:
        cfg.precision = args.precision
    if args.compile is not None:
        cfg.compile = args.compile
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.validate()
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CharLM model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--train-data", default="data/out/train.bin")
    parser.add_argument("--train-index", default="data/out/train.idx")
    parser.add_argument("--val-data", default="data/out/val.bin")
    parser.add_argument("--val-index", default="data/out/val.idx")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--batch-size-tokens", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--compile", type=bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-val", action="store_true", help="Skip validation during training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config)
    cfg = apply_overrides(cfg, args)
    val_bin = None if args.no_val else args.val_data
    val_idx = None if args.no_val else args.val_index
    trainer = Trainer(
        cfg,
        train_bin=args.train_data,
        train_idx=args.train_index,
        val_bin=val_bin,
        val_idx=val_idx,
        device=args.device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
