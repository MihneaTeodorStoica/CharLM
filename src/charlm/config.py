"""Configuration helpers for CharLM."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 512
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 2
    seq_len: int = 1024
    dropout: float = 0.0
    mlp_ratio: float = 5.33
    rope_base: float = 10_000.0

    def validate(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: Optional[bool] = None

    def validate(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 < self.betas[0] < 1 and 0 < self.betas[1] < 1):
            raise ValueError("betas must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


@dataclass
class SchedulerConfig:
    warmup_steps: int = 2000
    max_steps: int = 200_000
    min_lr: float = 1e-5

    def validate(self, base_lr: float) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.min_lr < 0 or self.min_lr > base_lr:
            raise ValueError("min_lr must be in [0, lr]")


@dataclass
class CheckpointConfig:
    out_dir: str = "checkpoints"
    resume_path: Optional[str] = None
    save_interval: int = 2000
    keep_last: int = 3

    def validate(self) -> None:
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.keep_last <= 0:
            raise ValueError("keep_last must be positive")


@dataclass
class LoggingConfig:
    log_interval: int = 100
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def validate(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.use_wandb and not self.wandb_project:
            raise ValueError("wandb_project is required when use_wandb is True")


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    batch_size_tokens: int = 1_048_576
    micro_batch_size: int = 8
    data_workers: int = 2
    grad_clip: float = 1.0
    precision: str = "auto"  # auto|fp32|fp16|bf16
    seed: int = 0
    compile: bool = False
    eval_interval: int = 2000
    eval_samples: int = 8
    eval_batches: int = 8

    def validate(self) -> None:
        self.model.validate()
        self.optimizer.validate()
        self.scheduler.validate(self.optimizer.lr)
        self.checkpoint.validate()
        self.logging.validate()
        if self.batch_size_tokens <= 0:
            raise ValueError("batch_size_tokens must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.data_workers < 0:
            raise ValueError("data_workers must be non-negative")
        tokens_per_micro = self.micro_batch_size * self.model.seq_len
        if self.batch_size_tokens % tokens_per_micro != 0:
            raise ValueError(
                "batch_size_tokens must be divisible by micro_batch_size * seq_len"
            )
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")
        if self.eval_samples < 0:
            raise ValueError("eval_samples must be non-negative")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of auto, fp32, fp16, bf16")

    @property
    def grad_accum_steps(self) -> int:
        return self.batch_size_tokens // (self.micro_batch_size * self.model.seq_len)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, obj: Dict[str, Any]) -> "TrainingConfig":
        def build(datacls, data):
            return datacls(**data) if isinstance(data, dict) else data

        obj = dict(obj)
        if "model" in obj:
            obj["model"] = build(ModelConfig, obj["model"])
        if "optimizer" in obj:
            obj["optimizer"] = build(OptimizerConfig, obj["optimizer"])
        if "scheduler" in obj:
            obj["scheduler"] = build(SchedulerConfig, obj["scheduler"])
        if "checkpoint" in obj:
            obj["checkpoint"] = build(CheckpointConfig, obj["checkpoint"])
        if "logging" in obj:
            obj["logging"] = build(LoggingConfig, obj["logging"])
        cfg = cls(**obj)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.from_mapping(data)


@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    max_context_tokens: int = 1024

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")


def load_training_config(path: Optional[str]) -> TrainingConfig:
    if path is None:
        cfg = TrainingConfig()
        cfg.validate()
        return cfg
    return TrainingConfig.from_yaml(path)


def dump_training_config(cfg: TrainingConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


__all__ = [
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "TrainingConfig",
    "InferenceConfig",
    "load_training_config",
    "dump_training_config",
]
