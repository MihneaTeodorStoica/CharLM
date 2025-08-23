from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainConfig:
    """Configuration for training and model architecture."""

    vocab_size: int = 256
    d_model: int = 512
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 2
    seq_len: int = 1024
    dropout: float = 0.0
    rope_base: float = 10000.0
    mlp_ratio: float = 5.33
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 2000
    max_steps: int = 200000
    min_lr: float = 1e-5
    batch_size_tokens: int = 1_048_576
    micro_bsz: int = 8
    log_interval: int = 100
    save_interval: int = 2000
    grad_clip: float = 1.0
    compile: bool = False
    seed: int = 0
    ckpt_path: Optional[str] = None
    wandb_project: Optional[str] = None
