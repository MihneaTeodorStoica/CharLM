"""Training loop implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .checkpoint import CheckpointManager, load_checkpoint
from .config import TrainingConfig
from .data import create_dataloader
from .model import CharLM
from .optim import build_optimizer
from .scheduler import cosine_with_warmup
from .utils import bits_per_byte, resolve_precision, select_device, set_seed


class Trainer:
    def __init__(
        self,
        cfg: TrainingConfig,
        *,
        train_bin: str,
        train_idx: str,
        val_bin: Optional[str] = None,
        val_idx: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.device = select_device(device)
        self.dtype = resolve_precision(self.device, cfg.precision)
        self.model = CharLM(cfg.model).to(self.device)
        if cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[arg-type]
        self.optimizer = build_optimizer(self.model, cfg.optimizer)
        use_amp = self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}
        amp_dtype = torch.float16 if self.dtype == torch.float16 else torch.bfloat16
        self.scaler = GradScaler(enabled=use_amp and self.dtype == torch.float16)
        self.autocast_kwargs = {
            "device_type": self.device.type,
            "dtype": amp_dtype,
            "enabled": use_amp,
        }
        self.manager = CheckpointManager(cfg.checkpoint)
        self.train_loader = create_dataloader(
            train_bin,
            train_idx,
            cfg.model.seq_len,
            cfg.micro_batch_size,
            seed=cfg.seed,
            num_workers=cfg.data_workers,
        )
        self.val_loader = None
        if val_bin and val_idx:
            self.val_loader = create_dataloader(
                val_bin,
                val_idx,
                cfg.model.seq_len,
                cfg.micro_batch_size,
                seed=cfg.seed + 1,
                num_workers=max(1, cfg.data_workers // 2),
            )
        self.start_step = 0
        if cfg.checkpoint.resume_path:
            ckpt_path = Path(cfg.checkpoint.resume_path)
            if ckpt_path.exists():
                ckpt = load_checkpoint(ckpt_path, self.model, self.optimizer)
                self.start_step = int(ckpt.get("step", 0))
        set_seed(cfg.seed)

    def _run_eval(self) -> Tuple[float, float]:
        if self.val_loader is None:
            return float("nan"), float("nan")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _, loss, _ = self.model(x, y)
                total_loss += loss.item() * x.numel()
                total_tokens += x.numel()
                batches += 1
                if batches >= self.cfg.eval_batches:
                    break
        if total_tokens == 0:
            return float("nan"), float("nan")
        avg_loss = total_loss / total_tokens
        return avg_loss, bits_per_byte(avg_loss)

    def fit(self) -> None:
        accum_steps = self.cfg.grad_accum_steps
        loader_iter = iter(self.train_loader)
        progress = tqdm(
            range(self.start_step, self.cfg.scheduler.max_steps),
            initial=self.start_step,
            total=self.cfg.scheduler.max_steps,
            dynamic_ncols=True,
            desc="train",
        )
        for step in progress:
            lr = cosine_with_warmup(step, self.cfg.scheduler, self.cfg.optimizer.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(accum_steps):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    x, y = next(loader_iter)
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.autocast(**self.autocast_kwargs):
                    _, loss, _ = self.model(x, y)
                    loss = loss / accum_steps
                losses.append(loss.item())
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            progress.set_postfix({"lr": f"{lr:.3e}", "loss": f"{sum(losses):.4f}"})
            if step % self.cfg.logging.log_interval == 0:
                print(f"step {step} loss {sum(losses):.4f} lr {lr:.5f}")
            if self.cfg.eval_interval and step % self.cfg.eval_interval == 0 and self.val_loader is not None:
                val_loss, val_bpb = self._run_eval()
                print(f"eval step {step} loss {val_loss:.4f} bpb {val_bpb:.4f}")
            if step % self.cfg.checkpoint.save_interval == 0 and step > self.start_step:
                self.manager.save(self.model, self.optimizer, self.cfg, step)
        self.manager.save(self.model, self.optimizer, self.cfg, self.cfg.scheduler.max_steps)


__all__ = ["Trainer"]
