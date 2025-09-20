# CharLM

A modern byte-level language-modeling toolkit built with PyTorch 2.x. The project provides a clean, well-tested reference implementation featuring grouped-query attention, rotary embeddings, and production-friendly tooling for training, evaluation, and sampling.

## Highlights
- Structured configuration with validation (nested YAML or CLI overrides)
- Device-aware training loop with safe AMP, gradient accumulation, and resumable checkpoints
- Deterministic packed memmap dataset with per-worker seeding
- KV-cache aware sampler with context window management
- CLI utilities for training, evaluation (bits-per-byte), and text generation

## Quickstart
```bash
make data-pull                 # fetch sample dataset (or DVC remote if configured)
make train CONFIG=configs/micro.yaml  # train a tiny model in minutes on CPU/GPU
make eval CHECKPOINT=checkpoints/step-200.pt DATA=data/out/val.bin INDEX=data/out/val.idx
make sample PROMPT="Hello" CHECKPOINT=checkpoints/step-200.pt
```

### Direct CLI usage
```bash
python -m src.train --config configs/small-50M.yaml
python -m src.eval_bpb --checkpoint checkpoints/last.pt --data data/out/val.bin --index data/out/val.idx
python -m src.generate --checkpoint checkpoints/last.pt --prompt "Once upon a byte" --max-new 128
```

## Configuration
Training is controlled by `TrainingConfig` (see `configs/`). Key sections:
- `model`: architecture knobs (`d_model`, `n_layer`, `n_head`, etc.)
- `optimizer`: AdamW hyperparameters and fused-mode toggle
- `scheduler`: cosine schedule with warmup
- `checkpoint`: output directory, save cadence, retention
- `logging`: console cadence and optional Weights & Biases metadata
- Top-level scalars: `batch_size_tokens`, `micro_batch_size`, `precision`, `compile`, etc.

All configs are validated at load time: incompatible head shapes, invalid batch math, or unsupported precision settings fail fast with helpful errors. CLI flags like `--max-steps`, `--precision`, or `--no-val` provide quick overrides without editing YAML.

## Data
`data_pull.sh` fetches Tiny Shakespeare when no DVC remote is present and writes byte-level memmaps (`.bin` + `.idx`) ready for the `PackedDataset`. The dataset packs documents with separator bytes and seeds every DataLoader worker independently for reproducibility.

## Sampling & Quantization
- Generation respects a configurable `max_context_tokens`, preventing runaway KV caches during extended sampling sessions.
- The `scripts/export_int8.sh` stub remains a starting point for future quantized export. Update it to use the model wrapper defined in `charlm` if you pursue production quantization.

## Testing
Pytest cases cover dataset packing, attention cache behaviour, sampler determinism, and configuration validation. Run them with:
```bash
make test
```

## Roadmap
- Multi-GPU/distributed training helpers
- Optional state-space blocks (`charlm.layers` placeholder)
- Export utilities (ONNX / TorchScript / int8) with automated validation
