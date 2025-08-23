# CharLM

Lightweight character/byte-level language model implemented in PyTorch.

## Features
- Byte-level vocabulary (256 symbols)
- RMSNorm + SwiGLU MLP
- RoPE positional embeddings
- GQA attention with FlashAttention path
- Training/evaluation scripts and Docker environment

## Quickstart
```bash
make data-pull        # obtain training data via DVC
make train            # train default 50M model
make eval             # compute validation bpb
make sample PROMPT="Hello"  # generate text
```

If no DVC remote is configured, `make data-pull` downloads a small
public-domain sample (Tiny Shakespeare) and prepares memmap binaries in
`data/out/` so the model can be exercised without external datasets.

## Troubleshooting
- Ensure PyTorch \>=2.3 with CUDA 12.4 for GPU acceleration.
- For CPU-only testing use wheels from `--index-url https://download.pytorch.org/whl/cpu`.
