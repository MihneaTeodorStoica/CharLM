#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import argparse
import torch

from charlm.config import TrainingConfig
from charlm.model import CharLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")
config = TrainingConfig.from_mapping(ckpt["config"])
model = CharLM(config.model)
model.load_state_dict(ckpt["model"])
model.eval()

quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

payload = {
    "model": quantized.state_dict(),
    "config": config.to_dict(),
}

torch.save(payload, args.out)
PY
