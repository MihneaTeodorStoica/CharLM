#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
from src.tinychar.model import TinyCharModel
from src.tinychar.configs import TrainConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location='cpu')
conf = TrainConfig(**ckpt['config'])
model = TinyCharModel(conf)
model.load_state_dict(ckpt['model'])
model.eval()
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save({'model': qmodel.state_dict(), 'config': conf.__dict__}, args.out)
PY
