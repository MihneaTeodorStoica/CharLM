# Deployment

The runtime container (`docker/Dockerfile.runtime`) provides a slim image for inference.

## CLI
```
python -m src.generate --checkpoint ckpt.pt --prompt "Hello" --max_new 32
```

## Python API
```
from src.tinychar.model import TinyCharModel
from src.tinychar.configs import TrainConfig
model = TinyCharModel(TrainConfig())
...
```

Supports dynamic int8 quantization via `scripts/export_int8.sh`.
