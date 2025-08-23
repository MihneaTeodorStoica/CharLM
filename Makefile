RECIPEPREFIX := >
.PHONY: setup data-pull data-push train eval sample export-int8 lint test docker-up

PYTHON=python

setup:
> pip install -r requirements.txt

data-pull:
> scripts/data_pull.sh

data-push:
> scripts/data_push.sh data/out

train:
> $(PYTHON) -m src.train --config configs/small-50M.yaml

eval:
> $(PYTHON) -m src.eval_bpb --config configs/small-50M.yaml

sample:
> $(PYTHON) -m src.generate --checkpoint checkpoints/last.pt --prompt "$(PROMPT)" --max_new 100

export-int8:
> scripts/export_int8.sh --checkpoint checkpoints/best.pt --out checkpoints/model-int8.pt

lint:
> pre-commit run --files $(shell git ls-files '*.py')

test:
> pytest -q

docker-up:
> docker compose -f docker/compose.yaml up -d
