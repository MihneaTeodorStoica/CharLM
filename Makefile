.RECIPEPREFIX := >
.PHONY: setup data-pull data-push train train-micro eval sample export-int8 lint test docker-up

PYTHON=python
CHECKPOINT ?= checkpoints/last.pt
TRAIN_CONFIG ?= configs/small-50M.yaml
MICRO_CONFIG ?= configs/micro.yaml
EVAL_DATA ?= data/out/val.bin
EVAL_INDEX ?= data/out/val.idx
TRAIN_DATA ?= data/out/train.bin
TRAIN_INDEX ?= data/out/train.idx

setup:
> pip install -r requirements.txt

data-pull:
> scripts/data_pull.sh

data-push:
> scripts/data_push.sh data/out

train:
> PYTHONPATH=$(CURDIR)/src $(PYTHON) -m train --config $(TRAIN_CONFIG) --train-data $(TRAIN_DATA) --train-index $(TRAIN_INDEX)

train-micro:
> PYTHONPATH=$(CURDIR)/src $(PYTHON) -m train --config $(MICRO_CONFIG) --train-data $(TRAIN_DATA) --train-index $(TRAIN_INDEX)

eval:
> PYTHONPATH=$(CURDIR)/src $(PYTHON) -m eval_bpb --checkpoint $(CHECKPOINT) --data $(EVAL_DATA) --index $(EVAL_INDEX)

sample:
> PYTHONPATH=$(CURDIR)/src $(PYTHON) -m generate --checkpoint $(CHECKPOINT) --prompt "$(PROMPT)" --max-new 128

export-int8:
> scripts/export_int8.sh --checkpoint $(CHECKPOINT) --out checkpoints/model-int8.pt

lint:
> pre-commit run --files $(shell git ls-files '*.py')

test:
> pytest -q

docker-up:
> docker compose -f docker/compose.yaml up -d
