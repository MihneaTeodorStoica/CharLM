#!/usr/bin/env bash
set -euo pipefail

docker compose -f docker/compose.yaml run --rm train python -m src.eval_bpb "$@"
