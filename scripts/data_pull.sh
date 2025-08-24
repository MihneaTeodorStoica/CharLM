#!/usr/bin/env bash
set -euo pipefail

# If this repository is initialized with DVC, pull data from the remote.
# Otherwise generate a small public-domain sample dataset for quick tests.
if [ -d .dvc ] && command -v dvc >/dev/null 2>&1; then
  dvc pull "$@"
else
  echo "DVC repo not found; creating tiny sample dataset"
  python scripts/prepare_sample_data.py
fi
