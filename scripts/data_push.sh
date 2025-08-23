#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path>" >&2
  exit 1
fi

path="$1"
dvc add "$path"
dvc push "$path.dvc"
