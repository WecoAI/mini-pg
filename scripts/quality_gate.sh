#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SOURCE="${1:-solution.py}"

if [[ $# -ge 2 ]]; then
  uv run python -m mini_pg.quality_gate "$SOURCE" "$2"
else
  uv run python -m mini_pg.quality_gate "$SOURCE"
fi
