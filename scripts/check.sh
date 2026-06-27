#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SOURCE="${1:-solution.py}"

uv run python -m mini_pg.check_solution "$SOURCE"

