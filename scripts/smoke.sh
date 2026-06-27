#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SOURCE="${1:-solution.py}"

uv run modal run mini_pg/modal_app.py --source "$SOURCE" --cpu-smoke
