#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uvx >/dev/null 2>&1; then
  UVX=(uvx)
elif [[ -x /opt/homebrew/bin/uvx ]]; then
  UVX=(/opt/homebrew/bin/uvx)
else
  echo "uvx not found. Install uv or add uvx to PATH." >&2
  exit 127
fi

if [[ "${1:-}" == "--fix" ]]; then
  "${UVX[@]}" ruff check --fix .
  "${UVX[@]}" ruff format .
else
  "${UVX[@]}" ruff check .
  "${UVX[@]}" ruff format --check .
fi
