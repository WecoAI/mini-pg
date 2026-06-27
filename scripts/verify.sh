#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:-solution.py}"
mkdir -p results

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
source_name="$(basename "$SOURCE")"
safe_source_name="${source_name//[^A-Za-z0-9_.-]/_}"
OUTPUT="${2:-results/score-${timestamp}-${safe_source_name}.txt}"

./scripts/score.sh "$SOURCE" | tee "$OUTPUT"
./scripts/quality_gate.sh "$SOURCE" "$OUTPUT"
printf 'score_output: %s\n' "$OUTPUT"
