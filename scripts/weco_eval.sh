#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:-solution.py}"

tmp_check="$(mktemp)"
tmp_quality="$(mktemp)"
tmp_json="$(mktemp)"
tmp_log="$(mktemp)"
trap 'rm -f "$tmp_check" "$tmp_quality" "$tmp_json" "$tmp_log"' EXIT

emit_buggy() {
  local reason="$1"
  local phase="${2:-preflight}"
  uv run python - "$reason" "$phase" <<'PY'
from __future__ import annotations

import json
import sys

reason = sys.argv[1]
phase = sys.argv[2]
result = {
    "status": "buggy",
    "reason": reason,
    "phase": phase,
}
print(json.dumps(result, sort_keys=True))
print("status: buggy")
print(f"reason: {reason}")
PY
}

if ! ./scripts/check.sh "$SOURCE" >"$tmp_check" 2>&1; then
  emit_buggy "check_failed" "check"
  exit 0
fi

quality_status=0
./scripts/quality_gate.sh "$SOURCE" >"$tmp_quality" 2>&1 || quality_status=$?
cat "$tmp_quality"
if (( quality_status != 0 )); then
  if (( quality_status == 2 )); then
    emit_buggy "quality_gate_review_required" "quality_gate"
  else
    emit_buggy "quality_gate_failed" "quality_gate"
  fi
  exit 0
fi

modal_status=0
uv run modal run mini_pg/modal_app.py --source "$SOURCE" --output-json "$tmp_json" >"$tmp_log" 2>&1 || modal_status=$?

if (( modal_status != 0 )); then
  emit_buggy "modal_command_failed" "modal"
  exit 0
fi

uv run python - "$tmp_json" "$tmp_log" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
log_path = Path(sys.argv[2])

try:
    result = json.loads(path.read_text())
except Exception:
    result = {}

val_bpb = result.get("val_bpb")
status = result.get("status", "unknown")
if status == "ok" and isinstance(val_bpb, int | float) and math.isfinite(float(val_bpb)):
    print(log_path.read_text(), end="")
    raise SystemExit(0)

reason = result.get("reason") or status
buggy_result = {
    "status": "buggy",
    "reason": str(reason),
    "phase": result.get("phase", "eval"),
}
if status != "unknown":
    buggy_result["original_status"] = status
if "artifact_bytes" in result:
    buggy_result["artifact_bytes"] = result["artifact_bytes"]
if "artifact_cap_bytes" in result:
    buggy_result["artifact_cap_bytes"] = result["artifact_cap_bytes"]
print(json.dumps(buggy_result, sort_keys=True))
print("status: buggy")
print(f"reason: {reason}")
PY
