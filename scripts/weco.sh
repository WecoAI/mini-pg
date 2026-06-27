#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="config.yaml"

config_value() {
  local key="$1"
  local default="$2"
  uv run python - "$CONFIG_FILE" "$key" "$default" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
target_key = sys.argv[2]
default = sys.argv[3]


def unquote(value: str) -> str:
    value = value.strip()
    if value in {"", "null", "None"}:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if not path.exists():
    print(default)
    raise SystemExit

for raw_line in path.read_text().splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if not line or ":" not in line:
        continue
    key, value = line.split(":", 1)
    if key.strip() == target_key:
        print(unquote(value))
        raise SystemExit

print(default)
PY
}

SOURCE="$(config_value weco_source solution.py)"
INSTRUCTIONS="$(config_value weco_instructions instructions/weco_instructions.md)"
METRIC="$(config_value weco_metric val_bpb)"
GOAL="$(config_value weco_goal minimize)"
STEPS="$(config_value weco_steps 20)"
EVAL_TIMEOUT="$(config_value weco_eval_timeout 210)"
MODEL="$(config_value weco_model "")"

require_nonempty() {
  local key="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "$CONFIG_FILE: $key must not be empty." >&2
    exit 2
  fi
}

require_positive_int() {
  local key="$1"
  local value="$2"
  case "$value" in
    "" | *[!0-9]*)
      echo "$CONFIG_FILE: $key must be a positive integer." >&2
      exit 2
      ;;
  esac
  if (( value < 1 )); then
    echo "$CONFIG_FILE: $key must be a positive integer." >&2
    exit 2
  fi
}

require_nonempty weco_source "$SOURCE"
require_nonempty weco_instructions "$INSTRUCTIONS"
require_nonempty weco_metric "$METRIC"
require_nonempty weco_goal "$GOAL"
require_positive_int weco_steps "$STEPS"
require_positive_int weco_eval_timeout "$EVAL_TIMEOUT"

case "$GOAL" in
  minimize | maximize) ;;
  *)
    echo "$CONFIG_FILE: weco_goal must be minimize or maximize." >&2
    exit 2
    ;;
esac

if [[ ! -f "$SOURCE" ]]; then
  echo "$CONFIG_FILE: weco_source does not exist: $SOURCE" >&2
  exit 2
fi

if [[ ! -f "$INSTRUCTIONS" ]]; then
  echo "$CONFIG_FILE: weco_instructions does not exist: $INSTRUCTIONS" >&2
  exit 2
fi

printf -v SOURCE_ARG "%q" "$SOURCE"
EVAL_COMMAND="./scripts/weco_eval.sh $SOURCE_ARG"

cmd=(
  uv run weco run
  --source "$SOURCE"
  --eval-command "$EVAL_COMMAND"
  --metric "$METRIC"
  --goal "$GOAL"
  --steps "$STEPS"
  --eval-timeout "$EVAL_TIMEOUT"
  --save-logs
  --additional-instructions "$INSTRUCTIONS"
  --output plain
  --no-open
)

if [[ -n "$MODEL" ]]; then
  cmd+=(--model "$MODEL")
fi

"${cmd[@]}"
