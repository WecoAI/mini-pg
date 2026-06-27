"""Validate a Mini PG solution file without launching training."""

from __future__ import annotations

import argparse
import json

from mini_pg.config import load_config_for_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="solution.py")
    args = parser.parse_args()
    cfg = load_config_for_check(args.source)
    print(json.dumps({"status": "ok", "config": cfg}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
