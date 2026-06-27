"""Configuration validation for Mini Parameter Golf solutions."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "model_dim": 192,
    "num_layers": 4,
    "num_heads": 4,
    "mlp_mult": 2,
    "seq_len": 512,
    "dropout": 0.0,
    "train_steps": 80,
    "train_batch_tokens": 32_768,
    "learning_rate": 0.002,
    "weight_decay": 0.08,
    "warmup_steps": 5,
    "warmdown_steps": 20,
    "grad_clip": 1.0,
    "q_clip_percentile": 0.9995,
}

INT_RANGES = {
    "model_dim": (128, 768),
    "num_layers": (2, 12),
    "num_heads": (1, 12),
    "mlp_mult": (1, 4),
    "seq_len": (128, 2048),
    "train_steps": (50, 900),
    "train_batch_tokens": (32_768, 1_572_864),
    "warmup_steps": (0, 400),
    "warmdown_steps": (0, 800),
}

FLOAT_RANGES = {
    "dropout": (0.0, 0.25),
    "learning_rate": (1e-5, 0.02),
    "weight_decay": (0.0, 0.5),
    "grad_clip": (0.0, 5.0),
    "q_clip_percentile": (0.95, 1.0),
}


class StaticConfigUnavailable(ValueError):
    """Raised when get_config cannot be read without executing the module."""


def load_solution_module(path: str | Path) -> ModuleType:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("mini_pg_candidate_solution", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import solution module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config_from_module(module: ModuleType) -> dict[str, Any]:
    if not hasattr(module, "get_config"):
        raise ValueError("solution.py must define get_config()")
    raw = module.get_config()
    if not isinstance(raw, dict):
        raise ValueError("get_config() must return a dict")
    return validate_config(raw)


def load_config(path: str | Path) -> dict[str, Any]:
    return load_config_from_module(load_solution_module(path))


def load_config_static(path: str | Path) -> dict[str, Any]:
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_config":
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    try:
                        raw = ast.literal_eval(stmt.value)
                    except (ValueError, TypeError) as exc:
                        raise StaticConfigUnavailable("get_config() is not a literal dict") from exc
                    if not isinstance(raw, dict):
                        raise ValueError("get_config() must return a dict")
                    return validate_config(raw)
            raise StaticConfigUnavailable("get_config() has no literal return")
    raise ValueError("solution.py must define get_config()")


def load_config_for_check(path: str | Path) -> dict[str, Any]:
    try:
        return load_config_static(path)
    except StaticConfigUnavailable:
        return load_config(path)


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULT_CONFIG)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(raw)

    for key, (lo, hi) in INT_RANGES.items():
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{key} must be an int")
        if not lo <= value <= hi:
            raise ValueError(f"{key}={value} outside [{lo}, {hi}]")

    for key, (lo, hi) in FLOAT_RANGES.items():
        value = cfg[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{key} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key}={value} outside [{lo}, {hi}]")
        cfg[key] = value

    if cfg["model_dim"] % cfg["num_heads"] != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if (cfg["model_dim"] // cfg["num_heads"]) % 2 != 0:
        raise ValueError("attention head dimension must be even")
    if cfg["train_batch_tokens"] % cfg["seq_len"] != 0:
        raise ValueError("train_batch_tokens must be divisible by seq_len")
    if cfg["warmup_steps"] + cfg["warmdown_steps"] >= cfg["train_steps"]:
        raise ValueError("warmup_steps + warmdown_steps must be < train_steps")

    return cfg
