"""Distributed train/eval entrypoint used inside Modal."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from mini_pg.config import load_config_from_module, load_solution_module
from mini_pg.data import VOCAB_SIZE, build_split
from mini_pg.model import ByteGPT, cosine_schedule, estimate_param_count
from mini_pg.quantize import quantized_roundtrip as default_quantized_roundtrip

ARTIFACT_CAP_BYTES = 4_000_000
TRAIN_BYTES = 24_000_000
PUBLIC_VAL_BYTES = 500_000
SMOKE_TRAIN_BYTES = 200_000
SMOKE_VAL_BYTES = 50_000
TRAIN_BUDGET_SECONDS = 60.0
EVAL_BUDGET_SECONDS = 60.0


class PhaseTimeout(RuntimeError):
    """Raised when a scored train or eval phase exceeds its budget."""


def _distributed_setup() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend)
    return rank, world_size, device


def _distributed_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _any_rank_true(value: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    flag = torch.tensor(1 if value else 0, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _check_deadline(deadline_s: float | None, reason: str) -> None:
    if deadline_s is not None and time.perf_counter() > deadline_s:
        raise PhaseTimeout(reason)


def make_batch(
    tokens: Tensor,
    *,
    seq_len: int,
    batch_tokens: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    batch_size = batch_tokens // seq_len
    max_start = tokens.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator, device=tokens.device)
    rows = [tokens[start : start + seq_len + 1] for start in starts.tolist()]
    block = torch.stack(rows).to(device=device, non_blocking=True)
    return block[:, :-1].long(), block[:, 1:].long()


@torch.no_grad()
def evaluate_bpb(
    model: torch.nn.Module,
    tokens: Tensor,
    seq_len: int,
    device: torch.device,
    *,
    deadline_s: float | None = None,
) -> float:
    model.eval()
    losses = []
    batch_seqs = 64 if device.type == "cuda" else 4
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    for start in range(0, usable, seq_len * batch_seqs):
        _check_deadline(deadline_s, "eval_timeout")
        end = min(start + seq_len * batch_seqs, usable)
        block = tokens[start : end + 1].to(device=device, non_blocking=True)
        actual = (block.numel() - 1) // seq_len
        if actual <= 0:
            continue
        x = block[:-1].reshape(actual, seq_len).long()
        y = block[1:].reshape(actual, seq_len).long()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        losses.append((float(loss.item()), int(y.numel())))
        _check_deadline(deadline_s, "eval_timeout")
    model.train()
    total_tokens = sum(count for _, count in losses)
    mean_loss = sum(loss * count for loss, count in losses) / total_tokens
    return mean_loss / math.log(2.0)


def _smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["train_steps"] = min(cfg["train_steps"], 5)
    smoke_batch_seqs = max(1, min(32, 16_384 // cfg["seq_len"]))
    cfg["train_batch_tokens"] = cfg["seq_len"] * smoke_batch_seqs
    cfg["warmup_steps"] = 1
    cfg["warmdown_steps"] = 1
    return cfg


def _artifact_source_bytes(source: Path) -> tuple[int, dict[str, int]]:
    source_sizes = {"solution.py": len(source.read_bytes())}
    return sum(source_sizes.values()), source_sizes


def _default_model(cfg: dict[str, Any]) -> torch.nn.Module:
    return ByteGPT(
        vocab_size=VOCAB_SIZE,
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        mlp_mult=cfg["mlp_mult"],
        seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
    )


def _build_model(solution: ModuleType, cfg: dict[str, Any]) -> torch.nn.Module:
    build_model = getattr(solution, "build_model", None)
    if build_model is not None:
        model = build_model(vocab_size=VOCAB_SIZE, config=cfg)
        if model is None:
            return _default_model(cfg)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("build_model() must return a torch.nn.Module or None")
        return model
    return _default_model(cfg)


def _quantized_roundtrip(
    solution: ModuleType, model: torch.nn.Module, clip_percentile: float
) -> tuple[dict[str, Tensor], int]:
    quantize = getattr(solution, "quantized_roundtrip", None)
    result = None
    if quantize is not None:
        result = quantize(model, clip_percentile)
    if result is None:
        result = default_quantized_roundtrip(model, clip_percentile)
    state_dict, artifact_bytes = result
    if not isinstance(state_dict, dict):
        raise TypeError("quantized_roundtrip() must return a state_dict as its first value")
    if not isinstance(artifact_bytes, int):
        raise TypeError("quantized_roundtrip() must return artifact_bytes as an int")
    return state_dict, artifact_bytes


def _buggy_result(
    *,
    reason: str,
    runtime_s: float,
    split: str,
    smoke: bool,
    cfg: dict[str, Any],
    param_count: int,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "buggy",
        "reason": reason,
        "metric": "val_bpb",
        "runtime_s": runtime_s,
        "split": split,
        "smoke": smoke,
        "config": cfg,
        "param_count": param_count,
    }
    if timeout_seconds is not None:
        result["timeout_seconds"] = timeout_seconds
    return result


def _emit_result(result: dict[str, Any], output_json: Path | None) -> None:
    print(json.dumps(result, sort_keys=True), flush=True)
    if "val_bpb" in result:
        print(f"val_bpb: {result['val_bpb']:.8f}", flush=True)
    else:
        print(f"status: {result.get('status', 'unknown')}", flush=True)
        if result.get("reason"):
            print(f"reason: {result['reason']}", flush=True)
    if "runtime_s" in result:
        print(f"runtime_s: {result['runtime_s']:.3f}", flush=True)
    if "artifact_bytes" in result:
        print(f"artifact_bytes: {result['artifact_bytes']}", flush=True)
    if output_json is not None:
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def run_training(
    source: Path,
    split: str,
    output_json: Path | None,
    *,
    smoke: bool = False,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    rank, world_size, device = _distributed_setup()
    try:
        solution = load_solution_module(source)
        cfg = load_config_from_module(solution)
        if smoke:
            cfg = _smoke_config(cfg)
        torch.manual_seed(1337)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1337)
            torch.backends.cuda.matmul.allow_tf32 = True

        train_bytes = SMOKE_TRAIN_BYTES if smoke else TRAIN_BYTES
        train_tokens = torch.frombuffer(
            bytearray(build_split("train", train_bytes)), dtype=torch.uint8
        )
        val_bytes = SMOKE_VAL_BYTES if smoke else PUBLIC_VAL_BYTES
        val_tokens = torch.frombuffer(bytearray(build_split(split, val_bytes)), dtype=torch.uint8)

        model = _build_model(solution, cfg).to(device)
        param_count = estimate_param_count(model)
        if world_size > 1:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=cfg["weight_decay"],
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(10_000 + rank)

        local_batch_seqs = max(1, cfg["train_batch_tokens"] // (world_size * cfg["seq_len"]))
        local_batch_tokens = local_batch_seqs * cfg["seq_len"]
        train_started_at = time.perf_counter()
        for step in range(cfg["train_steps"]):
            lr_mult = cosine_schedule(
                step, cfg["train_steps"], cfg["warmup_steps"], cfg["warmdown_steps"]
            )
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rate"] * lr_mult

            x, y = make_batch(
                train_tokens,
                seq_len=cfg["seq_len"],
                batch_tokens=local_batch_tokens,
                device=device,
                generator=generator,
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and (step == 0 or (step + 1) % 100 == 0):
                print(f"step:{step + 1} train_loss:{float(loss.item()):.6f}", flush=True)

            train_timed_out = time.perf_counter() - train_started_at > TRAIN_BUDGET_SECONDS
            if _any_rank_true(train_timed_out, device):
                result = {}
                if rank == 0:
                    result = _buggy_result(
                        reason="train_timeout",
                        runtime_s=time.perf_counter() - start_time,
                        split=split,
                        smoke=smoke,
                        cfg=cfg,
                        param_count=param_count,
                        timeout_seconds=TRAIN_BUDGET_SECONDS,
                    )
                    _emit_result(result, output_json)
                return result

        if world_size > 1:
            dist.barrier()

        result: dict[str, Any] = {}
        if rank == 0:
            raw_model = model.module if isinstance(model, DDP) else model
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                train_runtime_s = time.perf_counter() - start_time
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in raw_model.state_dict().items()
                        },
                        "config": cfg,
                        "param_count": param_count,
                        "train_runtime_s": train_runtime_s,
                    },
                    checkpoint,
                )
                result = {
                    "status": "train_ok",
                    "phase": "train",
                    "runtime_s": train_runtime_s,
                    "split": split,
                    "smoke": smoke,
                    "config": cfg,
                    "param_count": param_count,
                }
                _emit_result(result, output_json)
                return result
            eval_deadline_s = time.perf_counter() + EVAL_BUDGET_SECONDS
            try:
                state_dict, artifact_bytes = _quantized_roundtrip(
                    solution, raw_model, cfg["q_clip_percentile"]
                )
                _check_deadline(eval_deadline_s, "eval_timeout")
                source_bytes, source_file_bytes = _artifact_source_bytes(source)
                artifact_bytes += source_bytes
                raw_model.load_state_dict(state_dict, strict=True)
                _check_deadline(eval_deadline_s, "eval_timeout")
                val_bpb = evaluate_bpb(
                    raw_model, val_tokens, cfg["seq_len"], device, deadline_s=eval_deadline_s
                )
            except PhaseTimeout as exc:
                result = _buggy_result(
                    reason=str(exc),
                    runtime_s=time.perf_counter() - start_time,
                    split=split,
                    smoke=smoke,
                    cfg=cfg,
                    param_count=param_count,
                    timeout_seconds=EVAL_BUDGET_SECONDS,
                )
                _emit_result(result, output_json)
                return result
            runtime_s = time.perf_counter() - start_time
            if artifact_bytes > ARTIFACT_CAP_BYTES:
                result = {
                    "status": "buggy",
                    "reason": "artifact_cap_exceeded",
                    "phase": "eval",
                    "runtime_s": runtime_s,
                    "artifact_bytes": artifact_bytes,
                    "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
                    "source_bytes": source_bytes,
                    "source_file_bytes": source_file_bytes,
                    "param_count": param_count,
                    "split": split,
                    "smoke": smoke,
                    "config": cfg,
                }
                _emit_result(result, output_json)
                return result
            result = {
                "status": "ok",
                "metric": "val_bpb",
                "val_bpb": val_bpb,
                "runtime_s": runtime_s,
                "artifact_bytes": artifact_bytes,
                "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
                "source_bytes": source_bytes,
                "source_file_bytes": source_file_bytes,
                "param_count": param_count,
                "split": split,
                "smoke": smoke,
                "config": cfg,
            }
            _emit_result(result, output_json)
        return result
    finally:
        _distributed_cleanup()


def run_eval_phase(
    source: Path, split: str, output_json: Path | None, checkpoint: Path, *, smoke: bool = False
) -> dict[str, Any]:
    start_time = time.perf_counter()
    rank, world_size, device = _distributed_setup()
    try:
        if world_size != 1:
            raise ValueError("eval phase must run with one process")
        solution = load_solution_module(source)
        cfg = load_config_from_module(solution)
        if smoke:
            cfg = _smoke_config(cfg)
        torch.manual_seed(1337)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1337)
            torch.backends.cuda.matmul.allow_tf32 = True

        val_bytes = SMOKE_VAL_BYTES if smoke else PUBLIC_VAL_BYTES
        val_tokens = torch.frombuffer(bytearray(build_split(split, val_bytes)), dtype=torch.uint8)
        model = _build_model(solution, cfg).to(device)
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["state_dict"], strict=True)
        param_count = int(saved.get("param_count", estimate_param_count(model)))

        eval_deadline_s = time.perf_counter() + EVAL_BUDGET_SECONDS
        try:
            state_dict, artifact_bytes = _quantized_roundtrip(
                solution, model, cfg["q_clip_percentile"]
            )
            _check_deadline(eval_deadline_s, "eval_timeout")
            source_bytes, source_file_bytes = _artifact_source_bytes(source)
            artifact_bytes += source_bytes
            model.load_state_dict(state_dict, strict=True)
            _check_deadline(eval_deadline_s, "eval_timeout")
            val_bpb = evaluate_bpb(
                model, val_tokens, cfg["seq_len"], device, deadline_s=eval_deadline_s
            )
        except PhaseTimeout as exc:
            result = _buggy_result(
                reason=str(exc),
                runtime_s=time.perf_counter() - start_time,
                split=split,
                smoke=smoke,
                cfg=cfg,
                param_count=param_count,
                timeout_seconds=EVAL_BUDGET_SECONDS,
            )
            result["phase"] = "eval"
            _emit_result(result, output_json)
            return result

        runtime_s = time.perf_counter() - start_time
        if artifact_bytes > ARTIFACT_CAP_BYTES:
            result = {
                "status": "buggy",
                "reason": "artifact_cap_exceeded",
                "phase": "eval",
                "runtime_s": runtime_s,
                "train_runtime_s": saved.get("train_runtime_s"),
                "artifact_bytes": artifact_bytes,
                "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
                "source_bytes": source_bytes,
                "source_file_bytes": source_file_bytes,
                "param_count": param_count,
                "split": split,
                "smoke": smoke,
                "config": cfg,
            }
            _emit_result(result, output_json)
            return result
        result = {
            "status": "ok",
            "phase": "eval",
            "metric": "val_bpb",
            "val_bpb": val_bpb,
            "runtime_s": runtime_s,
            "train_runtime_s": saved.get("train_runtime_s"),
            "artifact_bytes": artifact_bytes,
            "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
            "source_bytes": source_bytes,
            "source_file_bytes": source_file_bytes,
            "param_count": param_count,
            "split": split,
            "smoke": smoke,
            "config": cfg,
        }
        _emit_result(result, output_json)
        return result
    finally:
        _distributed_cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="solution.py")
    parser.add_argument("--split", choices=["public"], default="public")
    parser.add_argument("--output-json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=["full", "train", "eval"], default="full")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    output_json = Path(args.output_json) if args.output_json else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.phase == "eval":
        if checkpoint is None:
            raise SystemExit("--checkpoint is required for --phase eval")
        run_eval_phase(Path(args.source), args.split, output_json, checkpoint, smoke=args.smoke)
    else:
        run_training(
            Path(args.source),
            args.split,
            output_json,
            smoke=args.smoke,
            checkpoint=checkpoint if args.phase == "train" else None,
        )


if __name__ == "__main__":
    main()
