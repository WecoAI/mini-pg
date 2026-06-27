"""Modal entrypoint for Mini Parameter Golf evaluation."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import modal

APP_NAME = "mini-pg-workshop"
TRAIN_TIMEOUT_SECONDS = 60
EVAL_TIMEOUT_SECONDS = 60
SCORE_MODAL_TIMEOUT_SECONDS = 180
SMOKE_MODAL_TIMEOUT_SECONDS = 180
UNTRUSTED_METRIC_PATTERN = re.compile(r"\b(val_bpb|runtime_s|artifact_bytes)\s*:")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install("torch>=2.6.0", "numpy")
    .add_local_python_source("mini_pg")
)

app = modal.App(APP_NAME)


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _sanitize_untrusted_output(value: str | bytes | None) -> str:
    text = _coerce_output(value)
    return UNTRUSTED_METRIC_PATTERN.sub(r"\1_from_candidate_log:", text)


def _phase_buggy_result(
    *,
    reason: str,
    phase: str,
    started_at: float,
    phase_runtime_s: float,
    stdout: str | bytes | None = "",
    stderr: str | bytes | None = "",
    timeout_seconds: int | None = None,
    returncode: int | None = None,
) -> dict:
    result = {
        "status": "buggy",
        "reason": reason,
        "phase": phase,
        "runtime_s": time.perf_counter() - started_at,
        f"{phase}_runtime_s": phase_runtime_s,
        "stdout_tail": _sanitize_untrusted_output(stdout)[-4000:],
        "stderr_tail": _sanitize_untrusted_output(stderr)[-4000:],
    }
    if timeout_seconds is not None:
        result["timeout_seconds"] = timeout_seconds
    if returncode is not None:
        result["returncode"] = returncode
    return result


def _run_phase_command(
    cmd: list[str], *, phase: str, timeout_seconds: int, started_at: float
) -> tuple[subprocess.CompletedProcess[str] | None, dict | None]:
    phase_started_at = time.perf_counter()
    process = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return None, _phase_buggy_result(
            reason=f"{phase}_timeout",
            phase=phase,
            started_at=started_at,
            phase_runtime_s=time.perf_counter() - phase_started_at,
            stdout=stdout or getattr(exc, "stdout", None) or getattr(exc, "output", None),
            stderr=stderr or exc.stderr,
            timeout_seconds=timeout_seconds,
        )

    phase_runtime_s = time.perf_counter() - phase_started_at
    completed = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        return None, _phase_buggy_result(
            reason=f"{phase}_failed",
            phase=phase,
            started_at=started_at,
            phase_runtime_s=phase_runtime_s,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    return completed, None


def _run_eval(
    source_text: str,
    split: str,
    nproc: int,
    *,
    smoke: bool = False,
    train_timeout_seconds: int = TRAIN_TIMEOUT_SECONDS,
    eval_timeout_seconds: int = EVAL_TIMEOUT_SECONDS,
) -> dict:
    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_path = tmp_path / "solution.py"
        checkpoint_path = tmp_path / "checkpoint.pt"
        train_output_path = tmp_path / "train_result.json"
        output_path = tmp_path / "result.json"
        source_path.write_text(source_text)

        train_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "-m",
            "mini_pg.train",
            "--source",
            str(source_path),
            "--split",
            split,
            "--phase",
            "train",
            "--checkpoint",
            str(checkpoint_path),
            "--output-json",
            str(train_output_path),
        ]
        if smoke:
            train_cmd.append("--smoke")
        train_completed, error_result = _run_phase_command(
            train_cmd, phase="train", timeout_seconds=train_timeout_seconds, started_at=started_at
        )
        if error_result is not None:
            return error_result
        train_phase_end_s = (
            time.perf_counter() - started_at if train_completed is not None else None
        )
        if train_output_path.exists():
            train_result = json.loads(train_output_path.read_text())
            if train_result.get("status") != "train_ok":
                train_result.setdefault("phase", "train")
                train_result["runtime_s"] = time.perf_counter() - started_at
                return train_result

        if not checkpoint_path.exists():
            return {
                "status": "buggy",
                "reason": "missing_train_checkpoint",
                "phase": "train",
                "runtime_s": time.perf_counter() - started_at,
            }

        eval_cmd = [
            sys.executable,
            "-m",
            "mini_pg.train",
            "--source",
            str(source_path),
            "--split",
            split,
            "--phase",
            "eval",
            "--checkpoint",
            str(checkpoint_path),
            "--output-json",
            str(output_path),
        ]
        if smoke:
            eval_cmd.append("--smoke")
        _, error_result = _run_phase_command(
            eval_cmd, phase="eval", timeout_seconds=eval_timeout_seconds, started_at=started_at
        )
        if error_result is not None:
            return error_result
        if not output_path.exists():
            return {
                "status": "buggy",
                "reason": "missing_result_json",
                "phase": "eval",
                "runtime_s": time.perf_counter() - started_at,
            }
        result = json.loads(output_path.read_text())
        result["runtime_s"] = time.perf_counter() - started_at
        if train_phase_end_s is not None:
            result["train_phase_end_s"] = train_phase_end_s
        return result


@app.function(
    image=image, gpu="H100:8", cpu=32.0, memory=131_072, timeout=SCORE_MODAL_TIMEOUT_SECONDS
)
def evaluate_solution_8h100(source_text: str, split: str = "public") -> dict:
    return _run_eval(source_text, split, nproc=8)


@app.function(image=image, cpu=4.0, memory=16_384, timeout=SMOKE_MODAL_TIMEOUT_SECONDS)
def evaluate_solution_cpu_smoke(source_text: str, split: str = "public") -> dict:
    return _run_eval(source_text, split, nproc=1, smoke=True)


@app.local_entrypoint()
def main(
    source: str = "solution.py",
    split: str = "public",
    cpu_smoke: bool = False,
    output_json: str = "",
) -> None:
    source_text = Path(source).read_text()
    if cpu_smoke:
        result = evaluate_solution_cpu_smoke.remote(source_text, split)
    else:
        result = evaluate_solution_8h100.remote(source_text, split)

    print(json.dumps(result, indent=2, sort_keys=True))
    if "val_bpb" in result:
        print(f"val_bpb: {float(result['val_bpb']):.8f}")
    else:
        print(f"status: {result.get('status', 'unknown')}")
        if result.get("reason"):
            print(f"reason: {result['reason']}")
    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
