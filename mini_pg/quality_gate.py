"""Post-hoc legality and result quality gate for Mini PG submissions."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mini_pg.config import DEFAULT_CONFIG, StaticConfigUnavailable, load_config_static

ARTIFACT_CAP_BYTES = 4_000_000
TRAIN_BUDGET_SECONDS = 60.0
EVAL_BUDGET_SECONDS = 60.0

SAFE_STDLIB_IMPORTS = {
    "__future__",
    "abc",
    "array",
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "enum",
    "functools",
    "heapq",
    "io",
    "itertools",
    "json",
    "math",
    "operator",
    "statistics",
    "struct",
    "types",
    "typing",
    "zlib",
}

BANNED_IMPORTS = {
    "builtins": "dynamic import or interpreter access",
    "ctypes": "native/runtime side effects",
    "ftplib": "network access",
    "glob": "filesystem access",
    "http": "network access",
    "importlib": "dynamic import",
    "inspect": "runtime introspection",
    "mini_pg": "harness internals",
    "mmap": "filesystem access",
    "os": "filesystem, process, or environment access",
    "pathlib": "filesystem access",
    "pkgutil": "dynamic import",
    "requests": "network access",
    "runpy": "dynamic code execution",
    "shutil": "filesystem access",
    "site": "runtime environment access",
    "socket": "network access",
    "ssl": "network access",
    "subprocess": "process execution",
    "sys": "runtime environment access",
    "tempfile": "filesystem access",
    "urllib": "network access",
}

BANNED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "print",
}

BANNED_ATTRS = {
    "environ",
    "exists",
    "expanduser",
    "getenv",
    "glob",
    "iglob",
    "iterdir",
    "listdir",
    "mkdir",
    "open",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "rglob",
    "rmdir",
    "scandir",
    "stat",
    "unlink",
    "walk",
    "write_bytes",
    "write_text",
}

BANNED_NAMES = {
    "OFFICIAL_VAL_SEED",
    "PUBLIC_VAL_BYTES",
    "SMOKE_VAL_BYTES",
    "TRAIN_SEED",
    "build_split",
}

MUTATING_METHODS = {
    "add_",
    "append",
    "clear",
    "copy_",
    "extend",
    "mul_",
    "pop",
    "set_",
    "update",
    "zero_",
}

DATASET_STRING_FRAGMENTS = {
    "Parameter Golf asks for the best language model",
    "A fair benchmark keeps the official validation seed fixed",
    "Modal supplies the GPUs; Weco supplies the search loop",
    "The baseline is intentionally beatable",
    "val_bpb: 1.2345 runtime_s",
    "seed=31",
}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    line: int | None = None


@dataclass
class MethodProfile:
    primary: str
    categories: list[str]
    changed_config_keys: list[str]
    changed_shape_keys: list[str]
    changed_training_keys: list[str]
    has_custom_model: bool
    has_custom_quantization: bool


class StaticGate(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.forward_depth = 0
        self.quantized_roundtrip_depth = 0
        self.literal_artifact_byte_names: list[set[str]] = []

    def error(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(Finding("error", code, message, getattr(node, "lineno", None)))

    def review(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(Finding("review", code, message, getattr(node, "lineno", None)))

    def note(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(Finding("note", code, message, getattr(node, "lineno", None)))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import_root(node, alias.name.split(".", 1)[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.error(node, "relative_import", "Relative imports are not allowed in solution.py.")
            return
        self._check_import_root(node, node.module.split(".", 1)[0])
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in BANNED_NAMES:
            self.error(node, "harness_name", f"`{node.id}` reaches into the fixed harness/data.")
        if self.forward_depth and isinstance(node.ctx, ast.Load) and node.id == "target_ids":
            self.note(
                node,
                "target_ids_branch",
                "`target_ids` branches are ignored by the scorer; prefer returning logits only.",
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_ATTRS:
            self.error(
                node,
                "side_effect_attribute",
                f"`.{node.attr}` suggests filesystem, environment, or side-effect access.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node.func)
        if name in BANNED_CALL_NAMES:
            self.error(node, "dynamic_or_io_call", f"`{name}(...)` is not allowed.")
        if name in {"cross_entropy", "nll_loss"} and self.forward_depth:
            self.note(
                node,
                "forward_loss_branch",
                "Loss branches inside `forward` are ignored by the scorer; verify logits path is primary.",
            )
        if (
            self.forward_depth
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATING_METHODS
            and self._root_name(node.func.value) == "self"
        ):
            self.review(
                node,
                "stateful_forward",
                f"`forward` mutates `self` through `{node.func.attr}`; verify no eval-state adaptation.",
            )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self.quantized_roundtrip_depth and isinstance(node.value, ast.Tuple):
            values = node.value.elts
            if len(values) >= 2 and isinstance(values[1], ast.Constant):
                self.error(
                    node,
                    "literal_artifact_bytes",
                    "quantized_roundtrip() must not return a literal artifact byte count.",
                )
            if (
                len(values) >= 2
                and isinstance(values[1], ast.Name)
                and self.literal_artifact_byte_names
                and values[1].id in self.literal_artifact_byte_names[-1]
            ):
                self.error(
                    node,
                    "literal_artifact_bytes",
                    "quantized_roundtrip() must not return a literal artifact byte count.",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.quantized_roundtrip_depth:
            for target in node.targets:
                self._track_artifact_byte_assignment(target, node.value)
        if self.forward_depth:
            for target in node.targets:
                self._check_forward_assignment(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.quantized_roundtrip_depth and node.value is not None:
            self._track_artifact_byte_assignment(node.target, node.value)
        if self.forward_depth:
            self._check_forward_assignment(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self.quantized_roundtrip_depth and isinstance(node.target, ast.Name):
            self._current_literal_artifact_names().discard(node.target.id)
        if self.forward_depth:
            self._check_forward_assignment(node.target)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.error(node, "global_state", "`global` state is not allowed in submitted solutions.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.error(
            node, "nonlocal_state", "`nonlocal` state is not allowed in submitted solutions."
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        entering_forward = node.name == "forward"
        entering_quantized_roundtrip = node.name == "quantized_roundtrip"
        if entering_forward:
            self.forward_depth += 1
        if entering_quantized_roundtrip:
            self.quantized_roundtrip_depth += 1
            self.literal_artifact_byte_names.append(set())
        self.generic_visit(node)
        if entering_forward:
            self.forward_depth -= 1
        if entering_quantized_roundtrip:
            self.literal_artifact_byte_names.pop()
            self.quantized_roundtrip_depth -= 1

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            for fragment in DATASET_STRING_FRAGMENTS:
                if fragment in node.value:
                    self.error(
                        node,
                        "dataset_literal",
                        "String literal appears to hardcode generated train/validation text.",
                    )
                    break
        self.generic_visit(node)

    def _check_import_root(self, node: ast.AST, root: str) -> None:
        if root in BANNED_IMPORTS:
            self.error(node, "banned_import", f"`{root}` import enables {BANNED_IMPORTS[root]}.")
            return
        if root == "torch" or root in SAFE_STDLIB_IMPORTS:
            return
        if root in sys.stdlib_module_names:
            self.review(
                node,
                "stdlib_import_review",
                f"`{root}` is stdlib but not on the Mini PG safe import allowlist.",
            )
            return
        self.error(node, "external_import", f"`{root}` is not part of PyTorch or the stdlib.")

    def _current_literal_artifact_names(self) -> set[str]:
        if not self.literal_artifact_byte_names:
            return set()
        return self.literal_artifact_byte_names[-1]

    def _track_artifact_byte_assignment(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        names = self._current_literal_artifact_names()
        if isinstance(value, ast.Constant) and isinstance(value.value, int | float):
            names.add(target.id)
        else:
            names.discard(target.id)

    def _check_forward_assignment(self, target: ast.AST) -> None:
        if isinstance(target, ast.Attribute) and self._root_name(target.value) == "self":
            self.review(
                target,
                "stateful_forward",
                "`forward` assigns to `self`; verify no eval-state adaptation.",
            )
        elif isinstance(target, ast.Subscript) and self._root_name(target.value) == "self":
            self.review(
                target,
                "stateful_forward",
                "`forward` mutates `self[...]`; verify no eval-state adaptation.",
            )

    @staticmethod
    def _call_name(func: ast.AST) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    @classmethod
    def _root_name(cls, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return cls._root_name(node.value)
        if isinstance(node, ast.Subscript):
            return cls._root_name(node.value)
        return None


def _top_level_findings(tree: ast.Module) -> list[Finding]:
    findings: list[Finding] = []
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.ClassDef):
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
            try:
                ast.literal_eval(value)
                continue
            except (ValueError, TypeError):
                pass
        findings.append(
            Finding(
                "review",
                "top_level_execution",
                "Top-level executable code should be reviewed for hidden side effects.",
                getattr(node, "lineno", index + 1),
            )
        )
    return findings


def _has_function(tree: ast.Module, name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body)


def _is_noop_extension_function(node: ast.FunctionDef) -> bool:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return all(
        isinstance(item, ast.Pass)
        or (
            isinstance(item, ast.Return)
            and (
                item.value is None
                or (isinstance(item.value, ast.Constant) and item.value.value is None)
            )
        )
        for item in body
    )


def _has_active_extension_function(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return not _is_noop_extension_function(node)
    return False


def _method_profile(tree: ast.Module | None, cfg: dict[str, Any] | None) -> MethodProfile:
    has_custom_model = tree is not None and _has_active_extension_function(tree, "build_model")
    has_custom_quantization = tree is not None and _has_active_extension_function(
        tree, "quantized_roundtrip"
    )

    changed_config_keys: list[str] = []
    if cfg is not None:
        changed_config_keys = [
            key for key in DEFAULT_CONFIG if cfg.get(key) != DEFAULT_CONFIG.get(key)
        ]

    shape_keys = {"model_dim", "num_layers", "num_heads", "mlp_mult", "seq_len"}
    training_keys = {
        "dropout",
        "train_steps",
        "train_batch_tokens",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "warmdown_steps",
        "grad_clip",
        "q_clip_percentile",
    }
    changed_shape_keys = [key for key in changed_config_keys if key in shape_keys]
    changed_training_keys = [key for key in changed_config_keys if key in training_keys]

    categories: list[str] = []
    if has_custom_model:
        categories.append("new_computation")
    if has_custom_quantization:
        categories.append("compression_serialization")
    if changed_shape_keys:
        categories.append("architecture_shape")
    if changed_training_keys:
        categories.append("training_hpo")
    if changed_config_keys:
        categories.append("config_tuning")
    if changed_config_keys and not has_custom_model and not has_custom_quantization:
        categories.append("config_only")
    if not categories:
        categories.append("baseline")

    if has_custom_model and has_custom_quantization:
        primary = "model_and_compression"
    elif has_custom_model:
        primary = "new_computation"
    elif has_custom_quantization:
        primary = "compression_serialization"
    elif changed_shape_keys:
        primary = "architecture_shape"
    elif changed_config_keys:
        primary = "config_only"
    else:
        primary = "baseline"

    return MethodProfile(
        primary=primary,
        categories=categories,
        changed_config_keys=changed_config_keys,
        changed_shape_keys=changed_shape_keys,
        changed_training_keys=changed_training_keys,
        has_custom_model=has_custom_model,
        has_custom_quantization=has_custom_quantization,
    )


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return objects
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = start + end


def _check_result(result_path: Path) -> tuple[dict[str, Any] | None, list[Finding]]:
    findings: list[Finding] = []
    if not result_path.exists():
        return None, [
            Finding("error", "missing_result", f"Result file does not exist: {result_path}")
        ]

    objects = _json_objects(result_path.read_text(errors="replace"))
    scored = [
        item
        for item in objects
        if "status" in item
        and (
            "val_bpb" in item
            or "artifact_bytes" in item
            or "reason" in item
            or item.get("phase") in {"train", "eval"}
        )
    ]
    if not scored:
        return None, [Finding("review", "no_scored_result", "No scored result JSON found.")]

    final = scored[-1]
    status = final.get("status")
    if status != "ok":
        findings.append(
            Finding("error", "result_status", f"Scored status is `{status}`, not `ok`.")
        )

    val_bpb = final.get("val_bpb")
    if not isinstance(val_bpb, int | float) or not math.isfinite(float(val_bpb)):
        findings.append(
            Finding("error", "missing_metric", "Result must report finite numeric val_bpb.")
        )
    elif float(val_bpb) < 0.0:
        findings.append(
            Finding(
                "error",
                "negative_metric",
                f"val_bpb={val_bpb:.6g} is impossible for cross-entropy bits per byte.",
            )
        )

    artifact_bytes = final.get("artifact_bytes")
    if isinstance(artifact_bytes, int | float) and artifact_bytes > ARTIFACT_CAP_BYTES:
        findings.append(
            Finding(
                "error",
                "artifact_cap",
                f"artifact_bytes={artifact_bytes} exceeds {ARTIFACT_CAP_BYTES}.",
            )
        )

    for item in scored:
        phase = item.get("phase")
        runtime = item.get("runtime_s")
        if (
            phase == "train"
            and item.get("status") == "train_ok"
            and isinstance(runtime, int | float)
            and runtime > TRAIN_BUDGET_SECONDS
        ):
            findings.append(
                Finding("error", "train_budget", f"train runtime {runtime:.3f}s exceeds 60s.")
            )
        if phase == "eval" and isinstance(runtime, int | float):
            train_phase_end = item.get("train_phase_end_s")
            if isinstance(train_phase_end, int | float):
                eval_runtime = runtime - train_phase_end
            elif "train_runtime_s" in item:
                eval_runtime = runtime
            else:
                continue
            if eval_runtime > EVAL_BUDGET_SECONDS:
                findings.append(
                    Finding(
                        "error", "eval_budget", f"eval runtime {eval_runtime:.3f}s exceeds 60s."
                    )
                )

    return final, findings


def run_gate(source: Path, result_path: Path | None) -> dict[str, Any]:
    findings: list[Finding] = []
    text = source.read_text()
    source_bytes = len(source.read_bytes())
    cfg: dict[str, Any] | None = None
    if source_bytes > ARTIFACT_CAP_BYTES:
        findings.append(
            Finding("error", "source_cap", f"source file alone exceeds {ARTIFACT_CAP_BYTES} bytes.")
        )
    elif source_bytes > 100_000:
        findings.append(
            Finding("review", "large_source", "Large source files may hide lookup tables or data.")
        )

    try:
        cfg = load_config_static(source)
    except StaticConfigUnavailable as exc:
        findings.append(
            Finding("error", "dynamic_config", f"get_config() must be a literal dict: {exc}")
        )
    except Exception as exc:
        findings.append(Finding("error", "invalid_config", str(exc)))

    try:
        tree = ast.parse(text, filename=str(source))
    except SyntaxError as exc:
        findings.append(Finding("error", "syntax_error", exc.msg, exc.lineno))
        tree = None

    if tree is not None:
        gate = StaticGate()
        gate.visit(tree)
        findings.extend(gate.findings)
        findings.extend(_top_level_findings(tree))

    method_profile = _method_profile(tree, cfg)

    result: dict[str, Any] | None = None
    if result_path is not None:
        result, result_findings = _check_result(result_path)
        findings.extend(result_findings)

    errors = sum(1 for finding in findings if finding.severity == "error")
    reviews = sum(1 for finding in findings if finding.severity == "review")
    notes = sum(1 for finding in findings if finding.severity == "note")
    status = "illegal" if errors else "review_required" if reviews else "legal"
    return {
        "status": status,
        "source": str(source),
        "source_bytes": source_bytes,
        "method_profile": asdict(method_profile),
        "result": result,
        "summary": {"errors": errors, "review": reviews, "notes": notes},
        "findings": [asdict(finding) for finding in findings],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a Mini PG solution as legal, illegal, or review_required."
    )
    parser.add_argument("source", nargs="?", default="solution.py")
    parser.add_argument(
        "result",
        nargs="?",
        help="Optional score output log or JSON result to verify against acceptance rules.",
    )
    args = parser.parse_args()

    report = run_gate(Path(args.source), Path(args.result) if args.result else None)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "illegal":
        raise SystemExit(1)
    if report["status"] == "review_required":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
