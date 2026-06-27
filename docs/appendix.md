# Weco Appendix

This is the Mini PG appendix for Weco operators after the main flow in
[../README.md](../README.md). The submission rules live in [rules.md](rules.md).
Do not use this appendix as the starting path from Claude Code.
Use the README for setup and the command to start a Weco run.

For general Weco behavior, use `uv run weco <command> --help` or the
[Weco docs](https://docs.weco.ai/).

## Evaluation Contract

Keep the eval command exactly as:

```bash
./scripts/weco_eval.sh solution.py
```

Do not inline the quality gate, call multiple commands, or replace this wrapper
in the Weco command. For every candidate, the wrapper runs:

```text
./scripts/check.sh solution.py
./scripts/quality_gate.sh solution.py
uv run modal run mini_pg/modal_app.py --source solution.py --output-json <tmp-json>
```

The final command is the same Modal app used by
[../scripts/score.sh](../scripts/score.sh). `scripts/weco_eval.sh` adds
`--output-json` so it can inspect the evaluator result before returning output
to Weco.

Valid candidates return evaluator `status: ok` and finite `val_bpb`. Candidates
that fail preflight checks, require review, time out, exceed the artifact cap, or
return an unscored evaluator status are reported to Weco as `status: buggy`
without `val_bpb`.

The quality gate output includes `method_profile` so operators can distinguish
config-only changes from architecture, computation, quantization, or
serialization changes. It is a method legality and review signal; it is not a
separate score.

## Monitor

Poll status by run ID:

```bash
uv run weco run status <run-id>
```

Use lineage-wide status when derived runs are active:

```bash
uv run weco run status --lineage <run-id>
```

Do not use `tail -f`, `watch`, or a blocking log stream from a chat agent. Poll,
summarize the current state, then return control to the user.

Common status fields include run status, current and total steps, best metric,
best step, pending nodes, and `agent_guidance`. Treat `agent_guidance` as the
live operating guidance for that run.

Repeated Modal auth or build failures are infrastructure issues; rerun
`uv run modal setup` and continue. Candidate-level `status: buggy` without
`val_bpb` means the candidate did not produce a legal score.

## Steer

For CLI steering, derive from the best known solution when the search direction
changes:

```bash
uv run weco run derive <run-id> \
  --from-step best \
  -i "<new direction>" \
  --output plain \
  --no-open
```

Append `--daemon` for unattended or agent-driven derived runs. To stop a run:

```bash
uv run weco run stop <run-id>
```

## Report Results

Use these commands to inspect the run:

```bash
uv run weco run results <run-id> --top 5 --format json
uv run weco run diff <run-id> --step best
uv run weco run show <run-id> --step best
uv run weco credits cost <run-id>
```

Report the best valid `val_bpb`, method change, `runtime_s`, `artifact_bytes`,
and Weco credit cost. If there are derived runs, use `--lineage` with
`uv run weco run results` or `uv run weco run status` when comparing across the
run family.

A leaderboard result still needs a fresh verifier run:

```bash
./scripts/verify.sh path/to/submission.py
```

[../scripts/verify.sh](../scripts/verify.sh) reruns the official evaluator,
saves the output under `results/`, and gates the saved result.
