# Modal Setup

## Why Modal

The workshop evaluator uses Modal so participants do not need to manage GPU
drivers, CUDA environments, or 8-GPU scheduling. The real function requests:

```python
gpu="H100:8"
```

in `mini_pg/modal_app.py`.

## Organizer Setup

```bash
cd mini-pg
uv sync
uv run modal setup
./scripts/smoke.sh
```

The score scripts use the Modal CLI configuration already available on the
current machine. No workshop-specific account setting is required in the
commands.

Then warm the real image before the event:

```bash
./scripts/score.sh
```

This launches an 8 H100 job. Do this intentionally, not as a casual smoke test.
The smoke command uses a tiny data/step budget and is only a packaging check;
it is not comparable to leaderboard results.

The full score path enforces a 60-second train subprocess timeout and a
60-second post-training artifact/eval subprocess timeout inside the remote Modal
function. The parent Modal evaluator owns those timeouts, so candidate code
cannot extend them from inside `solution.py`. The Modal function timeout is
slightly higher so timed-out runs can return `status: buggy` with a reason
instead of disappearing as infrastructure errors. The default baseline is sized
below those caps, so a timeout usually means a participant increased model,
batch, steps, artifact work, or validation work too aggressively.

## Final Rerun

Rerun submitted files from an organizer environment:

```bash
./scripts/score.sh path/to/submission.py
```

This uses the same fixed official validation seed as `./scripts/score.sh`.
There is no seed flag in the workshop interface.

For submissions with custom model or quantization extension functions, inspect the submitted
`solution.py` for allowed behavior, then run:

```bash
./scripts/check.sh path/to/submission.py
./scripts/score.sh path/to/submission.py
```

The same Modal app and fixed seed are used. The evaluator counts the submitted
`solution.py` source toward the artifact cap.

## Common Failures

### Modal Auth Missing

Run:

```bash
uv run modal setup
```

### First Run Is Slow

The first run builds the Modal image. Warm it before attendees arrive.

### Candidate Fails Validation

Run:

```bash
./scripts/check.sh
```

Common issues are `model_dim` not divisible by `num_heads`, train batch tokens
not divisible by `seq_len`, or warmup plus warmdown exceeding train steps.

### Artifact Cap Exceeded

The evaluator returns `status: buggy` with `reason: artifact_cap_exceeded` if
`artifact_bytes` exceeds the cap. Reduce width, depth, MLP multiplier, or
sequence position embeddings.
