# Rules and Constraints

This document is the source of truth for Mini Parameter Golf submissions. It is
written for both humans and coding agents.

## Goal

Minimize:

```text
val_bpb
```

Lower is better. `val_bpb` is the byte-level language-model validation score
reported by the fixed evaluator.

## What Participants Submit

Submissions are exactly one file:

```text
solution.py
```

The final submitted file must define:

```python
def get_config() -> dict:
    ...
```

The evaluator imports `get_config()`, validates the returned dictionary, trains
the model, compresses the artifact, evaluates validation loss, and prints the
score.

Config-only submissions are valid. Stronger submissions may also define optional
model and quantization extension functions inside the same `solution.py` file.

The competition is intended to allow model-craft and compression work, not only
hyperparameter search. Config-only changes are a valid strategy, but participants
are encouraged to use the optional extension functions when exploring architecture,
parameter-sharing, internal byte features, or quantization mechanisms.

## Allowed Edits

Participants and agents may edit only:

```text
solution.py
```

Allowed changes include:

- config values returned by `get_config()`
- helper classes or functions used by the submitted file
- an optional `build_model(...)` extension function
- an optional `quantized_roundtrip(...)` extension function

The harness falls back to the default `mini_pg/model.py` and
`mini_pg/quantize.py` implementations when extension functions are absent.

## Optional Extension Functions

`solution.py` may define:

```python
def build_model(vocab_size: int, config: dict):
    ...
```

`build_model` must return a `torch.nn.Module`. The module must accept byte-token
inputs in its `forward(input_ids, target_ids=None)` method and must return
logits of shape `[batch, seq_len, 256]` when called as `forward(input_ids)`.
The evaluator computes byte-level next-token loss itself; models must not use
`target_ids` to construct or shortcut the scored loss.

`solution.py` may also define:

```python
def quantized_roundtrip(model, clip_percentile: float):
    ...
```

`quantized_roundtrip` must return:

```python
(state_dict, artifact_bytes)
```

The returned `state_dict` must load back into the same model with
`strict=True`. `artifact_bytes` must be an `int` and must include the
compressed model payload bytes created by the extension function. The evaluator adds the
submitted `solution.py` source bytes.

## Non-HPO Search Surface

The following are first-class, leaderboard-valid ideas when implemented inside
`solution.py` and kept causal:

- shared or recurrent transformer blocks in `build_model(...)`
- cheaper position handling such as sinusoidal positions, RoPE, or no learned
  position table
- small causal convolutional features or hashed n-gram byte embeddings
- output byte-frequency bias or other causal byte-prior features
- custom raw packed int8 serialization in `quantized_roundtrip(...)`
- group-wise, per-row, mixed int4/int8, or codebook quantization
- deduplicating shared tensors in the serialized artifact and reconstructing the
  full `state_dict` before `load_state_dict(strict=True)`

Changing only `model_dim`, `num_layers`, `train_steps`, batch size, learning
rate, or schedules is valid but should be treated as config tuning rather than
model-craft.

Use these categories when recording experiments:

- `config-only`: only values returned by `get_config()` changed.
- `architecture-shape`: width, depth, heads, or sequence length changed inside
  the same model family.
- `new-computation`: `build_model(...)` changes the actual forward computation,
  parameter sharing, byte features, or internal representation.
- `compression-serialization`: `quantized_roundtrip(...)` changes artifact
  construction, quantization, packing, or tensor deduplication.
- `legality-risk`: the proposal touches causality, runtime side effects,
  artifact accounting, metric reporting, or validation-data boundaries and needs
  organizer review.

Architecture-shape changes are useful, but they are still tuning inside a model
family. Count a proposal as a new mechanism only when it changes the computation
or compression/serialization path, not merely width, depth, heads, step count,
or schedule.

## Disallowed Edits

Do not edit these files for a leaderboard submission:

```text
mini_pg/train.py
mini_pg/modal_app.py
mini_pg/model.py
mini_pg/quantize.py
mini_pg/data.py
mini_pg/config.py
scripts/*
docs/*
instructions/*
```

Those files define the task, data, scoring path, and command surface. Changing
them creates a separate harness experiment, not an official leaderboard result.

Do not add behavior to `solution.py` that:

- adds network calls or external data
- reads files
- reads environment variables
- detects evaluator split or seed
- tries to inspect validation data outside the evaluator's normal forward pass
- changes evaluator behavior
- bypasses runtime or artifact limits
- hardcodes the fixed validation seed or generated validation text
- uses future validation bytes to improve current-byte predictions
- adapts on validation bytes before scoring those same bytes

Imports should stay limited to the Python standard library and PyTorch. Do not
add new package dependencies.

## Config Keys

The evaluator accepts only these keys:

| Key | Type | Valid Range | Notes |
| --- | --- | --- | --- |
| `model_dim` | int | 128 to 768 | Must be divisible by `num_heads`. |
| `num_layers` | int | 2 to 12 | More layers usually cost more artifact bytes and runtime. |
| `num_heads` | int | 1 to 12 | `model_dim / num_heads` must be even. |
| `mlp_mult` | int | 1 to 4 | Multiplier for MLP hidden width. |
| `seq_len` | int | 128 to 2048 | Context length. |
| `dropout` | float | 0.0 to 0.25 | Usually keep small. |
| `train_steps` | int | 50 to 900 | More steps cost runtime. |
| `train_batch_tokens` | int | 32,768 to 1,572,864 | Must be divisible by `seq_len`. |
| `learning_rate` | float | 0.00001 to 0.02 | Optimizer learning rate. |
| `weight_decay` | float | 0.0 to 0.5 | AdamW weight decay. |
| `warmup_steps` | int | 0 to 400 | Must satisfy schedule constraint below. |
| `warmdown_steps` | int | 0 to 800 | Must satisfy schedule constraint below. |
| `grad_clip` | float | 0.0 to 5.0 | Gradient clipping norm; 0 disables clipping. |
| `q_clip_percentile` | float | 0.95 to 1.0 | Quantization clipping percentile. |

Additional validation constraints:

- `model_dim % num_heads == 0`
- `(model_dim / num_heads)` must be even
- `train_batch_tokens % seq_len == 0`
- `warmup_steps + warmdown_steps < train_steps`

Unknown keys fail validation.

## Scoring Commands

Use these commands from the repo root:

```bash
./scripts/check.sh
./scripts/smoke.sh
./scripts/score.sh
./scripts/verify.sh
```

Meaning:

- `check.sh` validates `solution.py` without training.
- `smoke.sh` runs a cheap Modal CPU smoke test; it is not leaderboard scoring.
- `score.sh` runs the official Modal score path.
- `verify.sh` reruns a submitted file, saves the scorer output, and runs the
  quality gate against that saved output.

The participant-visible commands intentionally have no flags.

## Official Scoring

During the workshop, participants iterate on one fixed official score:

```bash
./scripts/score.sh
```

The official validation seed is fixed in the evaluator for the event. There is
no seed flag in the interface. This keeps the 90-minute competition focused on
more iterations instead of multi-seed reruns or seed selection.

Final ranking uses the same score command. Organizers can rerun submitted files
for consistency:

```bash
./scripts/score.sh path/to/submission.py
```

For acceptance, prefer the combined verification command:

```bash
./scripts/verify.sh path/to/submission.py
```

Do not trust `val_bpb`, `runtime_s`, or `artifact_bytes` copied from an agent
summary or chat message. Only metrics from a fresh official evaluator run count.

For leaderboard PRs, store accepted entries under `record/<record-id>/` with
the submitted `solution.py` and a short result README. The scorer still evaluates
one submitted Python file; the record directory is only for archival metadata
and leaderboard review.

Participants and agents should not infer, alter, or route around the fixed
official validation seed.

## Artifact Cap

The mini artifact cap is:

```text
4,000,000 bytes
```

The artifact count includes the compressed model artifact plus counted source
bytes. The evaluator counts:

```text
solution.py
```

Large lookup tables, embedded data, or generated code in `solution.py` are not
free. If the artifact exceeds the cap, the evaluator returns
`status: buggy` with `reason: artifact_cap_exceeded` and does not report
`val_bpb`.

## Runtime

The real score path is intended to run on Modal with:

```text
H100:8
```

The hard scored budgets are:

```text
train: 60 seconds
post-training artifact + eval: 60 seconds
```

The Modal evaluator enforces these as two separate subprocess timeouts. It runs
training first and requires a checkpoint, then runs post-training artifact
construction, `load_state_dict(strict=True)`, and validation loss in a second
subprocess. The outer Modal/Weco wrappers allow a small amount of extra time for
startup, logging, and returning a parseable result. If either scored phase times
out, the run returns `status: buggy` with a timeout reason and does not report
`val_bpb`.

Runs report `runtime_s`. Organizers may use runtime as a tie-breaker after
`val_bpb` and artifact validity.

Do not deliberately waste compute to gain an unfair advantage.

The default baseline is intentionally compact: 80 train steps, 32,768 train
tokens per step, and a 500,000-byte validation slice. Larger batches, larger
models, and more steps are allowed by validation, but they can timeout.

## What Counts as a Valid Result

A valid leaderboard result must:

- pass `./scripts/check.sh`
- run through `./scripts/score.sh` without evaluator failure
- be verified from saved scorer output, not from an agent-written summary
- report `status: ok`
- report a finite, nonnegative `val_bpb`
- finish within the 60-second train budget and 60-second eval/artifact budget
- not exceed the artifact cap
- use only `solution.py` changes
- preserve causal byte prediction and correct `val_bpb` semantics
- avoid validation-data lookup, seed hardcoding, network calls, external data,
  hidden state, or evaluator changes
- avoid printing or fabricating metric/result lines from `solution.py`

Submissions with custom extension functions should be reviewed by diff before final rerun.

## Post-Hoc Quality Gate

Run the quality gate before accepting a result:

```bash
./scripts/quality_gate.sh path/to/solution.py path/to/score-output.txt
```

The gate classifies a submission as:

- `legal`: no static rule violations and, when a result log is provided, a
  valid `status: ok` scored result under the artifact and phase budgets.
- `review_required`: no obvious illegality, but the solution uses patterns that
  need human inspection.
- `illegal`: clear rule or result violation.

The gate is method-based, not score-threshold-based. It must not reject a
submission merely because `val_bpb` is unusually low. Strong legal methods can
beat expectations. Instead, inspect how the result was achieved: whether the
model predicts causally from prior bytes, whether the reported artifact bytes
match the submitted artifact representation, and whether the score came from
the official evaluator rather than candidate output. An impossible result such
as a negative `val_bpb`, a missing metric, a non-`ok` result status, an artifact
cap violation, or a phase-budget violation is illegal.

This gate follows the spirit of OpenAI Parameter Golf issue #1017: scored
probabilities must be causal, normalized over the byte alphabet, scored before
any update from the current byte, and produced in a single left-to-right pass.
For Mini PG, that means the evaluator computes loss from full logits, solutions
must not use `target_ids` to shortcut loss, and submissions must not read
validation data, seeds, files, environment variables, network resources, or
other runtime side information.

Static analysis is intentionally conservative. A `legal` gate result is not a
mathematical proof, and `review_required` is not an accusation. Treat it as the
minimum organizer review pass before rerunning a leaderboard candidate.

The quality gate also prints an advisory `method_profile` field. It reports
whether the submitted file is baseline, config-only, architecture-shape tuning,
new model computation, or compression/serialization work. This field helps
organizers audit search diversity; it is not a separate legality rule.

## Recommended Workflow

1. Run the baseline.
2. Choose one clear mechanism or config hypothesis. Prefer a non-HPO mechanism
   after the baseline is reproduced.
3. Edit `solution.py`.
4. Run `./scripts/check.sh`.
5. Run `./scripts/score.sh`.
6. Record `val_bpb`, `runtime_s`, `artifact_bytes`, status, and the diff from
   the evaluator output, not from an agent summary.
7. Keep, revert, or follow up based on evidence.

Agents should avoid broad rewrites and multi-change bundles until there is a
measured reason.

## Tie-Breakers

Suggested ranking order:

1. Official `val_bpb`, lower is better.
2. Valid artifact at or below 4,000,000 bytes.
3. Smaller artifact bytes.
4. Lower runtime.
5. Clearer experiment log.
