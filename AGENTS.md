# AGENTS.md - Mini Parameter Golf

## Objective

Improve `solution.py` for Mini Parameter Golf. The primary metric is `val_bpb`;
lower is better. The score comes from the Modal evaluator.

Before editing, read `docs/rules.md`. It is the source of truth for allowed
files, config ranges, scoring paths, and disallowed behavior. For Weco runs,
use the command in `README.md` and the operator flow in
`docs/appendix.md`.

## Submission Rules

- Edit only `solution.py` unless the user explicitly asks to modify the harness.
- Submissions are exactly one `solution.py` file.
- Preserve `get_config()` as the public API.
- Optional extension functions: `build_model(...)` and `quantized_roundtrip(...)`.
- Do not edit data generation, training/evaluation scoring, Modal, scripts,
  docs, instruction files, `mini_pg/model.py`, or `mini_pg/quantize.py` for a
  leaderboard submission.
- Keep prediction causal: current validation bytes must not benefit from future
  validation bytes or from adaptation on the same bytes before scoring.
- Do not add network calls, external data, validation-set lookup, or hidden state.
- Do not read files or environment variables from `solution.py`.
- Do not bypass artifact/runtime caps.
- Do not infer, alter, or route around the fixed official validation seed.
- Count source size as part of the artifact budget; large code tables are not
  free.
- Treat the Modal wrapper's 60-second train subprocess timeout and 60-second
  eval/artifact subprocess timeout as hard caps, not suggestions.

## Required Checks

Before running a GPU job:

```bash
./scripts/check.sh
```

For the real official score:

```bash
./scripts/score.sh
```

For final acceptance, rerun and gate saved evaluator output:

```bash
./scripts/verify.sh
```

## Experiment Logging

Record at least:

- hypothesis
- changed config keys or mechanism
- command used
- `val_bpb` from evaluator output
- `runtime_s` from evaluator output
- `artifact_bytes` from evaluator output
- whether the result should be kept, reverted, or followed up

## Good First Ideas

- shared or recurrent transformer blocks
- custom quantization or raw packed serialization
- causal byte features such as output bias, convolution, or hashed n-grams
- cheaper position handling that removes learned positional parameters
- learning-rate and warmdown schedule
- width/depth tradeoff under the artifact cap
- sequence length versus batch-token budget
- attention head count that divides `model_dim`
- mild regularization changes
- quantization clipping percentile

## Avoid

- broad rewrites
- changing only config knobs for an entire run
- changing many knobs or mechanisms at once
- making the model too large to fit the artifact cap
- increasing train steps without a runtime reason
