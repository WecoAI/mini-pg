# Weco Instructions for Mini Parameter Golf

Optimize `solution.py` for lower `val_bpb`.

The complete rules are in `docs/rules.md`. Follow that document if anything
here seems ambiguous.

Search objective:

- Prefer non-HPO mechanisms over pure config packing.
- Config-only changes are valid, but treat them as warmups, ablations, or
  support for a model/quantization mechanism.
- Do not spend the run only changing `get_config()` values. After at most two
  config-only attempts, the next proposal should add or modify `build_model(...)`
  or `quantized_roundtrip(...)`.
- A non-HPO change means changing the model computation, parameter sharing,
  internal byte representation, or artifact serialization/quantization.
- Label each proposal with one primary category:
  `config-only`, `architecture-shape`, `new-computation`,
  `compression-serialization`, or `legality-risk`.
- Treat `architecture-shape` as tuning inside a model family. It is useful, but
  it covers width, depth, heads, or context-shape changes and does not count as
  a new mechanism unless the code changes the forward pass, parameter sharing,
  internal representation, or artifact serialization.
- After two consecutive `config-only` or `architecture-shape` attempts in the
  same model family, make the next proposal `new-computation` or
  `compression-serialization`, or explicitly explain why a local ablation is
  more valuable.

Hard constraints:

- Edit only `solution.py`.
- The quality gate runs before scoring. It may reject or flag submissions that
  appear to violate causal scoring, artifact/result rules, or runtime-side
  information boundaries.
- The quality gate is method-based, not score-threshold-based. Very low
  `val_bpb` is acceptable when it comes from a legal causal model and official
  evaluator output; do not pursue shortcuts that hardcode data, fake metrics,
  bypass artifact accounting, or adapt on validation bytes before scoring them.
- Read the quality gate's `method_profile` field. Use it to check whether the
  evaluated file was only config tuning or whether it actually added model or
  compression logic.
- Preserve a top-level `get_config()` function returning a dict.
- Keep any imports limited to the Python standard library and PyTorch.
- Do not print or fabricate metric/result lines from `solution.py`; final
  metrics must come from the evaluator.
- Do not read files, environment variables, network resources, or validation data.
- Do not modify the evaluator, training harness, or Modal app.
- Do not infer, alter, or route around the fixed official validation seed.
- Keep each change small and explainable.
- `get_config()` may return only the allowed keys listed below. Do not add
  custom keys for architecture internals such as sharing factors, physical layer
  counts, RoPE bases, or KV-head counts.
- If an extension function needs internal constants, define them inside `build_model(...)` or
  helper classes instead of adding config keys.
- Keep integer config fields as integers. In particular, `mlp_mult` must never
  be a float.

Allowed `get_config()` schema:

- `model_dim`: int
- `num_layers`: int
- `num_heads`: int
- `mlp_mult`: int
- `seq_len`: int
- `dropout`: float
- `train_steps`: int
- `train_batch_tokens`: int
- `learning_rate`: float
- `weight_decay`: float
- `warmup_steps`: int
- `warmdown_steps`: int
- `grad_clip`: float
- `q_clip_percentile`: float

Optional extension functions:

- You may add `build_model(vocab_size: int, config: dict)`.
- `build_model` must return a `torch.nn.Module`.
- The model must support `forward(input_ids, target_ids=None)`.
- When called as `forward(input_ids)`, the model must return logits shaped
  `[batch, seq_len, 256]`.
- The evaluator computes byte-level next-token loss itself; do not use
  `target_ids` to construct or shortcut the scored loss.
- You may add `quantized_roundtrip(model, clip_percentile: float)`.
- `quantized_roundtrip` must return `(state_dict, artifact_bytes)`.
- The returned `state_dict` must load back into the model with `strict=True`.
- `artifact_bytes` must be an `int`, not a tensor, float, or string.
- The evaluator adds the `solution.py` source bytes to `artifact_bytes`.

Tokenizer rule:

- Do not change the external scoring tokenizer. The task is byte-level.
- Internal byte representations are allowed only if the model still scores
  causal next-byte prediction over 256 byte values.

The evaluator validates ranges. Stay within these practical bounds:

- `model_dim`: 128 to 768, divisible by `num_heads`
- `num_layers`: 2 to 12
- `num_heads`: 1 to 12, must divide `model_dim`; head dimension must be even
- `mlp_mult`: 1 to 4
- `seq_len`: 128 to 2048
- `dropout`: 0.0 to 0.25
- `train_steps`: 50 to 900
- `train_batch_tokens`: 32,768 to 1,572,864, divisible by `seq_len`
- `learning_rate`: 0.00001 to 0.02
- `weight_decay`: 0.0 to 0.5
- `warmup_steps`: 0 to 400
- `warmdown_steps`: 0 to 800
- `grad_clip`: 0.0 to 5.0
- `q_clip_percentile`: 0.95 to 1.0
- `warmup_steps + warmdown_steps < train_steps`

Prioritize:

1. Shared or recurrent transformer blocks that increase effective depth without
   storing a full copy of every layer.
2. Quantization or serialization changes that reduce artifact bytes without
   breaking validation loss.
3. Small causal architecture changes such as output byte-frequency bias, causal
   convolutional features, or hashed n-gram byte embeddings.
4. Removing or replacing learned positional embeddings with a cheaper causal
   alternative.
5. Config improvements that support the chosen mechanism.

Avoid:

- broad rewrites
- only changing `model_dim`, `num_layers`, `train_steps`, batch size, learning
  rate, or schedules across many attempts
- large embedded lookup tables
- validation-time adaptation
- future-byte leakage
- changing many mechanisms at once
- growing the model so artifact bytes exceed the cap
- optimizing for anything other than the official Modal score
- increasing runtime without a clear reason; train and artifact/eval each have a 60-second hard cap

The metric line to optimize is:

```text
val_bpb: <number>
```

When summarizing a run, copy `val_bpb`, `runtime_s`, and `artifact_bytes` from
the latest evaluator output. Do not invent, estimate, or self-report them.
