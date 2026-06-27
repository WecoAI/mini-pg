"""Editable Mini Parameter Golf starter.

Only edit this file for a submission.

Use the three extension functions below:
- get_config(): return the literal config dict used by the evaluator.
- build_model(vocab_size, config): return None to use the default model, or
  return a custom torch.nn.Module with forward(input_ids, target_ids=None).
- quantized_roundtrip(model, clip_percentile): return None to use the default
  quantizer, or return (state_dict, artifact_bytes) for custom serialization.

Keep solutions causal. Do not read files, environment variables, network data,
or validation bytes outside the model's normal forward pass.
"""


def get_config() -> dict:
    return {
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


def build_model(vocab_size: int, config: dict):
    return None


def quantized_roundtrip(model, clip_percentile: float):
    return None
