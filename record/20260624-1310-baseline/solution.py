"""Editable Mini Parameter Golf baseline.

Participants and agents edit this file. Submissions must define get_config().
They may also add build_model() and quantized_roundtrip() extension functions
in this same file.
"""


def get_config() -> dict:
    """Return the model and training configuration to score.

    Keep values simple and JSON-like. The harness validates ranges before
    launching the GPU job.
    """

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
