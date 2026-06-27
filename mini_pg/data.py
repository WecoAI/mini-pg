"""Deterministic byte-level data for the workshop harness.

The first event should avoid dataset logistics. This generator gives a stable,
language-like byte stream with fixed train and official validation splits. The
final workshop can replace this file with FineWeb-derived shards once the flow
is proven.
"""

from __future__ import annotations

import random

VOCAB_SIZE = 256
TRAIN_SEED = 17
OFFICIAL_VAL_SEED = 31

BASE_PARAGRAPHS = [
    "Parameter Golf asks for the best language model that fits inside a tiny artifact.",
    "A good experiment changes one thing, measures the result, and keeps a clear log.",
    "Compression rewards models that spend parameters on structure instead of memorization.",
    "The workshop score is bits per byte on a hidden validation stream.",
    "Small transformers can learn punctuation, indentation, and repeated technical phrases.",
    "Agents should reproduce the baseline before trying a clever optimization.",
    "A fair benchmark keeps the official validation seed fixed for the event.",
    "Modal supplies the GPUs; Weco supplies the search loop; participants supply judgment.",
    "The baseline is intentionally beatable, but not by breaking the evaluator.",
    "Short context is cheap; long context may help evaluation but costs wall clock.",
    "Learning rate schedules often matter more than architectural decoration.",
    "Quantization can win or lose depending on whether the compressed artifact still predicts well.",
    "A clean leaderboard needs the metric, runtime, artifact bytes, and source diff.",
    "Research systems need memory: dead ends should stop repeating across attempts.",
    "The fastest useful loop is the one that turns a hypothesis into a scored result.",
]

CODE_SNIPPETS = [
    "def score(loss, tokens, bytes_): return loss / 0.69314718056 * tokens / bytes_",
    "for step in range(train_steps): loss.backward(); optimizer.step(); optimizer.zero_grad()",
    "if artifact_bytes > cap: raise RuntimeError('artifact cap exceeded')",
    "config = {'model_dim': 384, 'num_layers': 6, 'seq_len': 512}",
    "val_bpb: 1.2345 runtime_s: 282.1 artifact_bytes: 3999123",
]


def build_text(seed: int, target_bytes: int) -> bytes:
    rng = random.Random(seed)
    chunks: list[bytes] = []
    total_bytes = 0
    topic_words = [
        "baseline",
        "leaderboard",
        "optimizer",
        "validation",
        "artifact",
        "latency",
        "gradient",
        "attention",
        "tokens",
        "bytes",
        "modal",
        "weco",
    ]

    while total_bytes < target_bytes:
        para = rng.choice(BASE_PARAGRAPHS)
        if rng.random() < 0.35:
            para += " " + rng.choice(BASE_PARAGRAPHS)
        if rng.random() < 0.22:
            para += "\n" + rng.choice(CODE_SNIPPETS)
        if rng.random() < 0.18:
            words = rng.sample(topic_words, k=4)
            para += "\n" + " | ".join(words) + f" | seed={seed} | bucket={rng.randrange(97)}"
        chunk = para + "\n\n"
        chunk_bytes = chunk.encode("utf-8")
        chunks.append(chunk_bytes)
        total_bytes += len(chunk_bytes)

    return b"".join(chunks)[:target_bytes]


def split_seed(split: str) -> int:
    if split == "train":
        return TRAIN_SEED
    if split == "public":
        return OFFICIAL_VAL_SEED
    raise ValueError(f"unknown split: {split}")


def build_split(split: str, target_bytes: int) -> bytes:
    return build_text(split_seed(split), target_bytes)
