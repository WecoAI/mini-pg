"""Small byte-level GPT used by the Mini PG harness."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_mult * dim, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class ByteGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_mult: int,
        seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, mlp_mult, dropout) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        _, seq_len = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if target_ids is None:
            return logits
        return F.cross_entropy(logits.flatten(0, 1), target_ids.flatten())


def estimate_param_count(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += param.numel()
    return total


def cosine_schedule(step: int, total_steps: int, warmup_steps: int, warmdown_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_start = max(warmup_steps, total_steps - warmdown_steps)
    if step < decay_start:
        return 1.0
    progress = (step - decay_start + 1) / max(1, total_steps - decay_start)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
