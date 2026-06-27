"""Tiny artifact accounting for Mini PG."""

from __future__ import annotations

import io
import json
import zlib

import torch
from torch import Tensor, nn


def _quantize_tensor(tensor: Tensor, clip_percentile: float) -> dict:
    t = tensor.detach().float().cpu().contiguous()
    if not t.is_floating_point() or t.numel() == 0:
        return {"kind": "raw", "dtype": str(t.dtype), "shape": list(t.shape), "data": t}

    abs_t = t.abs().flatten()
    if clip_percentile < 1.0 and abs_t.numel() > 16:
        clip = float(torch.quantile(abs_t, clip_percentile).item())
    else:
        clip = float(abs_t.max().item()) if abs_t.numel() else 1.0
    if clip <= 0.0:
        clip = 1.0
    scale = clip / 127.0
    q = torch.clamp(torch.round(torch.clamp(t, -clip, clip) / scale), -127, 127).to(torch.int8)
    return {
        "kind": "int8",
        "shape": list(t.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "scale": scale,
        "data": q.contiguous(),
    }


def _dequantize_tensor(payload: dict) -> Tensor:
    if payload["kind"] == "raw":
        return payload["data"]
    dtype = getattr(torch, payload["dtype"])
    return (payload["data"].float() * float(payload["scale"])).to(dtype=dtype).contiguous()


def quantized_roundtrip(model: nn.Module, clip_percentile: float) -> tuple[dict[str, Tensor], int]:
    payload = {
        name: _quantize_tensor(tensor, clip_percentile)
        for name, tensor in model.state_dict().items()
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = zlib.compress(buffer.getvalue(), level=9)
    state_dict = {name: _dequantize_tensor(item) for name, item in payload.items()}
    manifest = json.dumps(
        {
            "format": "mini_pg_int8_zlib_v1",
            "num_tensors": len(payload),
            "compressed_bytes": len(compressed),
        },
        sort_keys=True,
    ).encode("utf-8")
    return state_dict, len(compressed) + len(manifest)
