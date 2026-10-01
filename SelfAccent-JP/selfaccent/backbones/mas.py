"""Monotonic alignment search (Glow-TTS / Matcha-TTS), vectorized over tokens."""

from __future__ import annotations

import numpy as np
import torch


def maximum_path(value: np.ndarray) -> np.ndarray:
    """Hard monotonic alignment [tx, ty] maximizing the summed ``value``.

    Every frame is assigned to exactly one token, tokens are visited in order
    and each token gets at least one frame (requires ty >= tx).
    """
    tx, ty = value.shape
    if ty < tx:
        raise ValueError(f"cannot align {tx} tokens to {ty} frames")
    q = np.full((tx, ty), -np.inf)
    q[0, 0] = value[0, 0]
    for y in range(1, ty):
        stay = q[:, y - 1]
        move = np.concatenate(([-np.inf], q[:-1, y - 1]))
        q[:, y] = value[:, y] + np.maximum(stay, move)
        # token x can only be reached once x <= y and must leave room for the rest
        q[min(y + 1, tx) :, y] = -np.inf
        q[: max(0, tx - (ty - y)), y] = -np.inf
    path = np.zeros((tx, ty), dtype=np.int64)
    x = tx - 1
    for y in range(ty - 1, -1, -1):
        path[x, y] = 1
        if x > 0 and (x == y or q[x - 1, y - 1] > q[x, y - 1]):
            x -= 1
    return path


@torch.no_grad()
def monotonic_alignment(log_p: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor) -> torch.Tensor:
    """Batched MAS. ``log_p`` is [B, Tx, Ty]; returns a 0/1 tensor of the same shape."""
    out = torch.zeros_like(log_p)
    lp = log_p.detach().cpu().double().numpy()
    for b in range(lp.shape[0]):
        tx, ty = int(x_lengths[b]), int(y_lengths[b])
        out[b, :tx, :ty] = torch.from_numpy(maximum_path(lp[b, :tx, :ty])).to(out)
    return out
