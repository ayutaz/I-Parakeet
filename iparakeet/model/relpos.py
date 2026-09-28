"""Relative positional encoding and the relative shift as a static index map (paper Sec. 3.1)."""

import math

import torch


def relshift_index(L: int, device=None) -> torch.Tensor:
    """out[i, j] = x[i, L - 1 - i + j]: the column of P holding relative position i - j."""
    i = torch.arange(L, device=device).unsqueeze(1)
    j = torch.arange(L, device=device).unsqueeze(0)
    return L - 1 - i + j


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Relative shift of (..., L, 2L-1) position scores to (..., L, L) as a pure gather."""
    L = x.shape[-2]
    idx = relshift_index(L, x.device).expand(*x.shape[:-1], L)
    return torch.gather(x, -1, idx)


def rel_pos_table(max_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal table for positions max_len-1 .. -(max_len-1) (row r <-> position max_len-1-r)."""
    positions = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    table = torch.zeros(positions.shape[0], d_model)
    table[:, 0::2] = torch.sin(positions * div_term)
    table[:, 1::2] = torch.cos(positions * div_term)
    return table


def rel_pos_emb(table: torch.Tensor, L: int) -> torch.Tensor:
    """Centered slice of `table` with the 2L-1 positions L-1 .. -(L-1) (NeMo RelPositionalEncoding)."""
    max_len = (table.shape[0] + 1) // 2
    if L > max_len:
        raise ValueError(f"length {L} exceeds positional table size {max_len}")
    return table[max_len - L : max_len + L - 1]
