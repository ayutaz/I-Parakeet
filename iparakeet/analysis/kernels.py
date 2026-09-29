"""Accuracy of the integer kernels in isolation (no model or data needed)."""

import torch

from iparakeet.intops.softmax import IntSoftmax
from iparakeet.intops.swish import make_swish
from iparakeet.quant.fixed_point import dequantize, qmax, quantize, scale_from_alpha
from iparakeet.sim.report import swish_max_error

SWISH_SPECS = ("poly:linf_swish", "poly:l2_swish", "poly:linf_tanh", "poly:l2_tanh", "hardswish", "lut")


def swish_kernel_errors(in_bits: int = 8, alpha: float = 8.0, out_bits: int = 8) -> list[dict]:
    """Max |kernel(x) - x*sigmoid(x)| over the whole input grid, per approximation."""
    s_in = scale_from_alpha(alpha, in_bits)
    s_out = scale_from_alpha(alpha, out_bits)
    q = torch.arange(-qmax(in_bits), qmax(in_bits) + 1)
    x = dequantize(q, s_in)
    exact = x * torch.sigmoid(x)
    rows = []
    for spec in SWISH_SPECS:
        out = make_swish(spec, s_in, in_bits, s_out, out_bits)(q)
        rows.append({
            "spec": spec, "in_bits": in_bits, "alpha": alpha, "out_step": s_out,
            "approx_max_err": None if spec == "lut" else swish_max_error(spec),
            "int_max_err": float((dequantize(out, s_out) - exact).abs().max()),
        })
    return rows


def softmax_errors(alphas=(8.0, 16.0, 32.0), bits: int = 8, rows: int = 64, cols: int = 200, seed: int = 0) -> list[dict]:
    """Max |p_int - softmax| for INT score grids of range alpha, per range reduction."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for alpha in alphas:
        scores = (torch.rand(rows, cols, generator=gen, dtype=torch.float64) * 2 - 1) * alpha
        s = scale_from_alpha(alpha, bits)
        q = quantize(scores, s, bits)
        ref = torch.softmax(dequantize(q, s), dim=-1)
        for mode in ("ibert", "log2"):
            p = IntSoftmax(s, 8, mode)(q).double() / qmax(8)
            out.append({"alpha": alpha, "bits": bits, "range_reduction": mode, "max_abs_err": float((p - ref).abs().max())})
    return out


def kernel_report_markdown(swish_rows: list[dict], softmax_rows: list[dict]) -> str:
    lines = ["| Swish kernel | input bits | approx. max err (real) | integer kernel max err | output step |", "|---|---|---|---|---|"]
    for r in swish_rows:
        approx = "-" if r["approx_max_err"] is None else f"{r['approx_max_err']:.4f}"
        lines.append(f"| {r['spec']} | {r['in_bits']} | {approx} | {r['int_max_err']:.4f} | {r['out_step']:.4f} |")
    lines += ["", "| score range alpha | score bits | range reduction | max abs prob err |", "|---|---|---|---|"]
    for r in softmax_rows:
        lines.append(f"| {r['alpha']:g} | {r['bits']} | {r['range_reduction']} | {r['max_abs_err']:.4f} |")
    return "\n".join(lines)
