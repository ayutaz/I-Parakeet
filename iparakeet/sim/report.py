"""Comparison of simulated WERs with the paper's Table 2 / Table 3 and the M5 order checks."""

from itertools import combinations

import numpy as np

from iparakeet.intops.swish import SWISH_COEFFS
from iparakeet.sim.recipe import PAPER_FP32, PAPER_TABLE2, PAPER_TABLE3

TIE = 0.1  # pairs closer than this in the paper are treated as equal (not order-checked)
TESTSETS = ("test-clean", "test-other", "commonvoice-test")


def _pairs_consistent(items: list[tuple[float, float]]) -> bool:
    """items: (paper, reproduced). Every pair the paper separates by >= TIE keeps its order."""
    for (p1, r1), (p2, r2) in combinations(items, 2):
        if abs(p1 - p2) < TIE:
            continue
        if (p1 < p2) != (r1 < r2):
            return False
    return True


def check_table2_order(results: dict, fp32: dict | None = None) -> dict[str, bool]:
    checks = {}
    for i, ts in enumerate(TESTSETS):
        items = [(PAPER_TABLE2[name][i], results[name][ts]) for name in PAPER_TABLE2 if ts in results.get(name, {})]
        if len(items) < len(PAPER_TABLE2):
            continue
        if fp32 and ts in fp32:
            items.append((PAPER_FP32[i], fp32[ts]))
        checks[ts] = _pairs_consistent(items)
    return checks


def check_table3_order(results: dict, testset: str = "test-other") -> dict[str, bool | None]:
    checks = {}
    for group in dict.fromkeys(r.group for r in PAPER_TABLE3):
        rows = [r for r in PAPER_TABLE3 if r.group == group]
        if any(testset not in results.get(r.recipe, {}) for r in rows):
            checks[group] = None
            continue
        checks[group] = _pairs_consistent([(r.wer_test_other, results[r.recipe][testset]) for r in rows])
    return checks


def _cell(repro: float | None, paper: float) -> str:
    if repro is None:
        return f"n/a ({paper:.2f})"
    return f"{repro:.2f} ({paper:.2f}, {repro - paper:+.2f})"


def table2_markdown(results: dict, fp32: dict | None = None) -> str:
    lines = ["| model | " + " | ".join(TESTSETS) + " |", "|---" * (len(TESTSETS) + 1) + "|"]
    fp32 = fp32 or {}
    lines.append("| FP32 | " + " | ".join(_cell(fp32.get(ts), PAPER_FP32[i]) for i, ts in enumerate(TESTSETS)) + " |")
    for name, paper in PAPER_TABLE2.items():
        cells = [_cell(results.get(name, {}).get(ts), paper[i]) for i, ts in enumerate(TESTSETS)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: reproduced WER % (paper, difference).")
    return "\n".join(lines)


def swish_max_error(spec: str) -> float:
    """Max |sw(x) - approx(x)| over x (the Table 3 'Max err.' column, real-valued)."""
    x = np.arange(-30, 30, 1e-3)
    sw = x / (1 + np.exp(-x))
    if spec == "hardswish":
        approx = x * np.clip(x + 3, 0, 6) / 6
    else:
        a, c = SWISH_COEFFS[spec.split(":", 1)[1]]
        u = x / 2
        t = np.sign(u) * (a * (np.minimum(np.abs(u), c) - c) ** 2 + 1)
        approx = x * (1 + t) / 2
    return float(np.abs(sw - approx).max())


_SWISH_SPEC = {"iparakeet": "poly:linf_swish", "swish_l2_swish": "poly:l2_swish", "swish_linf_tanh": "poly:linf_tanh",
               "swish_l2_tanh": "poly:l2_tanh", "swish_hardswish": "hardswish"}


def table3_markdown(results: dict, testset: str = "test-other") -> str:
    lines = [
        "| group | configuration | max err (ours) | max err (paper) | WER | WER (paper) | diff |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in PAPER_TABLE3:
        ours_err = f"{swish_max_error(_SWISH_SPEC[r.recipe]):.3f}" if r.group == "swish" else "-"
        paper_err = f"{r.max_err:.3f}" if r.max_err is not None else "-"
        wer = results.get(r.recipe, {}).get(testset)
        wer_s = "n/a" if wer is None else f"{wer:.2f}"
        diff = "n/a" if wer is None else f"{wer - r.wer_test_other:+.2f}"
        lines.append(f"| {r.group} | {r.label} | {ours_err} | {paper_err} | {wer_s} | {r.wer_test_other:.2f} | {diff} |")
    return "\n".join(lines)
