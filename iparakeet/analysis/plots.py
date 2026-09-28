"""Figures reproducing the paper's Fig. 2 (static PNG, light surface)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 of the validated reference palette


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_bn_spread(spread: list[float], path) -> None:
    """Fig. 2(a): per-layer max/median of the BatchNorm scale (log y)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6), facecolor=SURFACE)
    _style(ax)
    layers = list(range(len(spread)))
    ax.bar(layers, spread, width=0.7, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("layer", color=TEXT_2)
    ax.set_ylabel("max / median", color=TEXT_2)
    ax.set_title("BatchNorm scale spread per layer", color=TEXT, fontsize=10, loc="left")
    for i in sorted(layers, key=lambda i: spread[i], reverse=True)[:2]:
        ax.annotate(f"{spread[i]:,.0f}", (i, spread[i]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=TEXT)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_max_over_percentile(ratios: dict[str, dict[str, float]], path, q: float = 99.9) -> None:
    """Fig. 2(b): max / p99.9 of every calibrated activation, pre-encoder vs encoder (log x)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.2), facecolor=SURFACE)
    _style(ax)
    rows = [("encoder", 0), ("pre", 1)]
    for (group, y), color in zip(rows, [SERIES[0], SERIES[1]]):
        values = list(ratios.get(group, {}).values())
        jitter = [((i * 37) % 11 - 5) / 40 for i in range(len(values))]
        ax.scatter(values, [y + j for j in jitter], s=14, color=color, edgecolors=SURFACE, linewidths=0.8, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.set_yticks([0, 1], ["encoder", "pre-enc."])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(f"max / p{q:g}", color=TEXT_2)
    ax.set_title("Observed max over the 99.9th percentile", color=TEXT, fontsize=10, loc="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
