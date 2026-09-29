"""M3: layer-wise activation range analysis (paper Sec. 3.3, Fig. 2) and calibration statistics.

    uv run python -m scripts.range_analysis --nemo data/parakeet-ctc-0.6b.nemo \
        --manifest data/manifests/dev-other.jsonl --out results/range

Writes calibration_stats.json (reused by the integer simulator and the NPU encodings),
summary.json, fig2a_bn_spread.png and fig2b_max_over_p999.png.
"""

import argparse
import json
from pathlib import Path

import torch

from iparakeet.analysis.plots import plot_bn_spread, plot_max_over_percentile
from iparakeet.analysis.range import bn_scale_spread, calibrate, fp16_overflow, max_over_percentile
from iparakeet.eval.manifest import read_manifest
from iparakeet.eval.runner import iter_features
from iparakeet.model.load_nemo import load_nemo


def load_model(path):
    return load_nemo(path).model


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--manifest", required=True, help="calibration set (paper: LibriSpeech dev-other)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-utts", type=int, default=None)
    parser.add_argument("--bins", type=int, default=4096)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    model = load_model(args.nemo).to(args.device).eval()
    entries = read_manifest(args.manifest)[: args.max_utts]
    stats = calibrate(model, lambda: iter_features(model, entries, args.device), args.bins, per_channel=("conv.dw",))
    out = Path(args.out)
    stats.save(out / "calibration_stats.json")

    spread = bn_scale_spread(model)
    ratios = max_over_percentile(stats, args.percentile)
    summary = {
        "n_utts": len(entries),
        "bn_scale_spread": spread,
        f"max_over_p{args.percentile:g}": ratios,
        "pre_ratio_range": [min(ratios["pre"].values(), default=0), max(ratios["pre"].values(), default=0)],
        "encoder_ratio_range": [min(ratios["encoder"].values(), default=0), max(ratios["encoder"].values(), default=0)],
        "fp16_overflow": fp16_overflow(stats),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_bn_spread(spread, out / "fig2a_bn_spread.png")
    plot_max_over_percentile(ratios, out / "fig2b_max_over_p999.png", args.percentile)
    print(json.dumps({k: v for k, v in summary.items() if k != f"max_over_p{args.percentile:g}"}, indent=2))
    return summary


if __name__ == "__main__":
    main()
