"""M5: WER of the integer-only simulator for one or more recipes on a manifest.

    uv run python -m scripts.eval_sim --nemo data/parakeet-ctc-0.6b.nemo \
        --stats results/range/calibration_stats.json --manifest data/manifests/test-other.jsonl \
        --recipes iparakeet naive_int8 ibert_recipe --out results/sim
"""

import argparse
import json
from pathlib import Path

import torch

from iparakeet.analysis.range import CalibrationStats
from iparakeet.eval.manifest import read_manifest
from iparakeet.model.buckets import uniform_buckets
from iparakeet.model.load_nemo import load_nemo
from iparakeet.sim.evaluate import evaluate_recipe
from iparakeet.sim.recipe import RECIPES


def load_model(path):
    bundle = load_nemo(path)
    return bundle.model, bundle.tokenizer


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--stats", required=True, help="calibration_stats.json from scripts.range_analysis")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=3500, help="longest input in feature frames (35 s)")
    parser.add_argument("--bucket-step", type=float, default=None, help="pad to uniform 3-35 s buckets (seconds)")
    parser.add_argument("--max-utts", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def run_one(model, tokenizer, stats, recipe_name, manifest, args) -> dict:
    entries = read_manifest(manifest)[: args.max_utts]
    buckets = uniform_buckets(3.0, 35.0, args.bucket_step) if args.bucket_step else None
    result = evaluate_recipe(model, tokenizer, stats, RECIPES[recipe_name], entries, args.max_frames, buckets, device=args.device)
    out = Path(args.out) / recipe_name / Path(manifest).stem
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hyps.jsonl").open("w") as f:
        for e, h in zip(entries, result["hyps"]):
            f.write(json.dumps({"utt_id": e.utt_id, "ref": e.text, "hyp": h}) + "\n")
    summary = {k: v for k, v in result.items() if k != "hyps"}
    summary["manifest"] = str(manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{recipe_name:>24s} {Path(manifest).stem:>18s}  WER {100 * result['wer']:.2f}%")
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recipes", nargs="+", default=["iparakeet"], choices=sorted(RECIPES))
    args = parser.parse_args(argv)
    model, tokenizer = load_model(args.nemo)
    stats = CalibrationStats.load(args.stats)
    return {name: run_one(model, tokenizer, stats, name, args.manifest, args) for name in args.recipes}


if __name__ == "__main__":
    main()
