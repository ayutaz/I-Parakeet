"""M5: run the paper's Table 2 / Table 3 configurations and compare with the paper.

    uv run python -m scripts.run_ablation --nemo data/parakeet-ctc-0.6b.nemo \
        --stats results/range/calibration_stats.json \
        --manifests data/manifests/test-clean.jsonl data/manifests/test-other.jsonl \
        --out results/ablation

Writes results.json (WER % per recipe and test set), table2.md, table3.md and checks.json.
"""

import argparse
import json
from pathlib import Path

from iparakeet.analysis.range import CalibrationStats
from iparakeet.sim.recipe import PAPER_TABLE2, PAPER_TABLE3, RECIPES
from iparakeet.sim.report import check_table2_order, check_table3_order, table2_markdown, table3_markdown
from scripts import eval_sim

DEFAULT_RECIPES = list(dict.fromkeys(list(PAPER_TABLE2) + [r.recipe for r in PAPER_TABLE3]))


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    eval_sim.add_common_args(parser)
    parser.add_argument("--manifests", nargs="+", required=True, help="stems must be test-clean / test-other / commonvoice-test")
    parser.add_argument("--recipes", nargs="+", default=DEFAULT_RECIPES, choices=sorted(RECIPES))
    parser.add_argument("--fp32", default=None, help="optional JSON {testset: WER %} of the FP32 model")
    args = parser.parse_args(argv)

    model, tokenizer = eval_sim.load_model(args.nemo)
    stats = CalibrationStats.load(args.stats)
    results: dict[str, dict[str, float]] = {}
    for manifest in args.manifests:
        for name in args.recipes:
            summary = eval_sim.run_one(model, tokenizer, stats, name, manifest, args)
            results.setdefault(name, {})[Path(manifest).stem] = 100 * summary["wer"]
    fp32 = json.loads(Path(args.fp32).read_text()) if args.fp32 else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "table2.md").write_text(table2_markdown(results, fp32))
    (out / "table3.md").write_text(table3_markdown(results))
    checks = {"table2_order": check_table2_order(results, fp32), "table3_order": check_table3_order(results)}
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return results


if __name__ == "__main__":
    main()
