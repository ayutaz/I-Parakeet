"""M7: WER / RTF / padding overhead from qnn-net-run outputs (Result_<i>/head.logits.raw per bucket).

    uv run python -m scripts.eval_device --nemo data/parakeet-ctc-0.6b.nemo \
        --manifest data/manifests/test-other.jsonl --mapping results/device_inputs/mapping.json \
        --results results/device_outputs --latency-csv results/device_latency.csv --out results/device.json

latency CSV: columns utt_id, seconds (per-utterance NPU time measured by the runner).
"""

import argparse
import json
from pathlib import Path

from iparakeet.deploy.device import eval_device_outputs
from iparakeet.eval.manifest import read_manifest
from iparakeet.model.load_nemo import load_nemo


def load_model(path):
    bundle = load_nemo(path)
    return bundle.model, bundle.tokenizer


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--latency-csv", default=None)
    parser.add_argument("--output-name", default="head.logits")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    model, tokenizer = load_model(args.nemo)
    mapping = json.loads(Path(args.mapping).read_text())
    summary = eval_device_outputs(mapping, args.results, read_manifest(args.manifest), tokenizer, model.cfg.vocab_size + 1,
                                  model.cfg.blank_id, args.latency_csv, model.cfg.subsampling_factor, args.output_name)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("hyps", "utt_ids")}, indent=2))
    return summary


if __name__ == "__main__":
    main()
