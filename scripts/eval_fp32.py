"""M1: FP32 WER of Parakeet-CTC on a manifest (standalone implementation, Whisper normalizer).

    uv run python -m scripts.eval_fp32 --nemo parakeet-ctc-0.6b.nemo \
        --manifest data/manifests/test-other.jsonl --out results/fp32/test-other
"""

import argparse
import json
import time
from pathlib import Path

import torch

from iparakeet.eval.manifest import read_manifest
from iparakeet.eval.runner import transcribe_manifest
from iparakeet.eval.text import corpus_wer
from iparakeet.model.load_nemo import load_nemo


def load_model(path):
    bundle = load_nemo(path)
    return bundle.model, bundle.tokenizer


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    model, tokenizer = load_model(args.nemo)
    entries = read_manifest(args.manifest)
    start = time.perf_counter()
    hyps = transcribe_manifest(model, tokenizer, entries, args.batch_size, args.device)
    elapsed = time.perf_counter() - start
    result = corpus_wer([e.text for e in entries], hyps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hyps.jsonl").open("w") as f:
        for e, h in zip(entries, hyps):
            f.write(json.dumps({"utt_id": e.utt_id, "ref": e.text, "hyp": h}) + "\n")
    summary = {
        "manifest": str(args.manifest),
        "wer": result.wer,
        "errors": result.errors,
        "ref_words": result.ref_words,
        "n_utts": result.n_utts,
        "n_skipped": result.n_skipped,
        "seconds": elapsed,
        "audio_seconds": sum(e.duration for e in entries),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
