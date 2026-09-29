"""M7: device inputs per bucket (qnn-net-run input lists) and evaluation of device outputs."""

import csv
import json
from pathlib import Path

import torch

from iparakeet.deploy.qnn import read_raw_output, write_raw_input
from iparakeet.eval.manifest import ManifestEntry
from iparakeet.eval.runner import iter_features
from iparakeet.eval.text import corpus_wer
from iparakeet.model.buckets import BucketSet, pad_features
from iparakeet.model.decoding import ctc_greedy_ids
from iparakeet.model.layers import subsampled_length


def prepare_device_inputs(model, entries: list[ManifestEntry], buckets: BucketSet, out_dir, silence=None) -> list[dict]:
    """Front-end features (CPU), silence-padded to their bucket, written as raw float32 (T, n_mels)."""
    out = Path(out_dir)
    lists: dict[str, list[str]] = {}
    mapping = []
    for entry, (feats, _) in zip(entries, iter_features(model, entries)):
        real = feats.shape[-1]
        idx = buckets.route(real)
        name, frames = buckets.names[idx], buckets.frames[idx]
        padded = pad_features(feats[0].cpu(), frames, silence).T.numpy()
        path = write_raw_input(padded, out / name / f"{entry.utt_id}.raw")
        lists.setdefault(name, []).append(f"pre.in:={path.resolve()}")
        mapping.append({"utt_id": entry.utt_id, "bucket": name, "index": len(lists[name]) - 1,
                        "real_frames": real, "padded_frames": frames, "duration": entry.duration})
    for name, lines in lists.items():
        (out / name / "input_list.txt").write_text("\n".join(lines) + "\n")
    (out / "mapping.json").write_text(json.dumps(mapping, indent=2))
    return mapping


def eval_device_outputs(mapping, results_root, entries, tokenizer, n_classes: int, blank: int,
                        latency_csv=None, subsampling_factor: int = 8, output_name: str = "head.logits") -> dict:
    by_id = {e.utt_id: e for e in entries}
    refs, hyps = [], []
    for item in mapping:
        L = subsampled_length(item["padded_frames"], subsampling_factor)
        logits = read_raw_output(Path(results_root) / item["bucket"], item["index"], (L, n_classes), output_name)
        hyps.append(tokenizer.decode(ctc_greedy_ids(torch.from_numpy(logits), blank)))
        refs.append(by_id[item["utt_id"]].text)
    wer = corpus_wer(refs, hyps)
    real = sum(m["real_frames"] for m in mapping)
    padded = sum(m["padded_frames"] for m in mapping)
    audio = sum(by_id[m["utt_id"]].duration for m in mapping)
    summary = {"wer": wer.wer, "n_utts": wer.n_utts, "padding_ratio": (padded - real) / real, "hyps": hyps,
               "utt_ids": [m["utt_id"] for m in mapping]}
    if latency_csv:
        with Path(latency_csv).open() as f:
            seconds = {row["utt_id"]: float(row["seconds"]) for row in csv.DictReader(f)}
        summary["rtf"] = sum(seconds[m["utt_id"]] for m in mapping) / audio
    return summary
