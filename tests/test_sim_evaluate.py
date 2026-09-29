import json

import numpy as np
import soundfile as sf
import torch

from conftest import randomize_, tiny_config
from iparakeet.analysis.range import calibrate
from iparakeet.eval.manifest import ManifestEntry, write_manifest
from iparakeet.eval.runner import iter_features
from iparakeet.model.buckets import BucketSet
from iparakeet.model.parakeet import ParakeetCTC
from iparakeet.sim.evaluate import evaluate_recipe
from iparakeet.sim.recipe import RECIPES


class IdTokenizer:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def _setup(tmp_path, seconds=(0.5, 0.9, 0.7)):
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    rng = np.random.default_rng(0)
    entries = []
    for i, s in enumerate(seconds):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, (rng.standard_normal(int(s * 16000)) * 0.1).astype(np.float32), 16000)
        entries.append(ManifestEntry(f"u{i}", str(path), s, "t1 t2"))
    stats = calibrate(model, lambda: iter_features(model, entries), n_bins=256, per_channel=("conv.dw",))
    return model, entries, stats


def test_evaluate_recipe_returns_wer_and_hypotheses(tmp_path):
    model, entries, stats = _setup(tmp_path)
    out = evaluate_recipe(model, IdTokenizer(), stats, RECIPES["iparakeet"], entries, max_frames=100)
    assert set(out) >= {"wer", "hyps", "seconds", "padding_ratio", "n_utts"}
    assert len(out["hyps"]) == 3 and out["padding_ratio"] == 0.0


def test_lossless_recipe_reproduces_fp32_transcripts(tmp_path):
    model, entries, stats = _setup(tmp_path)
    fp = []
    for feats, lengths in iter_features(model, entries):
        with torch.no_grad():
            logits, _ = model.forward_features(feats, lengths)
        fp.append(logits[0].argmax(-1))
    out = evaluate_recipe(model, IdTokenizer(), stats, RECIPES["lossless_int16"], entries, max_frames=100)
    agree = [
        h == IdTokenizer().decode([t for j, t in enumerate(ids.tolist()) if t != model.cfg.blank_id and (j == 0 or t != ids[j - 1])])
        for h, ids in zip(out["hyps"], fp)
    ]
    assert sum(agree) >= 2


def test_bucket_padding_is_applied_and_accounted(tmp_path):
    model, entries, stats = _setup(tmp_path)
    out = evaluate_recipe(model, IdTokenizer(), stats, RECIPES["iparakeet"], entries, max_frames=100, buckets=BucketSet([0.6, 1.0]))
    # 50, 90, 70 frames -> 60, 100, 100: (10 + 10 + 30) / 210
    assert abs(out["padding_ratio"] - 50 / 210) < 1e-9


def test_eval_sim_and_run_ablation_clis(tmp_path, monkeypatch):
    from scripts import eval_sim, run_ablation

    model, entries, stats = _setup(tmp_path)
    write_manifest(tmp_path / "test-other.jsonl", entries)
    write_manifest(tmp_path / "test-clean.jsonl", entries[:2])
    stats.save(tmp_path / "stats.json")
    monkeypatch.setattr(eval_sim, "load_model", lambda path: (model, IdTokenizer()))
    out = tmp_path / "sim"
    eval_sim.main([
        "--nemo", "x", "--stats", str(tmp_path / "stats.json"), "--manifest", str(tmp_path / "test-other.jsonl"),
        "--recipes", "iparakeet", "naive_int8", "--out", str(out), "--max-frames", "100",
    ])
    summary = json.loads((out / "iparakeet" / "test-other" / "summary.json").read_text())
    assert summary["recipe"] == "iparakeet" and "wer" in summary
    assert (out / "naive_int8" / "test-other" / "hyps.jsonl").exists()

    monkeypatch.setattr(run_ablation.eval_sim, "load_model", lambda path: (model, IdTokenizer()))
    run_ablation.main([
        "--nemo", "x", "--stats", str(tmp_path / "stats.json"),
        "--manifests", str(tmp_path / "test-clean.jsonl"), str(tmp_path / "test-other.jsonl"),
        "--recipes", "iparakeet", "naive_int8", "ibert_recipe", "--out", str(tmp_path / "ablation"), "--max-frames", "100",
    ])
    results = json.loads((tmp_path / "ablation" / "results.json").read_text())
    assert set(results) == {"iparakeet", "naive_int8", "ibert_recipe"}
    assert "test-other" in results["iparakeet"]
    assert (tmp_path / "ablation" / "table2.md").read_text().startswith("|")
