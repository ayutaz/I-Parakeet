"""End-to-end smoke test of the M1-M5 pipeline through the real CLIs (no monkeypatching).

A tiny random Parakeet-CTC is saved as a .nemo file next to a fake LibriSpeech tree, then
prepare_data -> eval_fp32 -> range_analysis -> run_ablation run exactly as documented.
"""

import io
import json
import tarfile

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import yaml

from conftest import randomize_, tiny_config
from iparakeet.model.parakeet import ParakeetCTC
from test_model_parakeet import _nemo_config


def _make_nemo(tmp_path):
    text = tmp_path / "text.txt"
    text.write_text("\n".join(["hello world", "the quick brown fox", "jumps over the lazy dog"] * 20))
    spm.SentencePieceTrainer.train(input=str(text), model_prefix=str(tmp_path / "tok"), vocab_size=30,
                                   model_type="bpe", bos_id=-1, eos_id=-1)
    cfg = tiny_config(vocab_size=30)
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(cfg)).eval()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    path = tmp_path / "tiny.nemo"
    with tarfile.open(path, "w") as tar:
        for name, data in (
            ("model_config.yaml", yaml.safe_dump(_nemo_config(cfg)).encode()),
            ("model_weights.ckpt", buf.getvalue()),
            ("abc_tokenizer.model", (tmp_path / "tok.model").read_bytes()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _fake_librispeech(root, split, n, rng):
    chapter = root / split / "1" / "2"
    chapter.mkdir(parents=True)
    lines = []
    for i in range(n):
        utt = f"1-2-{i:04d}"
        sf.write(chapter / f"{utt}.flac", (rng.standard_normal(4000 + 1500 * i) * 0.1).astype(np.float32), 16000)
        lines.append(f"{utt} HELLO WORLD")
    (chapter / "1-2.trans.txt").write_text("\n".join(lines) + "\n")


def test_pipeline_runs_end_to_end(tmp_path):
    from scripts import eval_fp32, prepare_data, range_analysis, run_ablation

    rng = np.random.default_rng(0)
    nemo = _make_nemo(tmp_path)
    root = tmp_path / "LibriSpeech"
    for split, n in (("dev-other", 3), ("test-clean", 2), ("test-other", 3)):
        _fake_librispeech(root, split, n, rng)
    manifests = tmp_path / "manifests"
    prepare_data.main(["--librispeech-root", str(root), "--out", str(manifests)])

    fp32 = eval_fp32.main(["--nemo", str(nemo), "--manifest", str(manifests / "test-other.jsonl"), "--out", str(tmp_path / "fp32")])
    assert fp32["n_utts"] == 3

    summary = range_analysis.main(["--nemo", str(nemo), "--manifest", str(manifests / "dev-other.jsonl"),
                                   "--out", str(tmp_path / "range"), "--bins", "128"])
    assert len(summary["bn_scale_spread"]) == 2

    results = run_ablation.main([
        "--nemo", str(nemo), "--stats", str(tmp_path / "range" / "calibration_stats.json"),
        "--manifests", str(manifests / "test-clean.jsonl"), str(manifests / "test-other.jsonl"),
        "--recipes", "ibert_recipe", "naive_int8", "iparakeet", "--out", str(tmp_path / "ablation"), "--max-frames", "200",
    ])
    assert set(results["iparakeet"]) == {"test-clean", "test-other"}
    checks = json.loads((tmp_path / "ablation" / "checks.json").read_text())
    assert set(checks) == {"table2_order", "table3_order"}
