import csv
import json

import numpy as np
import soundfile as sf
import torch

from conftest import randomize_, tiny_config
from iparakeet.eval.manifest import ManifestEntry, write_manifest
from iparakeet.model.parakeet import ParakeetCTC


class IdTokenizer:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def test_prepare_and_eval_device_clis(tmp_path, monkeypatch):
    from scripts import eval_device, prepare_device_inputs

    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    rng = np.random.default_rng(0)
    entries = []
    for i, s in enumerate((0.35, 0.8)):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, (rng.standard_normal(int(s * 16000)) * 0.1).astype(np.float32), 16000)
        entries.append(ManifestEntry(f"u{i}", str(path), s, "t1"))
    write_manifest(tmp_path / "m.jsonl", entries)
    monkeypatch.setattr(prepare_device_inputs, "load_model", lambda path: model)
    prepare_device_inputs.main(["--nemo", "x", "--manifest", str(tmp_path / "m.jsonl"), "--buckets", "0.5", "1.0",
                                "--out", str(tmp_path / "inputs")])
    mapping = json.loads((tmp_path / "inputs" / "mapping.json").read_text())
    assert [m["bucket"] for m in mapping] == ["b0.5s", "b1s"]

    for m in mapping:  # fake device outputs: all-blank logits
        L = ((m["padded_frames"] - 1) // 2 // 2) // 2 + 1
        folder = tmp_path / "outputs" / m["bucket"] / f"Result_{m['index']}"
        folder.mkdir(parents=True)
        logits = np.zeros((L, 17), dtype=np.float32)
        logits[:, 16] = 1.0
        logits.tofile(folder / "head_logits.raw")
    with (tmp_path / "lat.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "seconds"])
        w.writerows([["u0", 0.02], ["u1", 0.03]])
    monkeypatch.setattr(eval_device, "load_model", lambda path: (model, IdTokenizer()))
    summary = eval_device.main(["--nemo", "x", "--manifest", str(tmp_path / "m.jsonl"),
                                "--mapping", str(tmp_path / "inputs" / "mapping.json"),
                                "--results", str(tmp_path / "outputs"), "--latency-csv", str(tmp_path / "lat.csv"),
                                "--out", str(tmp_path / "device.json")])
    assert summary["wer"] == 1.0  # every word deleted
    assert abs(summary["rtf"] - 0.05 / 1.15) < 1e-9
    assert json.loads((tmp_path / "device.json").read_text())["n_utts"] == 2
