import csv
import json

import numpy as np
import pytest
import torch

from conftest import randomize_, tiny_config
from iparakeet.analysis.range import calibrate
from iparakeet.deploy.device import eval_device_outputs, prepare_device_inputs
from iparakeet.deploy.emulate import run_input_lists_with_ort
from iparakeet.deploy.qnn import build_script, htp_config, read_raw_output, write_raw_input
from iparakeet.eval.manifest import ManifestEntry
from iparakeet.eval.runner import iter_features
from iparakeet.model.buckets import BucketSet
from iparakeet.model.parakeet import ParakeetCTC


class IdTokenizer:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def test_raw_io_roundtrip(tmp_path):
    x = np.random.default_rng(0).standard_normal((50, 80)).astype(np.float32)
    path = write_raw_input(x, tmp_path / "a.raw")
    assert path.stat().st_size == 50 * 80 * 4
    (tmp_path / "Result_0").mkdir()
    x.tofile(tmp_path / "Result_0" / "head_logits.raw")
    np.testing.assert_array_equal(read_raw_output(tmp_path, 0, (50, 80)), x)  # sanitized name
    x.tofile(tmp_path / "Result_0" / "head.logits.raw")
    np.testing.assert_array_equal(read_raw_output(tmp_path, 0, (50, 80)), x)
    with pytest.raises(FileNotFoundError):
        read_raw_output(tmp_path, 1, (50, 80))


def test_qnn_script_covers_every_bucket_and_enables_weight_sharing(tmp_path):
    script = build_script(["b3s", "b11s"], encodings="encodings.json", out_dir="qnn")
    for name in ("b3s", "b11s"):
        assert f"{name}.onnx" in script
    assert "--quantization_overrides encodings.json" in script
    assert "qnn-context-binary-generator" in script and "libQnnHtp.so" in script
    cfg = htp_config(["b3s", "b11s"], dsp_arch="v73")
    assert cfg["context"]["weight_sharing_enabled"] is True
    assert cfg["graphs"][0]["graph_names"] == ["b3s", "b11s"]


def _setup(tmp_path, seconds=(0.35, 0.8, 0.6)):
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    rng = np.random.default_rng(0)
    import soundfile as sf

    entries = []
    for i, s in enumerate(seconds):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, (rng.standard_normal(int(s * 16000)) * 0.1).astype(np.float32), 16000)
        entries.append(ManifestEntry(f"u{i}", str(path), s, "t1 t2"))
    stats = calibrate(model, lambda: iter_features(model, entries), n_bins=256, per_channel=("conv.dw",))
    return model, entries, stats


def test_device_pipeline_with_ort_standing_in_for_qnn_net_run(tmp_path):
    from iparakeet.deploy.encodings import build_encodings
    from iparakeet.deploy.onnx_graph import build_onnx
    from iparakeet.deploy.qdq import insert_qdq
    from iparakeet.sim.int_parakeet import IntParakeet
    from iparakeet.sim.recipe import RECIPES

    model, entries, stats = _setup(tmp_path)
    buckets = BucketSet([0.5, 1.0])
    mapping = prepare_device_inputs(model, entries, buckets, tmp_path / "inputs")
    assert [m["bucket"] for m in mapping] == ["b0.5s", "b1s", "b1s"]
    assert mapping[1]["index"] == 0 and mapping[2]["index"] == 1
    lists = sorted((tmp_path / "inputs").glob("*/input_list.txt"))
    assert len(lists) == 2 and lists[0].read_text().startswith("pre.in:=")

    sim = IntParakeet(model, stats, RECIPES["iparakeet_lut"], max_frames=100)
    graphs = {}
    for name, frames in zip(buckets.names, buckets.frames):
        g = build_onnx(model, frames, max_frames=100)
        graphs[name] = insert_qdq(g, build_encodings(sim, g))
    run_input_lists_with_ort(graphs, tmp_path / "inputs", tmp_path / "outputs")

    with (tmp_path / "latency.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "seconds"])
        for e in entries:
            w.writerow([e.utt_id, 0.01])
    summary = eval_device_outputs(mapping, tmp_path / "outputs", entries, IdTokenizer(), model.cfg.vocab_size + 1,
                                  model.cfg.blank_id, latency_csv=tmp_path / "latency.csv")
    assert summary["n_utts"] == 3
    assert summary["rtf"] == pytest.approx(0.03 / sum(e.duration for e in entries), rel=1e-6)
    # padded to 50 / 100 / 100 frames from 35 / 80 / 60
    assert summary["padding_ratio"] == pytest.approx((15 + 20 + 40) / 175)
    assert len(summary["hyps"]) == 3


def test_export_npu_cli_writes_graphs_encodings_and_script(tmp_path, monkeypatch):
    from scripts import export_npu

    model, entries, stats = _setup(tmp_path)
    stats.save(tmp_path / "stats.json")
    monkeypatch.setattr(export_npu, "load_model", lambda path: model)
    out = tmp_path / "npu"
    summary = export_npu.main(["--nemo", "x", "--stats", str(tmp_path / "stats.json"), "--recipe", "iparakeet_lut",
                               "--buckets", "0.5", "1.0", "--out", str(out)])
    assert summary["missing_encodings"] == 0 and summary["unquantized_inputs"] == 0
    assert (out / "b0.5s.onnx").exists() and (out / "b1s.onnx").exists()
    enc = json.loads((out / "encodings.json").read_text())
    assert enc["activation_encodings"]["L0.conv.dw"][0]["bitwidth"] == 16
    assert (out / "qnn_build.sh").read_text().count(".onnx") >= 2
