import json

import numpy as np
import soundfile as sf
import torch

from conftest import randomize_, tiny_config
from iparakeet.analysis.plots import plot_bn_spread, plot_max_over_percentile
from iparakeet.eval.manifest import ManifestEntry, write_manifest
from iparakeet.model.parakeet import ParakeetCTC


def test_plots_write_png_files(tmp_path):
    plot_bn_spread([3.0, 10759.0, 12.0], tmp_path / "a.png")
    plot_max_over_percentile({"pre": {"pre.conv0": 15.0}, "encoder": {"L0.res1": 2.5, "L0.res2": 3.0}}, tmp_path / "b.png")
    for name in ("a.png", "b.png"):
        data = (tmp_path / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 1000


def test_range_analysis_cli_writes_stats_summary_and_figures(tmp_path, monkeypatch):
    from scripts import range_analysis

    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    rng = np.random.default_rng(0)
    entries = []
    for i in range(3):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, (rng.standard_normal(8000 + 1000 * i) * 0.1).astype(np.float32), 16000)
        entries.append(ManifestEntry(f"u{i}", str(path), 0.5, "x"))
    write_manifest(tmp_path / "dev.jsonl", entries)
    monkeypatch.setattr(range_analysis, "load_model", lambda path: model)
    out = tmp_path / "range"
    range_analysis.main(["--nemo", "x.nemo", "--manifest", str(tmp_path / "dev.jsonl"), "--out", str(out), "--bins", "64"])
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["bn_scale_spread"]) == 2
    assert set(summary["max_over_p99.9"]) == {"pre", "encoder"}
    assert summary["n_utts"] == 3
    assert "fp16_overflow" in summary
    assert (out / "calibration_stats.json").exists()
    assert (out / "fig2a_bn_spread.png").exists() and (out / "fig2b_max_over_p999.png").exists()
