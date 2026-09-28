import json

import numpy as np
import soundfile as sf

from iparakeet.eval.manifest import ManifestEntry, read_manifest, write_manifest


def _fake_librispeech(root, split, n):
    chapter = root / split / "10" / "20"
    chapter.mkdir(parents=True)
    lines = []
    for i in range(n):
        utt = f"10-20-{i:04d}"
        sf.write(chapter / f"{utt}.flac", np.zeros(1600 * (i + 1), dtype=np.float32), 16000)
        lines.append(f"{utt} WORD {i}")
    (chapter / "10-20.trans.txt").write_text("\n".join(lines) + "\n")


def test_prepare_data_builds_manifests_for_each_split(tmp_path):
    from scripts import prepare_data

    root = tmp_path / "LibriSpeech"
    _fake_librispeech(root, "dev-other", 2)
    _fake_librispeech(root, "test-other", 3)
    out = tmp_path / "manifests"
    prepare_data.main(["--librispeech-root", str(root), "--splits", "dev-other", "test-other", "--out", str(out)])
    assert len(read_manifest(out / "dev-other.jsonl")) == 2
    test_other = read_manifest(out / "test-other.jsonl")
    assert [e.text for e in test_other] == ["WORD 0", "WORD 1", "WORD 2"]
    assert test_other[2].duration == 0.3


def test_choose_buckets_reports_step_near_target(tmp_path, capsys):
    from scripts import choose_buckets

    entries = [ManifestEntry(f"u{i}", "x", float(d), "t") for i, d in enumerate(range(3, 36))]
    write_manifest(tmp_path / "m.jsonl", entries)
    result = choose_buckets.main(["--manifest", str(tmp_path / "m.jsonl"), "--target", "0.0", "--steps", "1", "2", "4"])
    assert result["step"] == 1.0
    assert result["padding_ratio"] == 0.0
    assert result["buckets"][0] == 3.0 and result["buckets"][-1] == 35.0
    assert json.loads(capsys.readouterr().out)["step"] == 1.0
