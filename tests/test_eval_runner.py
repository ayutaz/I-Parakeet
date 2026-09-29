import json

import numpy as np
import soundfile as sf
import torch

from conftest import randomize_, tiny_config
from iparakeet.eval.manifest import ManifestEntry, write_manifest
from iparakeet.eval.runner import load_audio, transcribe_manifest
from iparakeet.model.decoding import ctc_greedy_ids
from iparakeet.model.parakeet import ParakeetCTC


class IdTokenizer:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def test_load_audio_resamples_to_16k(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    sf.write(tmp_path / "a.wav", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    audio = load_audio(tmp_path / "a.wav", 16000)
    assert audio.shape == (16000,)
    spectrum = np.abs(np.fft.rfft(audio))
    assert abs(np.argmax(spectrum) * 16000 / len(audio) - 440) < 2


def test_load_audio_downmixes_stereo(tmp_path):
    stereo = np.stack([np.ones(1600), -np.ones(1600)], axis=1).astype(np.float32) * 0.5
    sf.write(tmp_path / "s.wav", stereo, 16000)
    assert np.allclose(load_audio(tmp_path / "s.wav", 16000), 0.0)


def _entries(tmp_path, seconds):
    rng = np.random.default_rng(0)
    entries = []
    for i, s in enumerate(seconds):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, (rng.standard_normal(int(s * 16000)) * 0.1).astype(np.float32), 16000)
        entries.append(ManifestEntry(f"u{i}", str(path), s, "ref"))
    return entries


def test_batched_transcription_matches_one_by_one(tmp_path):
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    entries = _entries(tmp_path, [0.7, 1.3, 0.4])
    batched = transcribe_manifest(model, IdTokenizer(), entries, batch_size=3)
    single = []
    for e in entries:
        audio = torch.from_numpy(load_audio(e.audio_filepath, 16000))[None]
        with torch.no_grad():
            logits, lengths = model(audio, torch.tensor([audio.shape[1]]))
        single.append(IdTokenizer().decode(ctc_greedy_ids(logits[0, : lengths.item()], model.cfg.blank_id)))
    assert batched == single


def test_eval_fp32_cli_writes_hypotheses_and_wer(tmp_path, monkeypatch):
    from scripts import eval_fp32

    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    entries = _entries(tmp_path, [0.5, 0.6])
    write_manifest(tmp_path / "m.jsonl", entries)
    monkeypatch.setattr(eval_fp32, "load_model", lambda path: (model, IdTokenizer()))
    out = tmp_path / "out"
    eval_fp32.main(["--nemo", "unused.nemo", "--manifest", str(tmp_path / "m.jsonl"), "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_utts"] == 2 and 0.0 <= summary["wer"]
    hyps = [json.loads(line) for line in (out / "hyps.jsonl").read_text().splitlines()]
    assert [h["utt_id"] for h in hyps] == ["u0", "u1"]
