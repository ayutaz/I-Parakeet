"""End-to-end smoke test of the command-line pipeline on a tiny random backbone."""

import json
import sys

import soundfile as sf
import torch

from scripts import infer, mine_pairs, train_selfdistill
from selfaccent.backbones.tiny_matcha import TinyMatcha, TinyMatchaConfig
from selfaccent.tokenizer import CharTokenizer

TEXTS = ["橋を見ました。", "箸が好きです。", "雨を見ました。", "飴が好きです。"]


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", *map(str, argv)])
    module.main()


def _tiny_backbone(path):
    tok = CharTokenizer.build(TEXTS)
    cfg = TinyMatchaConfig(
        n_vocab=len(tok), d_model=32, n_heads=2, enc_layers=1, ffn_dim=64,
        dp_channels=32, dec_channels=32, dec_blocks=2, dropout=0.0,
    )
    torch.manual_seed(0)
    model = TinyMatcha(cfg, torch.full((80,), -5.0), torch.full((80,), 2.0))
    with torch.no_grad():
        model.duration_predictor.proj.bias.fill_(1.5)
    model.save(path, extra={"tokenizer": tok.to_dict()})


def test_mine_train_infer_pipeline(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS * 3), encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    _run(mine_pairs, ["--sentences", corpus, "--out", pairs, "--min-count", 2, "--identity-ratio", 0.2], monkeypatch)
    lines = [json.loads(l) for l in pairs.read_text(encoding="utf-8").splitlines()]
    assert any(l["tag"] for l in lines)

    backbone = tmp_path / "model.pt"
    _tiny_backbone(backbone)
    out = tmp_path / "adapter"
    _run(
        train_selfdistill,
        ["--backbone", backbone, "--pairs", pairs, "--out", out, "--steps", 4, "--batch-size", 4,
         "--warmup-steps", 1, "--r", 2, "--alpha", 4, "--teacher-steps", 2, "--val-pairs", 2],
        monkeypatch,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trainable_params"] > 0
    assert (out / "adapter.pt").exists()

    wav = tmp_path / "out.wav"
    _run(
        infer,
        ["--backbone", backbone, "--adapter", out / "adapter.pt", "--text", "橋を見ました。",
         "--tag", "橋=ハシ'", "--out", wav, "--n-steps", 2, "--gl-iters", 2],
        monkeypatch,
    )
    audio, sr = sf.read(wav)
    assert sr == 22050 and len(audio) > 0
