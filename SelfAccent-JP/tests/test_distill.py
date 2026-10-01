import torch

from selfaccent.backbones.tiny_matcha import TinyMatcha, TinyMatchaBackbone, TinyMatchaConfig
from selfaccent.distill import (
    AdapterConfig,
    DistillConfig,
    TAG_SYMBOLS,
    generate_teacher_targets,
    load_adapter,
    prepare_student,
    save_adapter,
    train_student,
)
from selfaccent.lora import LoRALinear
from selfaccent.mining import DistillPair
from selfaccent.tags import PHON_END, PHON_START
from selfaccent.tokenizer import CharTokenizer

TEXTS = ["橋を渡る。", "箸で食べる。", "端を歩く。"]
PAIRS = [
    DistillPair("橋を渡る。", f"{PHON_START}ハシ'{PHON_END}を渡る。", "橋", "ハシ'", 0, 1),
    DistillPair("箸で食べる。", f"{PHON_START}ハ'シ{PHON_END}で食べる。", "箸", "ハ'シ", 0, 1),
    DistillPair("端を歩く。", "端を歩く。"),
]


def _backbone(seed=0):
    tok = CharTokenizer.build(TEXTS)
    cfg = TinyMatchaConfig(
        n_vocab=len(tok), n_mels=8, d_model=32, n_heads=2, enc_layers=2, ffn_dim=64,
        dp_channels=32, dec_channels=32, dec_blocks=2, dropout=0.0,
    )
    torch.manual_seed(seed)
    model = TinyMatcha(cfg, torch.zeros(8), torch.ones(8))
    # give the toy model non-trivial durations
    with torch.no_grad():
        model.duration_predictor.proj.bias.fill_(1.0)
    return TinyMatchaBackbone(model, tok, n_steps=3)


def _base_params(model):
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not any(k in n for k in ("lora_", "extra"))
    }


def test_teacher_targets_are_deterministic_per_pair():
    bb = _backbone()
    a = generate_teacher_targets(bb, PAIRS, seed=3)
    b = generate_teacher_targets(bb, PAIRS, seed=3)
    assert len(a) == len(PAIRS)
    for x, y in zip(a, b):
        assert x.shape[0] == 8
        torch.testing.assert_close(x, y)


def test_prepare_student_freezes_backbone_and_keeps_plain_outputs():
    bb = _backbone()
    plain = bb.synthesize(["橋を渡る。"], seeds=[0])[0]
    names = prepare_student(bb, AdapterConfig(r=2, alpha=4, dropout=0.0))
    assert names and all(("lora_" in n) or n.endswith("extra") for n in names)
    assert isinstance(bb.model.encoder.layers[0].attn.q_proj, LoRALinear)
    torch.testing.assert_close(bb.synthesize(["橋を渡る。"], seeds=[0])[0], plain)
    assert not bb.tokenizer.unknown_symbols(PAIRS[0].student_text)
    assert set(TAG_SYMBOLS) <= set(bb.tokenizer.symbols)


def test_training_reduces_loss_and_leaves_backbone_untouched():
    bb = _backbone()
    targets = generate_teacher_targets(bb, PAIRS, seed=0)
    prepare_student(bb, AdapterConfig(r=4, alpha=8, dropout=0.0))
    before = _base_params(bb.model)
    history = train_student(bb, PAIRS, targets, DistillConfig(steps=40, batch_size=3, lr=1e-2, warmup_steps=0, seed=0))
    assert len(history) == 40
    first = sum(h["prior"] for h in history[:5]) / 5
    last = sum(h["prior"] for h in history[-5:]) / 5
    assert last < first
    after = _base_params(bb.model)
    for n in before:
        torch.testing.assert_close(after[n], before[n], msg=n)


def test_adapter_round_trip(tmp_path):
    bb = _backbone()
    targets = generate_teacher_targets(bb, PAIRS, seed=0)
    acfg = AdapterConfig(r=2, alpha=4, dropout=0.0)
    prepare_student(bb, acfg)
    train_student(bb, PAIRS, targets, DistillConfig(steps=3, batch_size=2, lr=1e-2, warmup_steps=0))
    path = tmp_path / "adapter.pt"
    save_adapter(bb, acfg, path)
    size = path.stat().st_size

    fresh = _backbone()
    load_adapter(fresh, path)
    text = PAIRS[0].student_text
    torch.testing.assert_close(
        fresh.synthesize([text], seeds=[1])[0], bb.synthesize([text], seeds=[1])[0]
    )
    full = sum(p.numel() for p in fresh.model.parameters()) * 4
    assert size < full


def test_load_adapter_rejects_other_backbone_vocab(tmp_path):
    bb = _backbone()
    acfg = AdapterConfig(r=2, alpha=4)
    prepare_student(bb, acfg)
    path = tmp_path / "adapter.pt"
    save_adapter(bb, acfg, path)
    other = _backbone()
    other.tokenizer.extend(["余"])
    try:
        load_adapter(other, path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a vocabulary mismatch error")
