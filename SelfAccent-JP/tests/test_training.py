import torch

from selfaccent.backbones.tiny_matcha import TinyMatcha, TinyMatchaBackbone, TinyMatchaConfig
from selfaccent.tokenizer import CharTokenizer
from selfaccent.training import TrainConfig, compute_mel_stats, length_batches, train_backbone


def test_length_batches_cover_everything_once_and_group_lengths():
    lengths = [5, 50, 6, 49, 7, 48, 8, 47]
    batches = list(length_batches(lengths, batch_size=2, seed=0, bucket_size=8))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(8))
    for b in batches:
        ls = [lengths[i] for i in b]
        assert max(ls) - min(ls) <= 3


def test_compute_mel_stats():
    mels = [torch.full((2, 3), 1.0), torch.full((2, 5), 3.0)]
    mean, std = compute_mel_stats(mels)
    torch.testing.assert_close(mean, torch.tensor([2.25, 2.25]))
    assert (std > 0.9).all()


def test_train_backbone_reduces_loss():
    texts = ["あい。", "いう。", "うえ。", "えお。"]
    tok = CharTokenizer.build(texts)
    torch.manual_seed(0)
    mels = [torch.randn(4, 12) + i for i in range(4)]
    mean, std = compute_mel_stats(mels)
    cfg = TinyMatchaConfig(
        n_vocab=len(tok), n_mels=4, d_model=32, n_heads=2, enc_layers=1, ffn_dim=64,
        dp_channels=32, dec_channels=32, dec_blocks=2, dropout=0.0,
    )
    bb = TinyMatchaBackbone(TinyMatcha(cfg, mean, std), tok)
    hist = train_backbone(bb, texts, mels, TrainConfig(steps=60, batch_size=4, lr=3e-3, warmup_steps=5))
    assert len(hist) == 60
    assert sum(h["loss"] for h in hist[-5:]) < sum(h["loss"] for h in hist[:5])
