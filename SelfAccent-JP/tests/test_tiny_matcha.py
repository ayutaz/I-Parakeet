import pytest
import torch

from selfaccent.backbones.tiny_matcha import TinyMatcha, TinyMatchaConfig, sequence_mask, upsample_by_durations
from selfaccent.lora import ExtendedEmbedding, LoRALinear, inject_lora


def _small_model(n_vocab=12, n_mels=8):
    cfg = TinyMatchaConfig(
        n_vocab=n_vocab, n_mels=n_mels, d_model=32, n_heads=2, enc_layers=2, ffn_dim=64,
        dp_channels=32, dec_channels=32, dec_blocks=2, dropout=0.0,
    )
    torch.manual_seed(0)
    return TinyMatcha(cfg, mel_mean=torch.zeros(n_mels), mel_std=torch.ones(n_mels))


def _batch(n_mels=8):
    ids = torch.tensor([[2, 4, 5, 6, 3], [2, 7, 3, 0, 0]])
    id_lens = torch.tensor([5, 3])
    mels = torch.randn(2, n_mels, 20)
    mel_lens = torch.tensor([20, 11])
    return ids, id_lens, mels, mel_lens


def test_sequence_mask():
    m = sequence_mask(torch.tensor([2, 4]), 5)
    assert m.tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]]


def test_upsample_by_durations():
    mu = torch.arange(6.0).view(1, 2, 3)  # 2 channels, 3 tokens
    dur = torch.tensor([[1, 2, 0]])
    out = upsample_by_durations(mu, dur)
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [0.0, 1.0, 1.0]


def test_loss_terms_are_finite_and_backpropagate():
    model = _small_model()
    losses = model.compute_loss(*_batch())
    assert set(losses) == {"loss", "prior", "duration", "cfm"}
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss"].backward()
    assert model.decoder.conv_in.weight.grad is not None
    assert model.encoder.embed.weight.grad is not None


def test_alignment_durations_cover_every_frame():
    model = _small_model()
    ids, id_lens, mels, mel_lens = _batch()
    durations = model.align(ids, id_lens, mels, mel_lens)
    assert durations.sum(1).tolist() == mel_lens.tolist()
    assert (durations[0, :5] >= 1).all() and (durations[1, 3:] == 0).all()


def test_synthesize_is_deterministic_with_generator():
    model = _small_model().eval()
    ids, id_lens, _, _ = _batch()
    a = model.synthesize(ids, id_lens, n_steps=3, generator=torch.Generator().manual_seed(1))
    b = model.synthesize(ids, id_lens, n_steps=3, generator=torch.Generator().manual_seed(1))
    assert len(a.mels) == 2
    for x, y, d in zip(a.mels, b.mels, a.durations):
        assert x.shape[0] == 8 and x.shape[1] == int(d.sum())
        torch.testing.assert_close(x, y)


def test_overfits_tiny_batch():
    model = _small_model()
    ids, id_lens, mels, mel_lens = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    torch.manual_seed(0)
    first = None
    for step in range(60):
        loss = model.compute_loss(ids, id_lens, mels, mel_lens)["prior"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first - 0.1


def test_vocab_extension_and_lora_do_not_change_untagged_outputs():
    model = _small_model().eval()
    ids, id_lens, _, _ = _batch()
    before = model.encode(ids, id_lens)[0]
    model.encoder.embed = ExtendedEmbedding(model.encoder.embed, n_new=3)
    names = inject_lora(model.encoder, ["q_proj", "k_proj", "v_proj", "o_proj"], r=2, alpha=4)
    assert len(names) == 8
    assert isinstance(model.encoder.layers[0].attn.q_proj, LoRALinear)
    after = model.encode(ids, id_lens)[0]
    torch.testing.assert_close(before, after)
    new_ids = torch.tensor([[2, 12, 13, 14, 3]])
    assert torch.isfinite(model.encode(new_ids, torch.tensor([5]))[0]).all()


def test_save_and_load_round_trip(tmp_path):
    model = _small_model().eval()
    path = tmp_path / "m.pt"
    model.save(path, extra={"tokenizer": {"symbols": ["a"]}})
    again, extra = TinyMatcha.load(path)
    assert extra["tokenizer"] == {"symbols": ["a"]}
    ids, id_lens, _, _ = _batch()
    torch.testing.assert_close(model.encode(ids, id_lens)[0], again.encode(ids, id_lens)[0])


def test_normalization_round_trip():
    model = _small_model()
    model.mel_mean.fill_(-5.0)
    model.mel_std.fill_(2.0)
    mel = torch.randn(1, 8, 4)
    torch.testing.assert_close(model.denormalize(model.normalize(mel)), mel)
