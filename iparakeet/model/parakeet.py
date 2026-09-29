"""Standalone FP32 Parakeet-CTC with NeMo-compatible state-dict keys."""

import contextlib
import math
from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from iparakeet.model.config import ParakeetConfig
from iparakeet.model.frontend import FilterbankFeatures
from iparakeet.model.layers import ConformerLayer, ConvSubsampling, Tap
from iparakeet.model.relpos import rel_pos_emb, rel_pos_table


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", rel_pos_table(max_len, d_model).unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        max_len = (self.pe.shape[1] + 1) // 2
        if length > max_len:
            self.pe = rel_pos_table(length, self.d_model).unsqueeze(0).to(self.pe.device)
        return rel_pos_emb(self.pe[0], length).unsqueeze(0)


class ConformerEncoder(nn.Module):
    def __init__(self, cfg: ParakeetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pre_encode = ConvSubsampling(cfg)
        self.pos_enc = RelPositionalEncoding(cfg.d_model, cfg.pos_emb_max_len)
        self.layers = nn.ModuleList(ConformerLayer(cfg, i) for i in range(cfg.n_layers))
        self.tap_out = Tap("pre.out")

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.pre_encode(feats.transpose(1, 2), lengths)
        if self.cfg.xscaling:
            x = x * math.sqrt(self.cfg.d_model)
        x = self.tap_out(x)
        L = x.shape[1]
        pos_emb = self.pos_enc(L).to(x.dtype)
        valid = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        if bool(valid.all()):
            pad_mask = att_mask = None
        else:
            pad_mask = ~valid
            att_mask = ~(valid.unsqueeze(1) & valid.unsqueeze(2))
        for layer in self.layers:
            x = layer(x, att_mask, pos_emb, pad_mask)
        return x, lengths


class ConvASRDecoder(nn.Module):
    def __init__(self, d_model: int, num_classes_with_blank: int) -> None:
        super().__init__()
        self.decoder_layers = nn.Sequential(nn.Conv1d(d_model, num_classes_with_blank, kernel_size=1))
        self.tap_logits = Tap("head.logits")

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        conv = self.decoder_layers[0]
        return self.tap_logits(F.linear(enc, conv.weight.squeeze(-1), conv.bias))


class Preprocessor(nn.Module):
    def __init__(self, cfg: ParakeetConfig) -> None:
        super().__init__()
        self.featurizer = FilterbankFeatures(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.feat_in,
            preemph=cfg.preemph,
            log_zero_guard_value=cfg.log_zero_guard_value,
        )

    def forward(self, audio, lengths):
        return self.featurizer(audio, lengths)


class ParakeetCTC(nn.Module):
    def __init__(self, cfg: ParakeetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = Preprocessor(cfg)
        self.encoder = ConformerEncoder(cfg)
        self.decoder = ConvASRDecoder(cfg.d_model, cfg.vocab_size + 1)

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats, feat_lengths = self.preprocessor(audio, audio_lengths)
        return self.forward_features(feats, feat_lengths)

    def forward_features(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats: (B, n_mels, T) log-mel features; returns logits (B, L, vocab+1) and lengths."""
        enc, enc_lengths = self.encoder(feats, lengths)
        return self.decoder(enc), enc_lengths


def iter_taps(model: nn.Module) -> Iterator[Tap]:
    return (m for m in model.modules() if isinstance(m, Tap))


@contextlib.contextmanager
def collect_taps(model: nn.Module, fn: Callable[[str, torch.Tensor], object]):
    """Call fn(name, tensor) at every tap during the forward passes inside the context (observe only)."""
    handles = [tap.register_forward_hook(lambda m, inp, out: (fn(m.name, out), None)[1]) for tap in iter_taps(model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def rewrite_taps(model: nn.Module, fn: Callable[[str, torch.Tensor], torch.Tensor]):
    """Replace every tap output by fn(name, tensor) inside the context (e.g. fake quantization)."""
    handles = [tap.register_forward_hook(lambda m, inp, out: fn(m.name, out)) for tap in iter_taps(model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
