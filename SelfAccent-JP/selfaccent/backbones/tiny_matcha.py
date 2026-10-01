"""A small raw-text conditional-flow-matching TTS (Matcha-TTS style) for CPU.

It plays the role of the paper's frozen "raw-text end-to-end backbone" in the
toy experiments: characters (kanji, kana, punctuation) go in, there is no
lexicon or G2P, and the reading of every word is learned from data. Structure:

    chars -> embed -> conv prenet -> RoPE Transformer -> mu (mel prior)
                                                     -> duration predictor
    MAS aligns mu to the target mel during training (Glow-TTS prior loss);
    an OT-CFM decoder turns noise into mel conditioned on the upsampled mu.

Linear layers in the encoder attention are named q_proj/k_proj/v_proj/o_proj
so LoRA can target them the same way as in LLM-based TTS backbones.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from selfaccent.backbones.mas import monotonic_alignment


@dataclass
class TinyMatchaConfig:
    n_vocab: int
    n_mels: int = 80
    d_model: int = 192
    n_heads: int = 2
    enc_layers: int = 4
    ffn_dim: int = 768
    prenet_layers: int = 3
    dropout: float = 0.1
    dp_channels: int = 256
    dec_channels: int = 256
    dec_blocks: int = 6
    sigma_min: float = 1e-4


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max())
    return (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]).long()


def upsample_by_durations(x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """[B, C, Tx] -> [B, C, sum(durations)] by repeating each token's column."""
    path = durations_to_path(durations)
    return x @ path.to(x.dtype)


def durations_to_path(durations: torch.Tensor) -> torch.Tensor:
    """Integer durations [B, Tx] -> hard alignment [B, Tx, Ty]."""
    durations = durations.long()
    total = int(durations.sum(1).max())
    ends = torch.cumsum(durations, dim=1)
    starts = ends - durations
    t = torch.arange(total, device=durations.device)[None, None, :]
    return ((t >= starts[..., None]) & (t < ends[..., None])).float()


class LayerNorm1d(nn.Module):
    """LayerNorm over channels of a [B, C, T] tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on [B, H, T, Dh]."""
    t, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    freqs = 10000.0 ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    ang = torch.arange(t, device=x.device, dtype=x.dtype)[:, None] * freqs[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half :]], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float, rope: bool = True) -> None:
        super().__init__()
        self.h = n_heads
        self.rope = rope
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        split = lambda y: y.view(b, t, self.h, d // self.h).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        if self.rope:
            q, k = _rope(q), _rope(k)
        attn_mask = mask.bool()[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, d))


class TransformerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_dim: int, dropout: float, rope: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = SelfAttention(d, n_heads, dropout, rope)
        self.norm2 = nn.LayerNorm(d)
        self.ffn_in = nn.Linear(d, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ffn_out(self.drop(F.gelu(self.ffn_in(self.norm2(x))))))
        return x * mask[..., None]


class TextEncoder(nn.Module):
    def __init__(self, cfg: TinyMatchaConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.n_vocab, d)
        nn.init.normal_(self.embed.weight, 0.0, d**-0.5)
        self.prenet = nn.ModuleList(
            nn.Sequential(nn.Conv1d(d, d, 5, padding=2), LayerNorm1d(d), nn.ReLU(), nn.Dropout(cfg.dropout))
            for _ in range(cfg.prenet_layers)
        )
        self.layers = nn.ModuleList(
            TransformerLayer(d, cfg.n_heads, cfg.ffn_dim, cfg.dropout) for _ in range(cfg.enc_layers)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids) * math.sqrt(self.embed.embedding_dim)
        h = x.transpose(1, 2) * mask[:, None]
        for block in self.prenet:
            h = h + block(h) * mask[:, None]
        x = h.transpose(1, 2)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) * mask[..., None]  # [B, Tx, d]


class DurationPredictor(nn.Module):
    def __init__(self, d_in: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, channels, 3, padding=1), nn.ReLU(), LayerNorm1d(channels), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(), LayerNorm1d(channels), nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x * mask) * mask) * mask  # log durations [B, 1, Tx]


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = 1000.0 * t[:, None] * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c: int, dilation: int, t_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, c)
        self.film = nn.Linear(t_dim, 2 * c)
        self.conv2 = nn.Conv1d(c, c, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = F.mish(self.norm1(self.conv1(x * mask)))
        scale, shift = self.film(temb)[..., None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        return (x + self.conv2(h)) * mask


class FlowDecoder(nn.Module):
    """Velocity estimator v(x_t, mu, t) for OT-CFM."""

    def __init__(self, n_mels: int, c: int, n_blocks: int, n_heads: int = 2) -> None:
        super().__init__()
        t_dim = c
        self.conv_in = nn.Conv1d(2 * n_mels, c, 1)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, 2 * t_dim), nn.SiLU(), nn.Linear(2 * t_dim, t_dim))
        dilations = [1, 2, 4] * math.ceil(n_blocks / 3)
        half = n_blocks // 2
        self.blocks1 = nn.ModuleList(ResBlock(c, dilations[i], t_dim) for i in range(half))
        self.mid = TransformerLayer(c, n_heads, 2 * c, 0.0, rope=True)
        self.blocks2 = nn.ModuleList(ResBlock(c, dilations[i], t_dim) for i in range(half, n_blocks))
        self.conv_out = nn.Conv1d(c, n_mels, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        self.t_dim = t_dim

    def forward(self, x: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[:, None].to(x.dtype)
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h = self.conv_in(torch.cat([x, mu], dim=1)) * m
        for blk in self.blocks1:
            h = blk(h, temb, m)
        h = self.mid(h.transpose(1, 2), mask).transpose(1, 2)
        for blk in self.blocks2:
            h = blk(h, temb, m)
        return self.conv_out(h) * m


@dataclass
class SynthesisOutput:
    mels: list[torch.Tensor]  # denormalized log-mel, each [n_mels, T]
    durations: list[torch.Tensor]  # frames per input token


class TinyMatcha(nn.Module):
    def __init__(self, cfg: TinyMatchaConfig, mel_mean: torch.Tensor, mel_std: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg)
        self.proj_mu = nn.Linear(cfg.d_model, cfg.n_mels)
        self.duration_predictor = DurationPredictor(cfg.d_model, cfg.dp_channels, cfg.dropout)
        self.decoder = FlowDecoder(cfg.n_mels, cfg.dec_channels, cfg.dec_blocks)
        self.register_buffer("mel_mean", mel_mean.clone().float().view(-1))
        self.register_buffer("mel_std", mel_std.clone().float().view(-1))
        # Matcha detaches the duration predictor input; adaptation turns this off
        # so that the duration loss can reach the adapter (the predictor stays frozen).
        self.detach_duration_input = True

    # -- feature normalization -------------------------------------------------
    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mel_mean[:, None]) / self.mel_std[:, None]

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        return mel * self.mel_std[:, None] + self.mel_mean[:, None]

    # -- encoder ------------------------------------------------------------------
    def encode(self, ids: torch.Tensor, id_lengths: torch.Tensor):
        x_mask = sequence_mask(id_lengths, ids.shape[1]).to(ids.device)
        h = self.encoder(ids, x_mask)
        mu = self.proj_mu(h).transpose(1, 2) * x_mask[:, None]  # [B, M, Tx]
        dp_in = h.detach() if self.detach_duration_input else h
        logw = self.duration_predictor(dp_in.transpose(1, 2), x_mask[:, None].float())
        return mu, logw, x_mask

    @staticmethod
    def _log_likelihood(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """-0.5 ||y_j - mu_i||^2 for every token i and frame j -> [B, Tx, Ty]."""
        mu_sq = (mu**2).sum(1)[:, :, None]
        y_sq = (y**2).sum(1)[:, None, :]
        return -0.5 * (mu_sq - 2 * mu.transpose(1, 2) @ y + y_sq)

    @torch.no_grad()
    def align(self, ids, id_lengths, mels, mel_lengths) -> torch.Tensor:
        """MAS durations [B, Tx] of ``mels`` (log-mel, not normalized)."""
        mu, _, _ = self.encode(ids, id_lengths)
        y = self.normalize(mels)
        attn = monotonic_alignment(self._log_likelihood(mu, y), id_lengths, mel_lengths)
        return attn.sum(-1).long()

    # -- training -----------------------------------------------------------------
    def compute_loss(self, ids, id_lengths, mels, mel_lengths) -> dict[str, torch.Tensor]:
        mu, logw, x_mask = self.encode(ids, id_lengths)
        y = self.normalize(mels)
        y_mask = sequence_mask(mel_lengths, y.shape[-1]).to(y.device)
        y = y * y_mask[:, None]
        attn = monotonic_alignment(self._log_likelihood(mu.detach(), y), id_lengths, mel_lengths)

        logw_target = torch.log(1e-8 + attn.sum(-1)).unsqueeze(1) * x_mask[:, None]
        dur_loss = ((logw - logw_target) ** 2).sum() / id_lengths.sum()

        mu_y = mu @ attn  # [B, M, Ty]
        n_frames = y_mask.sum() * self.cfg.n_mels
        prior = (0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask[:, None]).sum() / n_frames

        b = y.shape[0]
        t = torch.rand(b, device=y.device)
        z = torch.randn_like(y)
        sig = self.cfg.sigma_min
        tt = t[:, None, None]
        x_t = (1 - (1 - sig) * tt) * z + tt * y
        target = y - (1 - sig) * z
        pred = self.decoder(x_t, mu_y, t, y_mask)
        cfm = (((pred - target) ** 2) * y_mask[:, None]).sum() / n_frames
        return {"loss": dur_loss + prior + cfm, "prior": prior, "duration": dur_loss, "cfm": cfm}

    # -- inference ----------------------------------------------------------------
    @torch.no_grad()
    def synthesize(
        self,
        ids: torch.Tensor,
        id_lengths: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.667,
        length_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SynthesisOutput:
        mu, logw, x_mask = self.encode(ids, id_lengths)
        w = torch.exp(logw.squeeze(1)) * length_scale
        durations = torch.clamp(torch.ceil(w), min=1).long() * x_mask
        y_lengths = durations.sum(1)
        mu_y = upsample_by_durations(mu, durations)
        y_mask = sequence_mask(y_lengths, mu_y.shape[-1])
        z = torch.randn(mu_y.shape, generator=generator) * temperature
        x = z * y_mask[:, None]
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt)
            x = x + dt * self.decoder(x, mu_y, t, y_mask)
        mel = self.denormalize(x)
        return SynthesisOutput(
            mels=[mel[b, :, : int(y_lengths[b])] for b in range(mel.shape[0])],
            durations=[durations[b, : int(id_lengths[b])] for b in range(mel.shape[0])],
        )

    # -- persistence --------------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["TinyMatcha", dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TinyMatchaConfig(**ckpt["config"])
        model = cls(cfg, torch.zeros(cfg.n_mels), torch.ones(cfg.n_mels))
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt["extra"]


class TinyMatchaBackbone:
    """Implements :class:`selfaccent.backbones.base.Backbone` for TinyMatcha."""

    def __init__(
        self,
        model: TinyMatcha,
        tokenizer,
        n_steps: int = 10,
        temperature: float = 0.667,
        length_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.n_steps = n_steps
        self.temperature = temperature
        self.length_scale = length_scale

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> "TinyMatchaBackbone":
        from selfaccent.tokenizer import CharTokenizer

        model, extra = TinyMatcha.load(path)
        return cls(model.eval(), CharTokenizer.from_dict(extra["tokenizer"]), **kwargs)

    def encode_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.tokenizer.encode(t) for t in texts]
        lengths = torch.tensor([len(s) for s in seqs])
        ids = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return ids, lengths

    @torch.no_grad()
    def synthesize(self, texts: list[str], seeds: list[int]) -> list[torch.Tensor]:
        was_training = self.model.training
        self.model.eval()
        out = []
        for text, seed in zip(texts, seeds):
            ids, lengths = self.encode_texts([text])
            res = self.model.synthesize(
                ids, lengths, n_steps=self.n_steps, temperature=self.temperature,
                length_scale=self.length_scale, generator=torch.Generator().manual_seed(seed),
            )
            out.append(res.mels[0])
        self.model.train(was_training)
        return out

    def training_loss(self, texts: list[str], targets: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        ids, id_lengths = self.encode_texts(texts)
        mel_lengths = torch.tensor([m.shape[-1] for m in targets])
        mels = torch.zeros(len(targets), targets[0].shape[0], int(mel_lengths.max()))
        for i, m in enumerate(targets):
            mels[i, :, : m.shape[-1]] = m
        return self.model.compute_loss(ids, id_lengths, mels, mel_lengths)

    def add_tag_tokens(self, symbols: list[str]) -> int:
        from selfaccent.lora import ExtendedEmbedding

        embed = self.model.encoder.embed
        if isinstance(embed, ExtendedEmbedding):
            raise RuntimeError("tag tokens were already added")
        n_before = len(self.tokenizer)
        if n_before != embed.num_embeddings:
            raise ValueError("tokenizer and embedding table sizes differ")
        self.tokenizer.extend(symbols)
        n_new = len(self.tokenizer) - n_before
        if n_new:
            self.model.encoder.embed = ExtendedEmbedding(embed, n_new)
        return n_new

    def prepare_for_adaptation(self) -> None:
        self.model.detach_duration_input = False

    def adapter_root(self) -> nn.Module:
        return self.model.encoder

    def default_lora_targets(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
