"""FastConformer building blocks with NeMo-compatible parameter names and named tap points.

Every tensor that the integer-only model quantizes passes through a `Tap` module whose name is
shared with the calibration statistics and the integer simulator (e.g. "L3.conv.dw").
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from iparakeet.model.config import ParakeetConfig
from iparakeet.model.relpos import rel_shift

INF_VAL = 10000.0


class Tap(nn.Module):
    """Identity marking a named quantization point."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def extra_repr(self) -> str:
        return self.name


def subsampled_length(lengths, factor: int = 8):
    """Length after log2(factor) stride-2 convolutions with kernel 3 and padding 1."""
    for _ in range(int(math.log2(factor))):
        if isinstance(lengths, torch.Tensor):
            lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        else:
            lengths = (lengths - 1) // 2 + 1
    return lengths


def fold_batchnorm(weight: torch.Tensor, bias: torch.Tensor | None, bn: nn.BatchNorm1d):
    """Fold an eval-mode BatchNorm into the preceding (depthwise) convolution."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = torch.zeros_like(bn.running_mean) if bias is None else bias
    w = weight * scale.view(-1, *([1] * (weight.dim() - 1)))
    b = (bias - bn.running_mean) * scale + bn.bias
    return w, b


def _mask_time(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero time steps >= length in a (B, C, T, F) tensor."""
    valid = torch.arange(x.shape[2], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x * valid[:, None, :, None].to(x.dtype)


class ConvSubsampling(nn.Module):
    """NeMo `dw_striding` 8x subsampling: conv3x3/s2 + ReLU, then (dw3x3/s2, pw1x1, ReLU) x 2, then Linear."""

    def __init__(self, cfg: ParakeetConfig) -> None:
        super().__init__()
        channels = cfg.subsampling_conv_channels
        n_stages = int(math.log2(cfg.subsampling_factor))
        layers: list[nn.Module] = [nn.Conv2d(1, channels, 3, 2, 1), nn.ReLU()]
        tap_at = {1: "pre.conv0"}
        for stage in range(1, n_stages):
            layers.append(nn.Conv2d(channels, channels, 3, 2, 1, groups=channels))
            tap_at[len(layers) - 1] = f"pre.dw{stage}"
            layers += [nn.Conv2d(channels, channels, 1), nn.ReLU()]
            tap_at[len(layers) - 1] = f"pre.pw{stage}"
        self.conv = nn.Sequential(*layers)
        freq = subsampled_length(cfg.feat_in, cfg.subsampling_factor)
        self.out = nn.Linear(channels * freq, cfg.d_model)
        self.tap_in = Tap("pre.in")
        self.tap_idx = sorted(tap_at)
        self.taps = nn.ModuleList(Tap(tap_at[i]) for i in self.tap_idx)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tap_in(x).unsqueeze(1)  # (B, 1, T, F)
        taps = dict(zip(self.tap_idx, self.taps))
        cur = lengths
        for idx, layer in enumerate(self.conv):
            x = layer(_mask_time(x, cur))
            if isinstance(layer, nn.Conv2d) and layer.stride[0] == 2:
                cur = subsampled_length(cur, 2)
            if idx in taps:
                x = taps[idx](x)
        x = _mask_time(x, cur)
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f)), cur


class RelPositionMultiHeadAttention(nn.Module):
    """Transformer-XL style relative-position MHSA (NeMo `rel_pos`, untied biases)."""

    def __init__(self, n_heads: int, d_model: int, prefix: str) -> None:
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        for name in ("qu", "qv", "k", "v", "p", "scores", "probs", "ctx", "out"):
            setattr(self, f"tap_{name}", Tap(f"{prefix}.att.{name}"))

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, L, d = x.shape
        q = self.linear_q(x).view(b, L, self.h, self.d_k)
        k = self.tap_k(self.linear_k(x).view(b, L, self.h, self.d_k).transpose(1, 2))
        v = self.tap_v(self.linear_v(x).view(b, L, self.h, self.d_k).transpose(1, 2))
        p = self.tap_p(self.linear_pos(pos_emb).view(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2))
        q_u = self.tap_qu((q + self.pos_bias_u).transpose(1, 2))
        q_v = self.tap_qv((q + self.pos_bias_v).transpose(1, 2))
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        matrix_bd = rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
        scores = self.tap_scores((matrix_ac + matrix_bd) / math.sqrt(self.d_k))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -INF_VAL)
        probs = torch.softmax(scores, dim=-1)
        if mask is not None:
            probs = probs.masked_fill(mask.unsqueeze(1), 0.0)
        probs = self.tap_probs(probs)
        ctx = self.tap_ctx(torch.matmul(probs, v).transpose(1, 2).reshape(b, L, d))
        return self.tap_out(self.linear_out(ctx))


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, prefix: str) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.tap_lin1 = Tap(f"{prefix}.lin1")
        self.tap_act = Tap(f"{prefix}.act")
        self.tap_lin2 = Tap(f"{prefix}.lin2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tap_act(F.silu(self.tap_lin1(self.linear1(x))))
        return self.tap_lin2(self.linear2(x))


class ConformerConvolution(nn.Module):
    """PW1 -> GLU -> DW(K) -> BatchNorm -> Swish -> PW2 (paper Eq. 5)."""

    def __init__(self, d_model: int, kernel_size: int, prefix: str) -> None:
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        for name in ("pw1", "gate", "glu", "dw", "act", "pw2"):
            setattr(self, f"tap_{name}", Tap(f"{prefix}.conv.{name}"))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.tap_pw1(F.linear(x, self.pointwise_conv1.weight.squeeze(-1), self.pointwise_conv1.bias))
        a, gate_in = x.chunk(2, dim=-1)
        x = self.tap_glu(a * self.tap_gate(torch.sigmoid(gate_in)))
        x = x.transpose(1, 2)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.batch_norm(self.depthwise_conv(x)).transpose(1, 2)
        x = self.tap_act(F.silu(self.tap_dw(x)))
        return self.tap_pw2(F.linear(x, self.pointwise_conv2.weight.squeeze(-1), self.pointwise_conv2.bias))


class ConformerLayer(nn.Module):
    """Macaron Conformer block (paper Eq. 1-4)."""

    def __init__(self, cfg: ParakeetConfig, index: int) -> None:
        super().__init__()
        p = f"L{index}"
        d = cfg.d_model
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = ConformerFeedForward(d, cfg.d_ff, f"{p}.ff1")
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RelPositionMultiHeadAttention(cfg.n_heads, d, p)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = ConformerConvolution(d, cfg.conv_kernel_size, p)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = ConformerFeedForward(d, cfg.d_ff, f"{p}.ff2")
        self.norm_out = nn.LayerNorm(d)
        names = ("ff1.ln", "res1", "att.ln", "res2", "conv.ln", "res3", "ff2.ln", "res4", "out")
        self.taps = nn.ModuleDict({n.replace(".", "_"): Tap(f"{p}.{n}") for n in names})

    def forward(self, x, att_mask, pos_emb, pad_mask) -> torch.Tensor:
        t = self.taps
        res = t["res1"](x + 0.5 * self.feed_forward1(t["ff1_ln"](self.norm_feed_forward1(x))))
        res = t["res2"](res + self.self_attn(t["att_ln"](self.norm_self_att(res)), pos_emb, att_mask))
        res = t["res3"](res + self.conv(t["conv_ln"](self.norm_conv(res)), pad_mask))
        res = t["res4"](res + 0.5 * self.feed_forward2(t["ff2_ln"](self.norm_feed_forward2(res))))
        return t["out"](self.norm_out(res))
