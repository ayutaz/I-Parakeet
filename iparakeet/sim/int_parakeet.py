"""Integer-only Parakeet-CTC simulator (paper Sec. 2.2, 3.1-3.3, 4.2).

Built from the FP32 model, calibration statistics and a Recipe. `forward_int` maps INT input
features to INT32 CTC accumulators using only integer tensor operations (guarded by NoFloatMode
in the tests); floats are used only while building (scales, multipliers, LUTs, constants).
"""

import math

import torch
from torch import nn

from iparakeet.analysis.range import CalibrationStats
from iparakeet.intops.layernorm import IntLayerNorm
from iparakeet.intops.linear import IntAdd, IntConv1dDepthwise, IntConv2d, IntGLU, IntLinear
from iparakeet.intops.relpos_mhsa import IntRelPosMHSA
from iparakeet.intops.sigmoid import IntSigmoidLUT, IntSigmoidPoly
from iparakeet.intops.swish import SWISH_COEFFS, make_swish
from iparakeet.model.decoding import ctc_greedy_ids
from iparakeet.model.layers import fold_batchnorm, subsampled_length
from iparakeet.model.parakeet import ParakeetCTC
from iparakeet.quant.fixed_point import dequantize, qmax, quantize, scale_from_alpha
from iparakeet.sim.recipe import Recipe


class _Block:
    pass


class IntParakeet:
    def __init__(self, model: ParakeetCTC, stats: CalibrationStats, recipe: Recipe, max_frames: int) -> None:
        self.cfg, self.recipe, self.stats = model.cfg, recipe, stats
        self.max_frames = max_frames
        self.max_len = subsampled_length(max_frames, model.cfg.subsampling_factor)
        self.scales: dict[str, object] = {}
        self._sites: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        with torch.no_grad():
            self._build_pre_encoder(model)
            self.blocks = [self._build_block(layer, i) for i, layer in enumerate(model.encoder.layers)]
            last = f"L{len(self.blocks) - 1}.out"
            dec = model.decoder.decoder_layers[0]
            self.head = self._linear(dec.weight.squeeze(-1), dec.bias, last, "head.logits")

    # ---- scales ----
    def bits(self, name: str) -> int:
        if name.endswith("conv.dw"):
            return self.recipe.bn_bits
        if name.endswith("att.scores"):
            return self.recipe.score_bits
        if name.endswith("att.probs"):
            return self.recipe.softmax_bits
        if name == "head.logits":
            return self.recipe.head_bits
        return self.recipe.act_bits

    def _alpha(self, name: str) -> float:
        method = self.recipe.pre_calib if name.startswith("pre.") else self.recipe.calib
        return self.stats.alpha(name, method)

    def S(self, name: str):
        if name not in self.scales:
            if name.endswith("conv.dw") and self.recipe.bn_per_channel:
                alpha = torch.tensor(self.stats.tensors[name].channel_absmax, dtype=torch.float64)
                self.scales[name] = scale_from_alpha(alpha, self.bits(name))
            else:
                self.scales[name] = scale_from_alpha(self._alpha(name), self.bits(name))
        return self.scales[name]

    def dequantize(self, name: str, q: torch.Tensor) -> torch.Tensor:
        return dequantize(q, self.S(name))

    # ---- builders ----
    def _record(self, target, m, n) -> None:
        self._sites.append((torch.as_tensor(target, dtype=torch.float64).flatten(), m.flatten(), n))

    def _linear(self, weight, bias, in_name, out_name, extra=1.0, relu=False) -> IntLinear:
        r = self.recipe
        layer = IntLinear(weight, bias, self.S(in_name), self.S(out_name), self.bits(out_name), r.requant_shift, r.weight_bits, extra, relu)
        self._record(layer.ratio * extra, layer.m, layer.n)
        return layer

    def _add(self, in_names, factors, out_name) -> IntAdd:
        add = IntAdd([self.S(n) for n in in_names], factors, self.S(out_name), self.bits(out_name), self.recipe.requant_shift)
        for ratio, m in zip(add.ratios, add.ms):
            self._record(ratio, m, add.n)
        return add

    def _layernorm(self, ln: nn.LayerNorm, in_name, out_name) -> IntLayerNorm:
        return IntLayerNorm(ln.weight, ln.bias, self.S(in_name), self.S(out_name), self.bits(out_name))

    def _build_pre_encoder(self, model: ParakeetCTC) -> None:
        r = self.recipe
        pre = model.encoder.pre_encode
        taps = dict(zip(pre.tap_idx, (t.name for t in pre.taps)))
        in_name = "pre.in"
        self.pre_layers: list[tuple[str, IntConv2d]] = []
        for idx, layer in enumerate(pre.conv):
            if isinstance(layer, nn.Conv2d):
                relu = idx + 1 < len(pre.conv) and isinstance(pre.conv[idx + 1], nn.ReLU)
                out_name = taps[idx + 1] if relu else taps[idx]
                conv = IntConv2d(layer, self.S(in_name), self.S(out_name), self.bits(out_name), relu, r.requant_shift, r.weight_bits)
                self._record(conv.ratio, conv.m, conv.n)
                self.pre_layers.append((out_name, conv))
                in_name = out_name
        extra = math.sqrt(self.cfg.d_model) if self.cfg.xscaling else 1.0
        self.pre_out = self._linear(pre.out.weight, pre.out.bias, in_name, "pre.out", extra=extra)
        self.pos_table = model.encoder.pos_enc.pe[0]

    def _build_block(self, layer, i: int) -> _Block:
        r, p = self.recipe, f"L{i}."
        x = "pre.out" if i == 0 else f"L{i - 1}.out"
        b = _Block()
        b.ln_ff1 = self._layernorm(layer.norm_feed_forward1, x, p + "ff1.ln")
        b.ff1 = self._ffn(layer.feed_forward1, p + "ff1")
        b.res1 = self._add([x, p + "ff1.lin2"], [1.0, 0.5], p + "res1")

        b.ln_att = self._layernorm(layer.norm_self_att, p + "res1", p + "att.ln")
        att = layer.self_attn
        alpha_q = max(self._alpha(p + "att.qu"), self._alpha(p + "att.qv"))
        s_q = scale_from_alpha(alpha_q, r.act_bits)
        self.scales[p + "att.qu"] = self.scales[p + "att.qv"] = s_q
        self.scales[p + "att.probs"] = 1.0 / qmax(r.softmax_bits)
        scales = {"q": s_q}
        for key in ("k", "v", "scores", "ctx", "out"):
            scales[key] = self.S(p + f"att.{key}")
        b.att = IntRelPosMHSA(
            att, self.S(p + "att.ln"), scales, self.pos_table, self.max_len, r.requant_shift, r.act_bits,
            r.weight_bits, r.score_bits, r.softmax_bits, r.softmax_range_reduction,
        )
        self._record(b.att.q_ratio, b.att.m_q, b.att.n)
        self._record(b.att.ctx_ratio, b.att.m_ctx, b.att.n)
        for lin in (b.att.k, b.att.v, b.att.out):
            self._record(lin.ratio, lin.m, lin.n)
        root = math.sqrt(att.d_k)
        self._record(s_q * scales["k"] / (scales["scores"] * root), b.att.m_c, b.att.n)
        self._record(s_q * b.att.s_p / (scales["scores"] * root), b.att.m_p, b.att.n)
        b.res2 = self._add([p + "res1", p + "att.out"], [1.0, 1.0], p + "res2")

        conv = layer.conv
        b.ln_conv = self._layernorm(layer.norm_conv, p + "res2", p + "conv.ln")
        b.pw1 = self._linear(conv.pointwise_conv1.weight.squeeze(-1), conv.pointwise_conv1.bias, p + "conv.ln", p + "conv.pw1")
        s_pw1 = self.S(p + "conv.pw1")
        if r.glu_sigmoid == "lut":
            gate = IntSigmoidLUT(s_pw1, self.bits(p + "conv.pw1"), r.act_bits)
        else:
            gate = IntSigmoidPoly(*SWISH_COEFFS[r.glu_sigmoid.split(":", 1)[1]], in_scale=s_pw1, out_bits=r.act_bits)
        b.glu = IntGLU(s_pw1, self.S(p + "conv.glu"), self.bits(p + "conv.glu"), gate, r.act_bits)
        w, bias = fold_batchnorm(conv.depthwise_conv.weight, conv.depthwise_conv.bias, conv.batch_norm)
        b.dw = IntConv1dDepthwise(w, bias, self.S(p + "conv.glu"), self.S(p + "conv.dw"), self.bits(p + "conv.dw"), r.requant_shift, r.weight_bits)
        self._record(b.dw.ratio, b.dw.m, b.dw.n)
        b.conv_act = make_swish(r.swish, self.S(p + "conv.dw"), self.bits(p + "conv.dw"), self.S(p + "conv.act"), self.bits(p + "conv.act"))
        b.pw2 = self._linear(conv.pointwise_conv2.weight.squeeze(-1), conv.pointwise_conv2.bias, p + "conv.act", p + "conv.pw2")
        b.res3 = self._add([p + "res2", p + "conv.pw2"], [1.0, 1.0], p + "res3")

        b.ln_ff2 = self._layernorm(layer.norm_feed_forward2, p + "res3", p + "ff2.ln")
        b.ff2 = self._ffn(layer.feed_forward2, p + "ff2")
        b.res4 = self._add([p + "res3", p + "ff2.lin2"], [1.0, 0.5], p + "res4")
        b.ln_out = self._layernorm(layer.norm_out, p + "res4", p + "out")
        return b

    def _ffn(self, ff, p: str):
        lin1 = self._linear(ff.linear1.weight, ff.linear1.bias, p + ".ln", p + ".lin1")
        act = make_swish(self.recipe.swish, self.S(p + ".lin1"), self.bits(p + ".lin1"), self.S(p + ".act"), self.bits(p + ".act"))
        lin2 = self._linear(ff.linear2.weight, ff.linear2.bias, p + ".act", p + ".lin2")
        return lin1, act, lin2

    # ---- inference ----
    def quantize_input(self, feats: torch.Tensor) -> torch.Tensor:
        """Model-boundary quantization of (B, n_mels, T) features to the INT grid of `pre.in`."""
        return quantize(feats.transpose(1, 2), self.S("pre.in"), self.bits("pre.in"))

    def forward_int(self, q_in: torch.Tensor, trace=None) -> torch.Tensor:
        """(B, T, n_mels) INT features -> (B, L, vocab+1) CTC logits on one common INT grid."""
        if q_in.shape[1] > self.max_frames:
            raise ValueError(f"{q_in.shape[1]} frames exceed the compiled maximum {self.max_frames}")
        t = trace or (lambda name, q: None)
        t("pre.in", q_in)
        x = q_in.unsqueeze(1)
        for name, conv in self.pre_layers:
            x = conv(x)
            t(name, x)
        b, c, frames, f = x.shape
        x = self.pre_out(x.transpose(1, 2).reshape(b, frames, c * f))
        t("pre.out", x)
        for i, blk in enumerate(self.blocks):
            x = self._run_block(blk, x, trace, f"L{i}.")
        logits = self.head(x)
        t("head.logits", logits)
        return logits

    def _run_block(self, b: _Block, x: torch.Tensor, trace, p: str) -> torch.Tensor:
        t = trace or (lambda name, q: None)

        def step(name, value):
            t(p + name, value)
            return value

        def ffn(kernels, ln, tag):
            lin1, act, lin2 = kernels
            h = step(f"{tag}.ln", ln(x_in[0]))
            h = step(f"{tag}.lin1", lin1(h))
            h = step(f"{tag}.act", act(h))
            return step(f"{tag}.lin2", lin2(h))

        x_in = [x]
        res = step("res1", b.res1(x, ffn(b.ff1, b.ln_ff1, "ff1")))
        h = step("att.ln", b.ln_att(res))
        res = step("res2", b.res2(res, b.att(h, trace, p)))
        h = step("conv.ln", b.ln_conv(res))
        h = step("conv.pw1", b.pw1(h))
        h = step("conv.glu", b.glu(h))
        h = step("conv.dw", b.dw(h.transpose(1, 2)).transpose(1, 2))
        h = step("conv.act", b.conv_act(h))
        h = step("conv.pw2", b.pw2(h))
        res = step("res3", b.res3(res, h))
        x_in = [res]
        res = step("res4", b.res4(res, ffn(b.ff2, b.ln_ff2, "ff2")))
        return step("out", b.ln_out(res))

    # ---- convenience ----
    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Dequantized CTC logits for (B, n_mels, T) features."""
        return self.dequantize("head.logits", self.forward_int(self.quantize_input(feats)))

    def transcribe_ids(self, feats: torch.Tensor) -> list[int]:
        acc = self.forward_int(self.quantize_input(feats))
        return ctc_greedy_ids(acc[0], self.cfg.blank_id)

    def multiplier_report(self) -> dict:
        """Relative error |m 2^-n - ratio| / ratio of every Eq. (7) fixed-point multiplier."""
        errors, zeros = [], 0
        for target, m, n in self._sites:
            target = target.expand_as(m.double()) if target.numel() == 1 else target
            nz = target.abs() > 0
            zeros += int((m == 0).sum())
            errors.append(((m.double() / 2.0**n - target).abs()[nz] / target.abs()[nz]))
        err = torch.cat(errors)
        return {"n_multipliers": int(err.numel()), "zero_multipliers": zeros, "max_rel_error": float(err.max()), "median_rel_error": float(err.median())}
