"""Static-shape ONNX graph of Parakeet-CTC for one input-length bucket (M6).

Tensor names equal the tap names of the FP32 model / integer simulator (e.g. "L3.conv.dw"), so the
encodings of the simulator map 1:1 onto the graph. Batch is 1 and dropped: activations are (L, d).

  * relative shift: plain Gather with a constant flat index map (paper Sec. 3.1)
  * P: constant (h, d_k, 2L-1) slice of the one computed for the longest bucket
  * Swish: x * Sigmoid(x) (the NPU evaluates Sigmoid with a LUT, paper Sec. 4.1)
  * BatchNorm: folded into the depthwise Conv

Every produced tensor carries an encoding rule (stored in the model metadata) that says which
simulator scale it uses; see iparakeet.deploy.encodings.
"""

import json
import math

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from iparakeet.model.layers import fold_batchnorm, subsampled_length
from iparakeet.model.parakeet import ParakeetCTC
from iparakeet.model.relpos import rel_pos_emb

OPSET = 21
RULES_KEY = "iparakeet.encoding_rules"
ALLOWED_OPS = {
    "Reshape", "Transpose", "Conv", "Relu", "Gemm", "MatMul", "Add", "Mul",
    "Sigmoid", "Split", "Gather", "Softmax", "LayerNormalization",
}


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.rules: dict[str, list] = {}
        self.params: dict[str, dict] = {}

    def init(self, name: str, array, param_axis: int | None = None) -> str:
        self.inits.append(numpy_helper.from_array(np.asarray(array), name))
        if param_axis is not None:
            self.params[name] = {"axis": param_axis}
        return name

    def shape(self, name: str, dims) -> str:
        return self.init(f"{name}.shape", np.array(dims, dtype=np.int64))

    def node(self, op: str, inputs: list[str], output: str, rule: list, **attrs) -> str:
        self.nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        self.rules[output] = rule
        return output


def _tap(name: str, factor: float = 1.0, bits: int | None = None) -> list:
    return ["tap", name, factor, bits]


_SIGMOID = ["fixed_unsigned", 16]  # LUT sigmoid output on a 16-bit [0, 1] grid


def build_onnx(model: ParakeetCTC, n_frames: int, max_frames: int | None = None) -> onnx.ModelProto:
    cfg = model.cfg
    d, h, dk = cfg.d_model, cfg.n_heads, cfg.d_k
    max_frames = max_frames or n_frames
    L = subsampled_length(n_frames, cfg.subsampling_factor)
    L_max = subsampled_length(max_frames, cfg.subsampling_factor)
    b = _Builder()
    b.rules["pre.in"] = _tap("pre.in")

    with torch.no_grad():
        x = _pre_encoder(b, model, n_frames)
        pos = rel_pos_emb(model.encoder.pos_enc.pe[0], L_max).to(model.encoder.pos_enc.pe.dtype)
        for i, layer in enumerate(model.encoder.layers):
            p_full = layer.self_attn.linear_pos(pos)  # (2 L_max - 1, d)
            p = p_full[L_max - L : L_max + L - 1].view(2 * L - 1, h, dk).permute(1, 2, 0)
            x = _block(b, layer, x, f"L{i}.", L, d, h, dk, p)
        dec = model.decoder.decoder_layers[0]
        w = b.init("head.weight", _np(dec.weight.squeeze(-1)), param_axis=0)
        bias = b.init("head.bias", _np(dec.bias))
        b.node("Gemm", [x, w, bias], "head.logits", _tap("head.logits"), transB=1)

    graph = helper.make_graph(
        b.nodes,
        f"parakeet_ctc_{n_frames}f",
        [helper.make_tensor_value_info("pre.in", TensorProto.FLOAT, [n_frames, cfg.feat_in])],
        [helper.make_tensor_value_info("head.logits", TensorProto.FLOAT, [L, cfg.vocab_size + 1])],
        b.inits,
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="iparakeet")
    proto.ir_version = 10
    helper.set_model_props(proto, {RULES_KEY: json.dumps({"activations": b.rules, "params": b.params})})
    return proto


def _pre_encoder(b: _Builder, model: ParakeetCTC, T: int) -> str:
    cfg = model.cfg
    pre = model.encoder.pre_encode
    taps = dict(zip(pre.tap_idx, (t.name for t in pre.taps)))
    cur = b.node("Reshape", ["pre.in", b.shape("pre.in4", [1, 1, T, cfg.feat_in])], "pre.in4", _tap("pre.in"))
    last = "pre.in"
    for idx, layer in enumerate(pre.conv):
        if not isinstance(layer, nn.Conv2d):
            continue
        relu = idx + 1 < len(pre.conv) and isinstance(pre.conv[idx + 1], nn.ReLU)
        out = taps[idx + 1] if relu else taps[idx]
        w = b.init(f"{out}.weight", _np(layer.weight), param_axis=0)
        bias = b.init(f"{out}.bias", _np(layer.bias))
        k, s, p = layer.kernel_size[0], layer.stride[0], layer.padding[0]
        conv_out = f"{out}.conv" if relu else out
        b.node("Conv", [cur, w, bias], conv_out, _tap(out), kernel_shape=[k, k], strides=[s, s], pads=[p, p, p, p], group=layer.groups)
        if relu:
            b.node("Relu", [conv_out], out, _tap(out))
        cur = last = out
    t = subsampled_length(T, cfg.subsampling_factor)
    freq = subsampled_length(cfg.feat_in, cfg.subsampling_factor)
    channels = cfg.subsampling_conv_channels
    tr = b.node("Transpose", [cur], "pre.flat.t", _tap(last), perm=[0, 2, 1, 3])
    flat = b.node("Reshape", [tr, b.shape("pre.flat", [t, channels * freq])], "pre.flat", _tap(last))
    w = b.init("pre.out.weight", _np(pre.out.weight), param_axis=0)
    bias = b.init("pre.out.bias", _np(pre.out.bias))
    if not cfg.xscaling:
        return b.node("Gemm", [flat, w, bias], "pre.out", _tap("pre.out"), transB=1)
    scale = math.sqrt(cfg.d_model)
    lin = b.node("Gemm", [flat, w, bias], "pre.out.lin", _tap("pre.out", 1.0 / scale), transB=1)
    return b.node("Mul", [lin, b.init("pre.xscale", np.array(scale, dtype=np.float32))], "pre.out", _tap("pre.out"))


def _layernorm(b: _Builder, x: str, ln: nn.LayerNorm, out: str) -> str:
    g = b.init(f"{out}.gamma", _np(ln.weight))
    beta = b.init(f"{out}.beta", _np(ln.bias))
    return b.node("LayerNormalization", [x, g, beta], out, _tap(out), axis=-1, epsilon=ln.eps)


def _gemm(b: _Builder, x: str, weight, bias, out: str, rule=None) -> str:
    w = b.init(f"{out}.weight", _np(weight), param_axis=0)
    bb = b.init(f"{out}.bias", _np(bias))
    return b.node("Gemm", [x, w, bb], out, rule or _tap(out), transB=1)


def _swish(b: _Builder, x: str, out: str) -> str:
    sig = b.node("Sigmoid", [x], f"{out}.sig", _SIGMOID)
    return b.node("Mul", [x, sig], out, _tap(out))


def _ffn(b: _Builder, x: str, ff, tag: str) -> str:
    lin1 = _gemm(b, x, ff.linear1.weight, ff.linear1.bias, f"{tag}.lin1")
    act = _swish(b, lin1, f"{tag}.act")
    return _gemm(b, act, ff.linear2.weight, ff.linear2.bias, f"{tag}.lin2")


def _half(b: _Builder, x: str, src: str) -> str:
    return b.node("Mul", [x, b.init(f"{x}.half.c", np.array(0.5, dtype=np.float32))], f"{x}.half", _tap(src, 0.5))


def _block(b: _Builder, layer, x: str, p: str, L: int, d: int, h: int, dk: int, pos: torch.Tensor) -> str:
    ln = _layernorm(b, x, layer.norm_feed_forward1, p + "ff1.ln")
    res = b.node("Add", [x, _half(b, _ffn(b, ln, layer.feed_forward1, p + "ff1"), p + "ff1.lin2")], p + "res1", _tap(p + "res1"))

    ln = _layernorm(b, res, layer.norm_self_att, p + "att.ln")
    att_out = _attention(b, ln, layer.self_attn, p, L, d, h, dk, pos)
    res = b.node("Add", [res, att_out], p + "res2", _tap(p + "res2"))

    ln = _layernorm(b, res, layer.norm_conv, p + "conv.ln")
    res = b.node("Add", [res, _conv_block(b, ln, layer.conv, p, L, d)], p + "res3", _tap(p + "res3"))

    ln = _layernorm(b, res, layer.norm_feed_forward2, p + "ff2.ln")
    res = b.node("Add", [res, _half(b, _ffn(b, ln, layer.feed_forward2, p + "ff2"), p + "ff2.lin2")], p + "res4", _tap(p + "res4"))
    return _layernorm(b, res, layer.norm_out, p + "out")


def _attention(b: _Builder, x: str, att, p: str, L: int, d: int, h: int, dk: int, pos: torch.Tensor) -> str:
    a = p + "att."
    wq = b.init(a + "q.weight", _np(att.linear_q.weight.T), param_axis=1)
    q = b.node("MatMul", [x, wq], a + "q_mm", _tap(a + "qu"))
    bq = att.linear_q.bias
    qu = b.node("Add", [q, b.init(a + "bias_u", _np(bq + att.pos_bias_u.reshape(-1)))], a + "qu", _tap(a + "qu"))
    qv = b.node("Add", [q, b.init(a + "bias_v", _np(bq + att.pos_bias_v.reshape(-1)))], a + "qv", _tap(a + "qv"))
    k = _gemm(b, x, att.linear_k.weight, att.linear_k.bias, a + "k")
    v = _gemm(b, x, att.linear_v.weight, att.linear_v.bias, a + "v")

    def heads(t: str, perm: list[int], name: str) -> str:
        r = b.node("Reshape", [t, b.shape(f"{name}.r", [L, h, dk])], f"{name}.r", _tap(t))
        return b.node("Transpose", [r], name, _tap(t), perm=perm)

    qu_h, qv_h = heads(qu, [1, 0, 2], a + "qu.h"), heads(qv, [1, 0, 2], a + "qv.h")
    k_t, v_h = heads(k, [1, 2, 0], a + "k.hT"), heads(v, [1, 0, 2], a + "v.h")
    sum_rule = _tap(a + "scores", math.sqrt(dk), 16)
    ac = b.node("MatMul", [qu_h, k_t], a + "ac", sum_rule)
    p_const = b.init(a + "p", _np(pos), param_axis=None)
    b.params[a + "p"] = {"axis": None}
    bd = b.node("MatMul", [qv_h, p_const], a + "bd_full", sum_rule)
    flat = b.node("Reshape", [bd, b.shape(a + "bd_flat", [h, L * (2 * L - 1)])], a + "bd_flat", sum_rule)
    i = np.arange(L).reshape(L, 1)
    j = np.arange(L).reshape(1, L)
    index = (i * (2 * L - 1) + (L - 1 - i + j)).reshape(-1).astype(np.int64)
    gathered = b.node("Gather", [flat, b.init(a + "relshift_index", index)], a + "bd_g", sum_rule, axis=1)
    bd = b.node("Reshape", [gathered, b.shape(a + "bd", [h, L, L])], a + "bd", sum_rule)
    total = b.node("Add", [ac, bd], a + "sum", sum_rule)
    scores = b.node("Mul", [total, b.init(a + "inv_sqrt_dk", np.array(1 / math.sqrt(dk), dtype=np.float32))], a + "scores", _tap(a + "scores"))
    probs = b.node("Softmax", [scores], a + "probs", _tap(a + "probs"), axis=-1)
    ctx = b.node("MatMul", [probs, v_h], a + "ctx.h", _tap(a + "ctx"))
    ctx = b.node("Transpose", [ctx], a + "ctx.t", _tap(a + "ctx"), perm=[1, 0, 2])
    ctx = b.node("Reshape", [ctx, b.shape(a + "ctx", [L, d])], a + "ctx", _tap(a + "ctx"))
    return _gemm(b, ctx, att.linear_out.weight, att.linear_out.bias, a + "out")


def _conv_block(b: _Builder, x: str, conv, p: str, L: int, d: int) -> str:
    c = p + "conv."
    pw1 = _gemm(b, x, conv.pointwise_conv1.weight.squeeze(-1), conv.pointwise_conv1.bias, c + "pw1")
    split = b.init(c + "split", np.array([d, d], dtype=np.int64))
    b.nodes.append(helper.make_node("Split", [pw1, split], [c + "a", c + "b"], name=c + "split", axis=1))
    b.rules[c + "a"] = b.rules[c + "b"] = _tap(c + "pw1")
    gate = b.node("Sigmoid", [c + "b"], c + "gate", ["fixed_gate"])
    glu = b.node("Mul", [c + "a", gate], c + "glu", _tap(c + "glu"))
    t = b.node("Transpose", [glu], c + "glu.t", _tap(c + "glu"), perm=[1, 0])
    r = b.node("Reshape", [t, b.shape(c + "glu.r", [1, d, L])], c + "glu.r", _tap(c + "glu"))
    w, bias = fold_batchnorm(conv.depthwise_conv.weight, conv.depthwise_conv.bias, conv.batch_norm)
    k = w.shape[-1]
    wi = b.init(c + "dw.weight", _np(w), param_axis=0)
    bi = b.init(c + "dw.bias", _np(bias))
    dw = b.node("Conv", [r, wi, bi], c + "dw.c", _tap(c + "dw"), kernel_shape=[k], pads=[(k - 1) // 2] * 2, group=d)
    dw = b.node("Reshape", [dw, b.shape(c + "dw.r", [d, L])], c + "dw.r", _tap(c + "dw"))
    dw = b.node("Transpose", [dw], c + "dw", _tap(c + "dw"), perm=[1, 0])
    act = _swish(b, dw, c + "act")
    return _gemm(b, act, conv.pointwise_conv2.weight.squeeze(-1), conv.pointwise_conv2.bias, c + "pw2")


def encoding_rules(model: onnx.ModelProto) -> dict:
    for prop in model.metadata_props:
        if prop.key == RULES_KEY:
            return json.loads(prop.value)
    raise KeyError("model has no iparakeet encoding rules")
