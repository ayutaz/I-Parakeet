"""Integer relative-positional self-attention (paper Sec. 3.1, Eq. 8-11).

  q_u, q_v : Q + u, Q + v on one INT8 grid S_Q (u, v folded into the INT32 bias of W_Q)
  q_c      = q_u k^T                       (INT32, scale S_c = S_Q S_K)
  q_p      = q_v q_P^T                     (INT32, scale S_p = S_Q S_P), q_P an INT8 constant
  q_s      = round(2^-n (m_c q_c + m_p Phi(q_p)))   Eq. (11): scale alignment, addition and
                                                    1/sqrt(d_k) fused into one requantization
  Phi      = static gather (relative shift), applied to the integer tensor directly
  softmax  = I-BERT integer softmax, A V and W_out follow Eq. (7)
"""

import math

import torch

from iparakeet.intops.linear import IntLinear, quantize_weight_per_channel
from iparakeet.intops.softmax import IntSoftmax
from iparakeet.model.relpos import rel_pos_emb, rel_shift
from iparakeet.quant.fixed_point import check_int32, multiplier, qmax, quantize, requantize, scale_from_alpha
from iparakeet.quant.intmm import int_matmul


def fused_scores(qc: torch.Tensor, qp_shifted: torch.Tensor, m_c, m_p, n: int, bits: int) -> torch.Tensor:
    check_int32(qc, "content scores")
    check_int32(qp_shifted, "position scores")
    acc = qc * m_c + qp_shifted * m_p + (1 << (n - 1))
    return (acc >> n).clamp(-qmax(bits), qmax(bits))


class IntRelPosMHSA:
    def __init__(
        self,
        att,
        in_scale: float,
        scales: dict[str, float],
        pos_table: torch.Tensor,
        max_len: int,
        n: int = 16,
        act_bits: int = 8,
        w_bits: int = 8,
        score_bits: int = 8,
        softmax_bits: int = 8,
        range_reduction: str = "log2",
    ) -> None:
        self.h, self.d_k = att.h, att.d_k
        self.max_len, self.n, self.act_bits, self.score_bits = max_len, n, act_bits, score_bits
        s_q, s_k, s_v, s_s = scales["q"], scales["k"], scales["v"], scales["scores"]

        w_q, s_wq = quantize_weight_per_channel(att.linear_q.weight, w_bits)
        self.wq_t = w_q.T.contiguous()
        denom = in_scale * s_wq
        b_q = att.linear_q.bias.detach().double()
        self.bias_u = torch.round((b_q + att.pos_bias_u.detach().double().reshape(-1)) / denom).to(torch.int64)
        self.bias_v = torch.round((b_q + att.pos_bias_v.detach().double().reshape(-1)) / denom).to(torch.int64)
        self.q_ratio = denom / s_q
        self.m_q = multiplier(self.q_ratio, n)
        self.k = IntLinear(att.linear_k.weight, att.linear_k.bias, in_scale, s_k, act_bits, n, w_bits)
        self.v = IntLinear(att.linear_v.weight, att.linear_v.bias, in_scale, s_v, act_bits, n, w_bits)

        with torch.no_grad():
            p = att.linear_pos(rel_pos_emb(pos_table, max_len).to(att.linear_pos.weight.dtype)).double()
        s_p = scales.get("p") or scale_from_alpha(float(p.abs().max()), 8)
        self.s_p = s_p
        self.q_P = quantize(p, s_p, 8).view(2 * max_len - 1, self.h, self.d_k).permute(1, 0, 2).contiguous()

        root = math.sqrt(self.d_k)
        self.m_c = multiplier(s_q * s_k / (s_s * root), n)
        self.m_p = multiplier(s_q * s_p / (s_s * root), n)
        self.softmax = IntSoftmax(s_s, softmax_bits, range_reduction)
        self.ctx_ratio = s_v / qmax(softmax_bits) / scales["ctx"]
        self.m_ctx = multiplier(self.ctx_ratio, n)
        self.out = IntLinear(att.linear_out.weight, att.linear_out.bias, scales["ctx"], scales["out"], act_bits, n, w_bits)

    def position_rows(self, L: int) -> torch.Tensor:
        """INT8 position constant for length L: a slice of the one stored for max_len."""
        if L > self.max_len:
            raise ValueError(f"length {L} exceeds the compiled maximum {self.max_len}")
        return self.q_P[:, self.max_len - L : self.max_len + L - 1]

    def _heads(self, t: torch.Tensor, b: int, L: int) -> torch.Tensor:
        return t.view(b, L, self.h, self.d_k).transpose(1, 2)

    def __call__(self, q_x: torch.Tensor, trace=None, prefix: str = "") -> torch.Tensor:
        b, L, d = q_x.shape
        acc = int_matmul(q_x, self.wq_t)
        q_u = requantize(acc + self.bias_u, self.m_q, self.n, self.act_bits)
        q_v = requantize(acc + self.bias_v, self.m_q, self.n, self.act_bits)
        k, v = self.k(q_x), self.v(q_x)
        q_u, q_v, k, v = (self._heads(t, b, L) for t in (q_u, q_v, k, v))
        qc = int_matmul(q_u, k.transpose(-2, -1))
        qp = int_matmul(q_v, self.position_rows(L).transpose(-2, -1))
        q_s = fused_scores(qc, rel_shift(qp), self.m_c, self.m_p, self.n, self.score_bits)
        probs = self.softmax(q_s)
        ctx = requantize(int_matmul(probs, v), self.m_ctx, self.n, self.act_bits)
        ctx = ctx.transpose(1, 2).reshape(b, L, d)
        out = self.out(ctx)
        if trace is not None:
            for name, t in (("qu", q_u), ("qv", q_v), ("k", k), ("v", v), ("scores", q_s), ("probs", probs), ("ctx", ctx), ("out", out)):
                trace(f"{prefix}att.{name}", t)
        return out
