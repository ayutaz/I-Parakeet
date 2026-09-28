"""Quantization recipes: the paper's Table 2 / Table 3 configurations plus diagnostic variants."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Recipe:
    name: str
    act_bits: int = 8
    weight_bits: int = 8
    calib: str = "minmax"  # clipping range for every tensor outside the pre-encoder
    pre_calib: str = "minmax"  # clipping range for pre-encoder tensors ("pre.*")
    bn_bits: int = 8  # grid of the BatchNorm output (folded depthwise conv output)
    bn_per_channel: bool = False  # per-channel BN-output grid (not deployable on the NPU)
    swish: str = "poly:linf_swish"  # poly:<coeffs> | hardswish | lut
    glu_sigmoid: str = "lut"  # lut | poly:<coeffs>  (not specified in the paper)
    score_bits: int = 8  # grid of the fused attention scores q_s (not specified; default b = 8)
    softmax_bits: int = 8
    softmax_range_reduction: str = "log2"  # log2 | ibert
    requant_shift: int = 16  # n in Eq. (7)


_IPARAKEET = Recipe("iparakeet", pre_calib="p99.9", bn_bits=16)

RECIPES: dict[str, Recipe] = {
    # Table 2
    "ibert_recipe": Recipe("ibert_recipe", swish="poly:l2_tanh"),
    "naive_int8": Recipe("naive_int8"),
    "iparakeet": _IPARAKEET,
    # Table 3: Swish approximation
    "swish_l2_swish": replace(_IPARAKEET, name="swish_l2_swish", swish="poly:l2_swish"),
    "swish_linf_tanh": replace(_IPARAKEET, name="swish_linf_tanh", swish="poly:linf_tanh"),
    "swish_l2_tanh": replace(_IPARAKEET, name="swish_l2_tanh", swish="poly:l2_tanh"),
    "swish_hardswish": replace(_IPARAKEET, name="swish_hardswish", swish="hardswish"),
    # Table 3: BatchNorm output precision
    "bn_per_channel_int16": replace(_IPARAKEET, name="bn_per_channel_int16", bn_per_channel=True),
    "bn_int8": replace(_IPARAKEET, name="bn_int8", bn_bits=8),
    # Table 3: scale calibration
    "calib_minmax_only": replace(_IPARAKEET, name="calib_minmax_only", pre_calib="minmax"),
    "calib_p999_only": replace(_IPARAKEET, name="calib_p999_only", calib="p99.9", pre_calib="p99.9"),
    # diagnostics / sensitivity to unstated details
    "iparakeet_lut": replace(_IPARAKEET, name="iparakeet_lut", swish="lut"),  # NPU-like sigmoid LUT
    "iparakeet_ibert_softmax": replace(_IPARAKEET, name="iparakeet_ibert_softmax", softmax_range_reduction="ibert"),
    "iparakeet_int16_scores": replace(_IPARAKEET, name="iparakeet_int16_scores", score_bits=16),
    "iparakeet_glu_poly": replace(_IPARAKEET, name="iparakeet_glu_poly", glu_sigmoid="poly:linf_tanh"),
    "iparakeet_n24": replace(_IPARAKEET, name="iparakeet_n24", requant_shift=24),
    "lossless_int16": Recipe(
        "lossless_int16", act_bits=16, weight_bits=16, bn_bits=16, swish="lut", score_bits=16, softmax_bits=16, requant_shift=24
    ),
}

# Paper Table 2 (simulated WER %): LibriSpeech test-clean, test-other, Common Voice test.
PAPER_TABLE2: dict[str, tuple[float, float, float]] = {
    "ibert_recipe": (3.41, 7.41, 19.05),
    "naive_int8": (3.01, 6.38, 16.54),
    "iparakeet": (2.61, 5.32, 14.70),
}
PAPER_FP32 = (1.87, 3.76, 10.55)


@dataclass(frozen=True)
class Table3Row:
    group: str
    label: str
    recipe: str
    max_err: float | None
    wer_test_other: float


# Paper Table 3 (LibriSpeech test-other WER %).
PAPER_TABLE3: list[Table3Row] = [
    Table3Row("swish", "L_inf fit to Swish (ours)", "iparakeet", 0.039, 5.32),
    Table3Row("swish", "L2 fit to Swish", "swish_l2_swish", 0.045, 5.49),
    Table3Row("swish", "L_inf fit to tanh", "swish_linf_tanh", 0.068, 5.81),
    Table3Row("swish", "L2 fit to tanh", "swish_l2_tanh", 0.073, 5.95),
    Table3Row("swish", "Hard-Swish", "swish_hardswish", 0.142, 6.33),
    Table3Row("bn", "per-tensor INT16 (ours)", "iparakeet", None, 5.32),
    Table3Row("bn", "per-channel INT16 (not deployable)", "bn_per_channel_int16", None, 5.29),
    Table3Row("bn", "per-tensor INT8 (min-max)", "bn_int8", None, 5.91),
    Table3Row("calib", "hybrid min-max + p99.9 (ours)", "iparakeet", None, 5.32),
    Table3Row("calib", "min-max only", "calib_minmax_only", None, 5.64),
    Table3Row("calib", "p99.9 only", "calib_p999_only", None, 8.03),
]
