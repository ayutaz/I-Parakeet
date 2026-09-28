"""Model configuration for Parakeet-CTC (FastConformer encoder + CTC head)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ParakeetConfig:
    feat_in: int = 80
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 24
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    vocab_size: int = 1024  # excluding the CTC blank
    xscaling: bool = False
    pos_emb_max_len: int = 5000
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    preemph: float = 0.97
    log_zero_guard_value: float = 2.0**-24

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ff_expansion_factor

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads

    @property
    def blank_id(self) -> int:
        return self.vocab_size

    @classmethod
    def from_nemo(cls, cfg: dict) -> "ParakeetConfig":
        """Build from a NeMo `model_config.yaml` of an EncDecCTCModelBPE with a ConformerEncoder."""
        pre, enc, dec = cfg["preprocessor"], cfg["encoder"], cfg["decoder"]
        expected = {
            "subsampling": "dw_striding",
            "self_attention_model": "rel_pos",
            "conv_norm_type": "batch_norm",
        }
        for key, value in expected.items():
            if enc.get(key, value) != value:
                raise ValueError(f"unsupported encoder.{key}={enc.get(key)!r} (expected {value!r})")
        if not enc.get("untie_biases", True):
            raise ValueError("tied positional biases are not supported")
        if pre.get("normalize", "per_feature") != "per_feature":
            raise ValueError("only per_feature normalization is supported")
        sample_rate = int(pre.get("sample_rate", cfg.get("sample_rate", 16000)))
        win_length = int(pre.get("window_size", 0.02) * sample_rate)
        hop_length = int(pre.get("window_stride", 0.01) * sample_rate)
        n_fft = pre.get("n_fft") or 2 ** math.ceil(math.log2(win_length))
        num_classes = dec.get("num_classes", -1)
        if num_classes is None or num_classes <= 0:
            num_classes = len(dec["vocabulary"])
        guard = pre.get("log_zero_guard_value", 2.0**-24)
        return cls(
            feat_in=int(enc["feat_in"]),
            d_model=int(enc["d_model"]),
            n_heads=int(enc["n_heads"]),
            n_layers=int(enc["n_layers"]),
            ff_expansion_factor=int(enc.get("ff_expansion_factor", 4)),
            conv_kernel_size=int(enc.get("conv_kernel_size", 31)),
            subsampling_factor=int(enc.get("subsampling_factor", 4)),
            subsampling_conv_channels=int(enc.get("subsampling_conv_channels", -1)),
            vocab_size=int(num_classes),
            xscaling=bool(enc.get("xscaling", True)),
            pos_emb_max_len=int(enc.get("pos_emb_max_len", 5000)),
            sample_rate=sample_rate,
            n_fft=int(n_fft),
            win_length=win_length,
            hop_length=hop_length,
            preemph=float(pre.get("preemph", 0.97)),
            log_zero_guard_value=float(guard),
        )
