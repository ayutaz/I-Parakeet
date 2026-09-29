"""Generate NeMo reference outputs for the parity test (run in an environment with nemo-toolkit).

    python tests/nemo_parity/generate_reference.py tests/fixtures/nemo_tiny_reference.pt

Builds tiny random-weight NeMo modules with the Parakeet-CTC architecture (preprocessor,
dw_striding ConformerEncoder with rel_pos attention, ConvASRDecoder), runs a padded batch of two
utterances, and stores config, weights, inputs and outputs. The main test suite compares the
standalone implementation against this file without importing NeMo.
"""

import sys

import torch
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder, ConvASRDecoder


def nemo_config(xscaling: bool) -> dict:
    return {
        "sample_rate": 16000,
        "preprocessor": {
            "sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025, "window_stride": 0.01,
            "window": "hann", "features": 80, "n_fft": 512, "log": True, "frame_splicing": 1,
            "dither": 0.0, "pad_to": 0, "pad_value": 0.0,
        },
        "encoder": {
            "feat_in": 80, "n_layers": 2, "d_model": 32, "subsampling": "dw_striding", "subsampling_factor": 8,
            "subsampling_conv_channels": 8, "ff_expansion_factor": 4, "self_attention_model": "rel_pos",
            "n_heads": 4, "att_context_size": [-1, -1], "xscaling": xscaling, "untie_biases": True,
            "pos_emb_max_len": 512, "conv_kernel_size": 9, "conv_norm_type": "batch_norm",
            "dropout": 0.0, "dropout_pre_encoder": 0.0, "dropout_emb": 0.0, "dropout_att": 0.0,
        },
        "decoder": {"feat_in": 32, "num_classes": 16},
    }


class Container(torch.nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**cfg["preprocessor"])
        self.encoder = ConformerEncoder(**cfg["encoder"])
        self.decoder = ConvASRDecoder(**cfg["decoder"])


def randomize(model: torch.nn.Module, gen: torch.Generator) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "pos_bias" in name:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=gen) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=gen) + 0.5)
                m.weight.copy_(torch.rand(m.weight.shape, generator=gen) + 0.5)
                m.bias.copy_(torch.randn(m.bias.shape, generator=gen) * 0.1)


def plain(x):
    """Strip NeMo/typed tensor subclasses so the fixture loads without NeMo."""
    if isinstance(x, torch.Tensor):
        return torch.from_numpy(x.detach().cpu().numpy().copy())
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def main(out_path: str) -> None:
    gen = torch.Generator().manual_seed(1234)
    audio = torch.randn(2, 20000, generator=gen) * 0.1
    lengths = torch.tensor([13000, 20000])
    audio[0, 13000:] = 0.0
    variants = {}
    for xscaling in (False, True):
        torch.manual_seed(7)
        cfg = nemo_config(xscaling)
        model = Container(cfg).eval()
        randomize(model, gen)
        with torch.no_grad():
            feats, feat_len = model.preprocessor(input_signal=audio, length=lengths)
            enc, enc_len = model.encoder(audio_signal=feats, length=feat_len)
            log_probs = model.decoder(encoder_output=enc)
        variants[f"xscaling={xscaling}"] = {
            "config": cfg,
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "features": feats,
            "feature_lengths": feat_len,
            "encoder_output": enc.transpose(1, 2),
            "encoder_lengths": enc_len,
            "log_probs": log_probs,
        }
    torch.save(plain({"audio": audio, "audio_lengths": lengths, "variants": variants}), out_path)


if __name__ == "__main__":
    main(sys.argv[1])
