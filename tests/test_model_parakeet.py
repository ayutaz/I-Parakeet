import io
import tarfile

import sentencepiece as spm
import torch
import yaml

from conftest import randomize_, tiny_config
from iparakeet.model.config import ParakeetConfig
from iparakeet.model.decoding import ctc_greedy_ids
from iparakeet.model.load_nemo import load_nemo
from iparakeet.model.parakeet import ParakeetCTC, collect_taps


def test_forward_shapes(tiny_model, tiny_cfg):
    audio = torch.randn(1, 16000) * 0.1
    with torch.no_grad():
        logits, lengths = tiny_model(audio, torch.tensor([16000]))
    assert logits.shape == (1, lengths.item(), tiny_cfg.vocab_size + 1)
    assert lengths.item() == 13  # 100 feature frames -> 50 -> 25 -> 13


def test_taps_cover_every_quantization_point(tiny_model):
    seen = {}
    with collect_taps(tiny_model, lambda name, x: seen.setdefault(name, x.shape)):
        with torch.no_grad():
            tiny_model.forward_features(torch.randn(1, 80, 64), torch.tensor([64]))
    for name in [
        "pre.in", "pre.conv0", "pre.dw1", "pre.pw1", "pre.dw2", "pre.pw2", "pre.out",
        "L0.ff1.ln", "L0.ff1.lin1", "L0.ff1.act", "L0.ff1.lin2", "L0.res1",
        "L0.att.ln", "L0.att.qu", "L0.att.qv", "L0.att.k", "L0.att.v", "L0.att.p",
        "L0.att.scores", "L0.att.probs", "L0.att.ctx", "L0.att.out", "L0.res2",
        "L0.conv.ln", "L0.conv.pw1", "L0.conv.gate", "L0.conv.glu", "L0.conv.dw",
        "L0.conv.act", "L0.conv.pw2", "L0.res3",
        "L1.ff2.ln", "L1.ff2.lin1", "L1.ff2.act", "L1.ff2.lin2", "L1.res4", "L1.out",
        "head.logits",
    ]:
        assert name in seen, name


def test_ctc_greedy_collapses_repeats_and_removes_blank():
    blank = 4
    frames = [0, 0, 4, 0, 1, 1, 4, 4, 2, 3, 3]
    logits = torch.full((len(frames), 5), -1.0)
    logits[torch.arange(len(frames)), torch.tensor(frames)] = 1.0
    assert ctc_greedy_ids(logits, blank) == [0, 0, 1, 2, 3]


def _train_tokenizer(tmp_path) -> spm.SentencePieceProcessor:
    text = tmp_path / "text.txt"
    text.write_text("\n".join(["hello world", "the quick brown fox", "jumps over the lazy dog"] * 20))
    spm.SentencePieceTrainer.train(
        input=str(text), model_prefix=str(tmp_path / "tok"), vocab_size=30, model_type="bpe",
        character_coverage=1.0, bos_id=-1, eos_id=-1,
    )
    return spm.SentencePieceProcessor(model_file=str(tmp_path / "tok.model"))


def _nemo_config(cfg: ParakeetConfig) -> dict:
    return {
        "sample_rate": 16000,
        "preprocessor": {
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
            "window_stride": 0.01, "window": "hann", "features": cfg.feat_in, "n_fft": 512,
            "log": True, "frame_splicing": 1, "dither": 1e-5, "pad_to": 0, "pad_value": 0.0,
        },
        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": cfg.feat_in, "n_layers": cfg.n_layers, "d_model": cfg.d_model,
            "subsampling": "dw_striding", "subsampling_factor": cfg.subsampling_factor,
            "subsampling_conv_channels": cfg.subsampling_conv_channels,
            "ff_expansion_factor": cfg.ff_expansion_factor, "self_attention_model": "rel_pos",
            "n_heads": cfg.n_heads, "xscaling": cfg.xscaling, "untie_biases": True,
            "pos_emb_max_len": cfg.pos_emb_max_len, "conv_kernel_size": cfg.conv_kernel_size,
            "conv_norm_type": "batch_norm",
        },
        "decoder": {
            "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
            "feat_in": cfg.d_model, "num_classes": cfg.vocab_size,
        },
        "tokenizer": {"dir": None, "type": "bpe", "model_path": "nemo:abc123_tokenizer.model"},
    }


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_load_nemo_roundtrip(tmp_path):
    sp = _train_tokenizer(tmp_path)
    cfg = tiny_config(vocab_size=sp.get_piece_size())
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(cfg)).eval()
    weights = io.BytesIO()
    torch.save(model.state_dict(), weights)
    nemo_path = tmp_path / "tiny.nemo"
    with tarfile.open(nemo_path, "w") as tar:
        _add_bytes(tar, "./model_config.yaml", yaml.safe_dump(_nemo_config(cfg)).encode())
        _add_bytes(tar, "./model_weights.ckpt", weights.getvalue())
        _add_bytes(tar, "./abc123_tokenizer.model", (tmp_path / "tok.model").read_bytes())

    bundle = load_nemo(nemo_path)
    assert bundle.config == cfg
    audio = torch.randn(1, 8000) * 0.1
    with torch.no_grad():
        expected, _ = model(audio, torch.tensor([8000]))
        got, _ = bundle.model(audio, torch.tensor([8000]))
    assert torch.allclose(got, expected)
    assert bundle.tokenizer.decode(sp.encode("hello world")) == "hello world"


def test_config_from_nemo_derives_frontend_settings():
    cfg = ParakeetConfig.from_nemo(_nemo_config(tiny_config()))
    assert (cfg.win_length, cfg.hop_length, cfg.n_fft) == (400, 160, 512)
    assert cfg.d_model == 32 and cfg.vocab_size == 16
