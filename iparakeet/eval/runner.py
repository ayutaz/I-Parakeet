"""Audio loading and batched greedy transcription of a manifest."""

from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from iparakeet.eval.manifest import ManifestEntry
from iparakeet.model.decoding import ctc_greedy_ids


def load_audio(path, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    return audio


def _batches(entries: list[ManifestEntry], batch_size: int):
    order = sorted(range(len(entries)), key=lambda i: entries[i].duration)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def transcribe_manifest(model, tokenizer, entries: list[ManifestEntry], batch_size: int = 16, device: str = "cpu") -> list[str]:
    """Greedy CTC transcription; batches are formed from similar durations to limit padding."""
    model = model.to(device).eval()
    texts: list[str | None] = [None] * len(entries)
    for idx in _batches(entries, batch_size):
        audios = [load_audio(entries[i].audio_filepath, model.cfg.sample_rate) for i in idx]
        lengths = torch.tensor([len(a) for a in audios])
        batch = torch.zeros(len(audios), int(lengths.max()))
        for row, a in enumerate(audios):
            batch[row, : len(a)] = torch.from_numpy(a)
        logits, enc_lengths = model(batch.to(device), lengths.to(device))
        for row, i in enumerate(idx):
            ids = ctc_greedy_ids(logits[row, : int(enc_lengths[row])].cpu(), model.cfg.blank_id)
            texts[i] = tokenizer.decode(ids)
    return texts


@torch.no_grad()
def iter_features(model, entries: list[ManifestEntry], device: str = "cpu"):
    """Yield (features cropped to their valid length, lengths) one utterance at a time."""
    for entry in entries:
        audio = torch.from_numpy(load_audio(entry.audio_filepath, model.cfg.sample_rate))[None].to(device)
        feats, lengths = model.preprocessor(audio, torch.tensor([audio.shape[1]], device=device))
        n = int(lengths[0])
        yield feats[:, :, :n], lengths
