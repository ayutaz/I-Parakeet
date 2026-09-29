"""Parity of the standalone model against real NeMo modules (fixture from tests/nemo_parity/)."""

from pathlib import Path

import pytest
import torch

from iparakeet.model.config import ParakeetConfig
from iparakeet.model.parakeet import ParakeetCTC

FIXTURE = Path(__file__).parent / "fixtures" / "nemo_tiny_reference.pt"


@pytest.fixture(scope="module")
def reference():
    return torch.load(FIXTURE, weights_only=False)


@pytest.fixture(scope="module", params=["xscaling=False", "xscaling=True"])
def variant(request, reference):
    v = reference["variants"][request.param]
    model = ParakeetCTC(ParakeetConfig.from_nemo(v["config"])).eval()
    missing, unexpected = model.load_state_dict(v["state_dict"], strict=False)
    assert not missing, missing
    return model, v, reference


def test_state_dict_keys_are_nemo_keys(variant):
    model, v, _ = variant
    ours = set(model.state_dict())
    assert ours <= set(v["state_dict"]), sorted(ours - set(v["state_dict"]))


def test_features_match_nemo(variant):
    model, v, ref = variant
    feats, lengths = model.preprocessor(ref["audio"], ref["audio_lengths"])
    assert torch.equal(lengths, v["feature_lengths"])
    for b, n in enumerate(lengths.tolist()):
        assert torch.allclose(feats[b, :, :n], v["features"][b, :, :n], atol=1e-4)


def test_padded_batch_log_probs_match_nemo(variant):
    model, v, _ = variant
    with torch.no_grad():
        logits, lengths = model.forward_features(v["features"], v["feature_lengths"])
    assert torch.equal(lengths.long(), v["encoder_lengths"].long())
    log_probs = torch.log_softmax(logits, dim=-1)
    for b, n in enumerate(lengths.tolist()):
        assert torch.allclose(log_probs[b, :n], v["log_probs"][b, :n], atol=1e-4)


def test_single_utterance_on_cropped_features_matches_nemo(variant):
    # Batch-1 on features cropped to their valid length (the static-shape path used later).
    model, v, _ = variant
    n = int(v["feature_lengths"][0])
    with torch.no_grad():
        logits, lengths = model.forward_features(v["features"][:1, :, :n], torch.tensor([n]))
    m = int(v["encoder_lengths"][0])
    assert lengths.item() == m
    assert torch.allclose(torch.log_softmax(logits[0], -1), v["log_probs"][0, :m], atol=1e-4)
