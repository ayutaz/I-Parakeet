import math

import numpy as np
import onnx
import pytest
import torch

from conftest import randomize_, tiny_config
from iparakeet.analysis.range import calibrate
from iparakeet.deploy.emulate import ort_session
from iparakeet.deploy.encodings import build_encodings, missing_encodings
from iparakeet.deploy.onnx_graph import ALLOWED_OPS, build_onnx
from iparakeet.deploy.qdq import insert_qdq, unquantized_activation_inputs
from iparakeet.model.parakeet import ParakeetCTC, collect_taps
from iparakeet.sim.int_parakeet import IntParakeet
from iparakeet.sim.recipe import RECIPES

MAX_FRAMES = 120


def _inputs(n=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, 80, t, generator=gen), torch.tensor([t])) for t in (60, 90, 120, 75)[:n]]


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = randomize_(ParakeetCTC(tiny_config())).eval()
    stats = calibrate(model, lambda: iter(_inputs()), n_bins=512, per_channel=("conv.dw",))
    return model, stats


def _run(model_proto, feats):
    return ort_session(model_proto).run(["head.logits"], {"pre.in": feats[0].T.contiguous().numpy()})[0]


def _sqnr_db(ref, x):
    return 10 * math.log10(float((ref**2).sum() / ((ref - x) ** 2).sum()))


def test_fp_graph_matches_torch_for_each_bucket_length(setup):
    model, _ = setup
    for feats, lengths in _inputs(3):
        graph = build_onnx(model, n_frames=feats.shape[-1], max_frames=MAX_FRAMES)
        onnx.checker.check_model(graph)
        with torch.no_grad():
            ref, _ = model.forward_features(feats, lengths)
        np.testing.assert_allclose(_run(graph, feats), ref[0].numpy(), atol=1e-4)


def test_graph_uses_only_npu_friendly_ops_and_tap_names(setup):
    model, _ = setup
    graph = build_onnx(model, n_frames=90, max_frames=MAX_FRAMES)
    ops = {n.op_type for n in graph.graph.node}
    assert ops <= ALLOWED_OPS, ops - ALLOWED_OPS
    assert "Gather" in ops  # relative shift as a static index map
    produced = {o for n in graph.graph.node for o in n.output} | {i.name for i in graph.graph.input}
    taps = set()
    with collect_taps(model, lambda n, x: taps.add(n)), torch.no_grad():
        model.forward_features(*_inputs(1)[0])
    assert taps - {"L0.att.p", "L1.att.p"} <= produced, sorted(taps - produced)


def test_position_constant_is_the_shared_slice(setup):
    model, _ = setup
    long = {i.name: onnx.numpy_helper.to_array(i) for i in build_onnx(model, 120, MAX_FRAMES).graph.initializer}
    short = {i.name: onnx.numpy_helper.to_array(i) for i in build_onnx(model, 64, MAX_FRAMES).graph.initializer}
    p_long, p_short = long["L0.att.p"], short["L0.att.p"]  # (h, d_k, 2L-1)
    L_long, L_short = (p_long.shape[-1] + 1) // 2, (p_short.shape[-1] + 1) // 2
    np.testing.assert_array_equal(p_short, p_long[..., L_long - L_short : L_long + L_short - 1])


def test_encodings_cover_every_activation_and_mark_bn_output_int16(setup):
    model, stats = setup
    sim = IntParakeet(model, stats, RECIPES["iparakeet_lut"], MAX_FRAMES)
    graph = build_onnx(model, n_frames=90, max_frames=MAX_FRAMES)
    enc = build_encodings(sim, graph)
    assert missing_encodings(graph, enc) == []
    act = enc["activation_encodings"]
    assert act["L0.conv.dw"][0]["bitwidth"] == 16 and act["L0.ff1.lin1"][0]["bitwidth"] == 8
    assert act["pre.in"][0]["scale"] == pytest.approx(sim.S("pre.in"))
    e = act["L1.res2"][0]
    assert e["is_symmetric"] == "True" and e["offset"] == -128
    assert len(enc["param_encodings"]["L0.ff1.lin1.weight"]) == model.cfg.d_ff  # per-channel


def test_per_channel_activation_encodings_are_rejected(setup):
    model, stats = setup
    sim = IntParakeet(model, stats, RECIPES["bn_per_channel_int16"], MAX_FRAMES)
    with pytest.raises(ValueError, match="per-channel"):
        build_encodings(sim, build_onnx(model, 90, MAX_FRAMES))


def test_qdq_graph_quantizes_every_activation_and_tracks_the_integer_simulator(setup):
    model, stats = setup
    sim = IntParakeet(model, stats, RECIPES["iparakeet_lut"], MAX_FRAMES)
    vs_sim, vs_fp = [], []
    for feats, lengths in _inputs():
        graph = build_onnx(model, n_frames=feats.shape[-1], max_frames=MAX_FRAMES)
        qdq = insert_qdq(graph, build_encodings(sim, graph))
        assert unquantized_activation_inputs(qdq) == []
        out = torch.from_numpy(_run(qdq, feats))
        vs_sim.append(_sqnr_db(sim.logits(feats)[0].float(), out))
        with torch.no_grad():
            ref, _ = model.forward_features(feats, lengths)
        vs_fp.append(_sqnr_db(ref[0], out))
    # The QDQ graph sits closer to the integer simulator (~15-20 dB) than either does to FP32 (~13-15 dB).
    # Argmax agreement is not asserted: the tiny random model's 16 logits are near ties, so it swings
    # with last-bit float differences between CPUs.
    assert min(vs_sim) > 12
    assert min(vs_fp) > 5
