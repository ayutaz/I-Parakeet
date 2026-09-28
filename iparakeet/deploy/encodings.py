"""QNN / AIMET-style quantization encodings for the ONNX graph, taken from the integer simulator.

Activation encodings follow the simulator's per-tensor grids (bit width, symmetric scale). The
JSON uses the AIMET convention for signed symmetric grids (offset = -2^(b-1)); the simulator itself
uses the restricted range +-(2^(b-1)-1) of paper Eq. (6). Weights are per-output-channel INT8.
"""

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from iparakeet.deploy.onnx_graph import encoding_rules
from iparakeet.intops.linear import quantize_weight_per_channel
from iparakeet.quant.fixed_point import qmax
from iparakeet.sim.int_parakeet import IntParakeet


def signed_encoding(scale: float, bits: int) -> dict:
    return {
        "bitwidth": bits, "dtype": "int", "is_symmetric": "True", "scale": float(scale),
        "offset": -(2 ** (bits - 1)), "min": -(2 ** (bits - 1)) * float(scale), "max": (2 ** (bits - 1) - 1) * float(scale),
    }


def unsigned_encoding(scale: float, bits: int) -> dict:
    return {
        "bitwidth": bits, "dtype": "int", "is_symmetric": "False", "scale": float(scale),
        "offset": 0, "min": 0.0, "max": (2**bits - 1) * float(scale),
    }


def _activation(sim: IntParakeet, rule: list) -> dict:
    kind = rule[0]
    if kind == "fixed_unsigned":
        bits = rule[1]
        return unsigned_encoding(1.0 / (2**bits - 1), bits)
    if kind == "fixed_gate":
        bits = sim.recipe.act_bits
        return signed_encoding(1.0 / qmax(bits), bits)
    _, name, factor, bits = rule
    scale = sim.S(name)
    if isinstance(scale, torch.Tensor) and scale.numel() > 1:
        raise ValueError(f"{name}: per-channel activation encodings are not deployable on the NPU")
    bits = bits or sim.bits(name)
    s = float(scale) * factor
    if bits != sim.bits(name):  # same real range on a wider grid
        s = float(scale) * factor * qmax(sim.bits(name)) / qmax(bits)
    return signed_encoding(s, bits)


def build_encodings(sim: IntParakeet, model: onnx.ModelProto) -> dict:
    rules = encoding_rules(model)
    activations = {name: [_activation(sim, rule)] for name, rule in rules["activations"].items()}
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    params = {}
    for name, info in rules["params"].items():
        w = torch.from_numpy(np.array(inits[name])).double()
        if info["axis"] is None:  # position constant: the one INT8 grid shared by every bucket
            layer = int(name.split(".")[0][1:])
            params[name] = [signed_encoding(sim.blocks[layer].att.s_p, 8)]
            continue
        w = w if info["axis"] == 0 else w.transpose(0, info["axis"])
        _, scales = quantize_weight_per_channel(w, sim.recipe.weight_bits)
        params[name] = [signed_encoding(float(s), sim.recipe.weight_bits) for s in scales]
    return {"version": "0.6.1", "activation_encodings": activations, "param_encodings": params}


def missing_encodings(model: onnx.ModelProto, encodings: dict) -> list[str]:
    """Float activation tensors without an encoding (each would force an FP op on the NPU)."""
    act = encodings["activation_encodings"]
    tensors = [i.name for i in model.graph.input] + [o for n in model.graph.node for o in n.output]
    return [t for t in tensors if t not in act]
