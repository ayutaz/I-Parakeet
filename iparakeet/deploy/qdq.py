"""Insert QuantizeLinear/DequantizeLinear pairs from the encodings (fake-quantized graph for ORT).

Every activation tensor X is produced as X.raw, quantized to its INT grid and dequantized back to X;
weights and the position constant are stored as float and pass through a per-channel (or per-tensor)
Q/DQ pair. The result runs on onnxruntime CPU and is also the form QNN-style converters accept.
"""

import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper

from iparakeet.deploy.onnx_graph import encoding_rules

_DTYPES = {(8, True): np.int8, (16, True): np.int16, (8, False): np.uint8, (16, False): np.uint16}


def _qdq_nodes(src: str, dst: str, scale, bits: int, signed: bool, axis: int | None, inits: list) -> list:
    scale = np.asarray(scale, dtype=np.float32)
    zp = np.zeros(scale.shape, dtype=_DTYPES[(bits, signed)])
    s_name, z_name = f"{dst}.qscale", f"{dst}.qzp"
    inits += [numpy_helper.from_array(scale, s_name), numpy_helper.from_array(zp, z_name)]
    kw = {} if axis is None else {"axis": axis}
    return [
        helper.make_node("QuantizeLinear", [src, s_name, z_name], [f"{dst}.q"], name=f"{dst}.quant", **kw),
        helper.make_node("DequantizeLinear", [f"{dst}.q", s_name, z_name], [dst], name=f"{dst}.dequant", **kw),
    ]


def insert_qdq(model: onnx.ModelProto, encodings: dict) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    g = m.graph
    act, par = encodings["activation_encodings"], encodings["param_encodings"]
    param_axis = {k: v["axis"] for k, v in encoding_rules(model)["params"].items()}
    new_inits = [i for i in g.initializer if i.name not in par]
    nodes = []
    for init in g.initializer:
        if init.name in par:
            raw = numpy_helper.to_array(init)
            new_inits.append(numpy_helper.from_array(raw, init.name + ".fp"))
            enc = par[init.name]
            scale = [e["scale"] for e in enc] if len(enc) > 1 else enc[0]["scale"]
            nodes += _qdq_nodes(init.name + ".fp", init.name, scale, enc[0]["bitwidth"], True, param_axis[init.name] if len(enc) > 1 else None, new_inits)

    def enc_of(name):
        e = act[name][0]
        return e["scale"], e["bitwidth"], e["is_symmetric"] == "True"

    for inp in g.input:
        nodes += _qdq_nodes(inp.name, inp.name + ".dq", *enc_of(inp.name), None, new_inits)
    for node in g.node:
        node = copy.deepcopy(node)
        node.input[:] = [i + ".dq" if i in {x.name for x in g.input} else i for i in node.input]
        outs = list(node.output)
        node.output[:] = [o + ".raw" if o in act else o for o in outs]
        nodes.append(node)
        for o in outs:
            if o in act:
                nodes += _qdq_nodes(o + ".raw", o, *enc_of(o), None, new_inits)
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(new_inits)
    return m


def unquantized_activation_inputs(model: onnx.ModelProto) -> list[tuple[str, str]]:
    """(node, input) pairs where a compute op reads an activation that is not a DequantizeLinear output."""
    produced_by = {o: n.op_type for n in model.graph.node for o in n.output}
    inits = {i.name for i in model.graph.initializer}
    bad = []
    for n in model.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for i in n.input:
            if i and i not in inits and produced_by.get(i) != "DequantizeLinear":
                bad.append((n.name, i))
    return bad
