"""Run qnn-net-run style input lists with onnxruntime on (QDQ) ONNX graphs.

Stands in for the device when none is available: same input lists, same Result_<i>/<output>.raw
layout, so the M7 evaluation code is exercised end to end. It is a fake-quantized emulation, not
the HTP arithmetic.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def run_input_lists_with_ort(graphs: dict[str, onnx.ModelProto], inputs_root, outputs_root) -> None:
    for name, graph in graphs.items():
        list_file = Path(inputs_root) / name / "input_list.txt"
        if not list_file.exists():
            continue
        sess = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])
        shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
        for i, line in enumerate(l for l in list_file.read_text().splitlines() if l.strip()):
            path = line.split(":=", 1)[1]
            x = np.fromfile(path, dtype=np.float32).reshape(shape)
            (out,) = sess.run(["head.logits"], {"pre.in": x})
            folder = Path(outputs_root) / name / f"Result_{i}"
            folder.mkdir(parents=True, exist_ok=True)
            out.astype(np.float32).tofile(folder / "head.logits.raw")
