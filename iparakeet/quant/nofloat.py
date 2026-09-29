"""Runtime guard that the integer model produces no floating-point tensors."""

import contextlib
import threading

import torch
from torch.overrides import TorchFunctionMode

_state = threading.local()


class FloatLeakError(RuntimeError):
    pass


@contextlib.contextmanager
def exact_float_emulation():
    """Mark a region that emulates an integer primitive exactly with float64 (e.g. INT8 GEMM on CPU BLAS)."""
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _tensors(item)


class NoFloatMode(TorchFunctionMode):
    """Raise FloatLeakError when any torch op inside the context returns a floating/complex tensor."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if getattr(_state, "depth", 0) == 0:
            for t in _tensors(out):
                if t.is_floating_point() or t.is_complex():
                    raise FloatLeakError(f"{getattr(func, '__name__', func)} produced a {t.dtype} tensor")
        return out
