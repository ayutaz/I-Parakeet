"""Load a NeMo `.nemo` checkpoint (EncDecCTCModelBPE) into the standalone model without NeMo."""

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from iparakeet.model.config import ParakeetConfig
from iparakeet.model.decoding import Tokenizer
from iparakeet.model.parakeet import ParakeetCTC


@dataclass
class NemoBundle:
    model: ParakeetCTC
    tokenizer: Tokenizer | None
    config: ParakeetConfig
    raw_config: dict


def _read_members(path: Path) -> dict[str, bytes]:
    wanted = {}
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name in ("model_config.yaml", "model_weights.ckpt") or name.endswith("tokenizer.model"):
                wanted[name] = tar.extractfile(member).read()
    return wanted


def load_nemo(path, map_location: str = "cpu") -> NemoBundle:
    members = _read_members(Path(path))
    raw_cfg = yaml.safe_load(members["model_config.yaml"])
    cfg = ParakeetConfig.from_nemo(raw_cfg)
    model = ParakeetCTC(cfg)
    state = torch.load(io.BytesIO(members["model_weights.ckpt"]), map_location=map_location, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise KeyError(f"checkpoint is missing parameters: {missing[:10]}")
    ignored = [k for k in unexpected if not k.startswith(("preprocessor.", "spec_augmentation", "loss"))]
    if ignored:
        raise KeyError(f"checkpoint has unrecognized parameters: {ignored[:10]}")
    tok_name = next((n for n in members if n.endswith("tokenizer.model")), None)
    tokenizer = Tokenizer.from_bytes(members[tok_name]) if tok_name else None
    return NemoBundle(model.eval(), tokenizer, cfg, raw_cfg)
