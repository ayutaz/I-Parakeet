import importlib
import sys

import pytest

SCRIPTS = [
    "choose_buckets",
    "eval_device",
    "eval_fp32",
    "eval_sim",
    "export_npu",
    "kernel_report",
    "prepare_data",
    "prepare_device_inputs",
    "range_analysis",
    "run_ablation",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_help_renders(name, capsys):
    module = importlib.import_module(f"scripts.{name}")
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_fit_swish_approx_help_renders(monkeypatch, capsys):
    from scripts import fit_swish_approx

    monkeypatch.setattr(sys, "argv", ["fit_swish_approx.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        fit_swish_approx.main()
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
