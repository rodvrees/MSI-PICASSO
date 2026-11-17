"""The package's checks: each module's self-check, and bench parity layer A (exact) when its references exist.
Run from outside /home/robbe/MALDI_MSI_score (its psm_utils checkout shadows the installed package by name)."""
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("module", ["candidates", "score", "extract"])
def test_self_check(module):
    runpy.run_module(f"msi_picasso.{module}", run_name="__main__")


def test_cli_self_check():
    from msi_picasso.cli import _self_check
    _self_check()


@pytest.mark.skipif(not Path("/home/robbe/MALDI_MSI_score/port_ref").exists(), reason="no bench references")
def test_bench_parity_layer_a():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_bench_parity
    for ds in check_bench_parity.DS:
        check_bench_parity.layer_a(ds)  # asserts inside
