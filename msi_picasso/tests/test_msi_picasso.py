"""The package's checks: each module's self-check. Bench parity: scripts/check_bench_parity.py at tag
port-parity-exact. Run from outside /home/robbe/MALDI_MSI_score (its psm_utils checkout shadows psm_utils)."""
import runpy

import pytest


@pytest.mark.parametrize("module", ["candidates", "score", "extract"])
def test_self_check(module):
    runpy.run_module(f"msi_picasso.{module}", run_name="__main__")


def test_cli_self_check():
    from msi_picasso.cli import _self_check
    _self_check()
