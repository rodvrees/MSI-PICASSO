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


def test_gui_config_round_trip():
    """A config loaded into the GUI form and written back is the same config, types included (0 stays int,
    2.0 float, lists lists, paths and 'auto' text), so a GUI run is the CLI run of that config."""
    pytest.importorskip("dash")
    from msi_picasso.gui import _show, config_from_toml, section_from_form
    text = ('[MSI-PICASSO]\nmaldi-d = "/x/a.d"\nlcms_peptides = "/x/p.tsv"\noutput_dir = "/x/out"\n'
            'tic_quantile = 0.25\ntic_normalize = false\ndecoy_min_shift_da = 2.0\nmz_offset_ppm = 0\n'
            'decoy_seeds = [0, 3]\nmz_range = [700.0, 3000.0]\nthreads = 8\nfigures = true\nold_key = 1\n')
    cfg, unknown = config_from_toml(text)
    assert unknown == ["old_key"] and "maldi_d" in cfg
    back = section_from_form({k: _show(v) for k, v in cfg.items()})
    assert back == cfg and all(type(back[k]) is type(cfg[k]) for k in cfg)
    assert section_from_form({"mz_offset_ppm": "auto", "gt_peptides": " /x/gt.txt ", "threads": ""}) == \
        {"mz_offset_ppm": "auto", "gt_peptides": "/x/gt.txt"}
