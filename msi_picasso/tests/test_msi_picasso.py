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


def test_gui_ids_y_range_follows_visible_x():
    """IDs vs FDR: the y range fits the targets reported within the visible q range, so a long tail of high-q
    targets does not flatten the curves."""
    pytest.importorskip("dash_ag_grid")
    import pandas as pd
    from msi_picasso.gui_results import ids_y_range
    pt = pd.DataFrame({"q": [0.001, 0.02, 0.05, 0.5, 0.9, 0.95, 0.2], "is_decoy": [False] * 6 + [True]})
    prt = pd.DataFrame({"q_picked": [0.01, 0.6], "is_decoy": [False, False]})
    d = {"pt": pt, "prt": prt}
    assert ids_y_range(d, 0.1)[1] < 5 < ids_y_range(d, 1.0)[1]


def test_gui_views_on_a_run():
    """Every results view builds on a real run directory (set PICASSO_TEST_RUN to one with viewer/)."""
    import os
    run = os.environ.get("PICASSO_TEST_RUN")
    if not run:
        pytest.skip("PICASSO_TEST_RUN not set")
    from pathlib import Path
    import msi_picasso.gui_results as r
    d = r._load(run, Path(run, "run.json").stat().st_mtime)
    for f in (r.ids_vs_fdr, r.mirror_fig, r.tail_fig, r.score_vs_mz, r.centroid_dist):
        assert f(d, 0.05).data
    for f in (r.term_dists, r.tissue_figs, r.mz_dist, r.pp_plot):
        assert f(d).data
    pt = d["pt"]
    for pep in (pt[~pt.is_decoy].q.idxmin(), pt.s.idxmin()):
        for f in (r.peptide_images, r.envelope_fig, r.profile_fig, r.terms_fig, r.score_position):
            f(d, pep)
    assert r.peptide_images(d, pt[~pt.is_decoy].q.idxmin()).data  # a reported peptide has its images saved
    prot = d["prt"][~d["prt"].is_decoy].q_picked.idxmin()
    r.protein_images(d, d["prt"].protein[prot], 0.05)
