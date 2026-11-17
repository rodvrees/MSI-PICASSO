"""picasso -c config.toml: the adopted MSI-PICASSO method end to end.

Steps: targets (PSM file) -> decoys (K seeds) -> evidence at every query m/z (.d) -> m/z offset -> score ->
per-ion competition -> peptide q -> proteins -> output. Each step's wall time and the peak RSS after it go to run.json.
"""
import argparse
import json
import resource
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = dict(
    maldi_d=None, lcms_peptides=None, output_dir=None,  # required
    psm_utils_reader="fragpipe",
    decoy_method="d1c_raw",  # or "d1b" (fails G0: a known sequence-level leak)
    decoy_seeds=list(range(10)),  # K = 10 decoy sets
    decoy_min_shift_da=None,  # a decoy differs from its target by at least this (Da); None: 10 ppm
    extraction_ppm=10.0,
    tic_quantile=0.25,  # on-tissue pixels: frame TIC >= this quantile
    tic_normalize=False,  # divide each pixel by its TIC over all queries before coloc
    mz_offset_ppm="auto",  # label-free m/z offset for the flank term, or a number
    tissue_mask=None,  # optional .npy (H * W) bool mask instead of the TIC rule
    mz_range=None,  # optional [lo, hi]; default the .d's acquired range
    gt_peptides=None,  # optional list of known peptides, for the report and figures only
    figures=False,
    threads=32,
)


def load_config(path, **override):
    cfg = dict(DEFAULTS)
    if path:
        user = tomllib.loads(Path(path).read_text())["MSI-PICASSO"]
        user = {k.replace("-", "_"): v for k, v in user.items()}
        unknown = set(user) - set(DEFAULTS)
        assert not unknown, f"unknown config keys: {sorted(unknown)}"
        cfg.update(user)
    cfg.update({k: v for k, v in override.items() if v is not None})
    missing = [k for k in ("maldi_d", "lcms_peptides", "output_dir") if not cfg[k]]
    assert not missing, f"missing config keys: {missing}"
    return cfg


def gt_set(path):
    """Known peptides as candidate names: P[Oxidation] (or P[16]) forms -> '<seq>+h<n>'; other modified lines are
    unreachable (no such candidate)."""
    lines = [x.strip().replace("[16]", "[Oxidation]") for x in Path(path).read_text().splitlines() if x.strip()]
    out = {x for x in lines if "[" not in x}
    return out | {x.replace("[Oxidation]", "") + f"+h{x.count('P[Oxidation]')}" for x in lines
                  if "[" in x and x.count("[") == x.count("P[Oxidation]")}


def run(cfg, log=print):
    from msi_picasso.candidates import read_targets, world
    from msi_picasso.extract import extract, pair_r_fn, tdf_meta
    from msi_picasso.score import ids, mz_offset, score_run
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    steps, t = [], time.time()

    def step(name):
        nonlocal t
        steps.append(dict(step=name, seconds=round(time.time() - t, 1),
                          peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)))
        log(f"{name}: {steps[-1]['seconds']} s, peak RSS {steps[-1]['peak_rss_gb']} GB")
        t = time.time()

    targets = read_targets(cfg["lcms_peptides"], cfg["psm_utils_reader"])
    step("targets")
    mz_range = cfg["mz_range"] or tdf_meta(cfg["maldi_d"])[3]
    c = world(targets, cfg["decoy_seeds"], cfg["decoy_method"], cfg["decoy_min_shift_da"], mz_range)
    step("decoys")
    mzs = np.unique(c.feature_mz.to_numpy())
    mask = np.load(cfg["tissue_mask"]) if cfg["tissue_mask"] else None
    Q, X, img, prof, layout = extract(cfg["maldi_d"], mzs, cfg["extraction_ppm"], cfg["tic_quantile"], cfg["tic_normalize"],
                              mask, cfg["threads"], log)
    c["qi"] = np.searchsorted(Q.mz.to_numpy(), c.feature_mz.to_numpy())
    assert (Q.mz.to_numpy()[c.qi] == c.feature_mz.to_numpy()).all(), "candidate without its own query m/z"
    pair_r = pair_r_fn(X, Q.valid.to_numpy())
    step("evidence")
    auto = cfg["mz_offset_ppm"] == "auto"
    o_auto = mz_offset(c, Q, pair_r) if auto else (None, None, None)
    o = o_auto[0] if auto else float(cfg["mz_offset_ppm"])
    step("mz_offset")
    rows, pt, prt = score_run(c, Q, pair_r, o)
    step("score")
    gt = gt_set(cfg["gt_peptides"]) if cfg["gt_peptides"] else set()
    pt["is_gt"] = pt.index.isin(gt)
    rows.to_parquet(out / "rows.parquet")
    Q.to_parquet(out / "evidence.parquet")
    pt.drop(columns="qi").to_csv(out / "peptides.tsv", sep="\t")
    prt.drop(columns="qi").to_csv(out / "proteins.tsv", sep="\t")
    info = dict(config=cfg, n_targets=int((~pt.is_decoy).sum()), n_decoys=int(pt.is_decoy.sum()),
                mz_offset=dict(used=o, estimate=o_auto[1], z=o_auto[2], accepted=o_auto[0]),
                ids={f"{a:g}": ids(pt, a) for a in (.01, .05, .1)},
                proteins_picked={f"{a:g}": ids(prt, a, "q_picked") for a in (.01, .05, .1)},
                proteins_nc={f"{a:g}": ids(prt, a) for a in (.01, .05, .1)},
                gt={f"{a:g}": int(((pt.q <= a) & pt.is_gt).sum()) for a in (.01, .05, .1)} if gt else None)
    step("output")
    if cfg["figures"]:
        from msi_picasso.figures import save_figures
        save_figures(out / "figures", rows, pt, prt, Q, img, prof, layout, gt)
        step("figures")
    info["steps"] = steps
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    log(f"IDs at 1 / 5 / 10%: {list(info['ids'].values())}; proteins (picked): {list(info['proteins_picked'].values())}")
    return rows, pt, prt, Q, info


def main():
    ap = argparse.ArgumentParser(prog="picasso", description=__doc__)
    ap.add_argument("-c", "--config", help="TOML with an [MSI-PICASSO] table (keys: cli.DEFAULTS)")
    ap.add_argument("-o", "--output-dir")
    ap.add_argument("--figures", action="store_true", default=None)
    a = ap.parse_args()
    run(load_config(a.config, output_dir=a.output_dir, figures=a.figures))


def _self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "gt.txt"
        f.write_text("AGLQFPVGR\nGVVGLP[16]GQR\nGSAP[Oxidation]GP[Oxidation]GR\nM[Oxidation]AGK\n")
        assert gt_set(f) == {"AGLQFPVGR", "GVVGLPGQR+h1", "GSAPGPGR+h2"}
        c = Path(tmp) / "c.toml"
        c.write_text('[MSI-PICASSO]\nmaldi-d = "x.d"\nlcms_peptides = "p.tsv"\ntic_quantile = 0.5\n')
        cfg = load_config(c, output_dir=tmp)
        assert cfg["maldi_d"] == "x.d" and cfg["tic_quantile"] == 0.5 and cfg["decoy_method"] == "d1c_raw"
        c.write_text('[MSI-PICASSO]\ntypo_key = 1\n')
        try:
            load_config(c)
            raise SystemExit("unknown key accepted")
        except AssertionError:
            pass
    print("cli ok")


if __name__ == "__main__":  # with arguments: the CLI; without: the self-checks
    main() if len(__import__("sys").argv) > 1 else _self_check()
