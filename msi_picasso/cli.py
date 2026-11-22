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
    # viewer/ (for picasso-gui) always holds the ion images of the reported set; these add more (float16, M-1 / M /
    # M+1 on-tissue images: about 100 kB per query per 17k on-tissue pixels; kidney all targets ~0.5 GB, decoys ~2 GB)
    viewer_all_targets=False,
    viewer_all_decoys=False,
)
VIEWER_Q = 0.1  # the reported set: peptides and (picked) proteins at q <= this
VIEWER_TOP_DECOYS = 50


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


def save_viewer(out, cfg, rows, pt, prt, Q, img, prof, layout, prof2, mob):
    """viewer/ for picasso-gui: images.npy = float16 M-1 / M / M+1 on-tissue images of the saved queries, each scaled
    to its max (meta scale); slot_images.npy = float16 best-component M images of the saved queries that switch
    (meta slot_qi, slot_scale); meta.npz = which Q rows (qi), scale, on-tissue pixel index + grid shape, the TIC
    image, the +-25 ppm m/z profile of every query, and per saved query the M-1 / M profiles (prof2, +-60 ppm), the
    M mobilogram (mob_mg on the 1/K0 bins mob_kb) and its components (mob_cen, mob_area, mob_siso, mob_best). Saved queries: the reported set (peptides at q <= VIEWER_Q, the target
    peptides of proteins at picked q <= VIEWER_Q, GT, the top decoys) plus all targets / decoys on request.
    Read-only use of the run's arrays: nothing here feeds the score."""
    from msi_picasso.extract import tdf_meta
    v = Path(out) / "viewer"
    v.mkdir(exist_ok=True)
    tg = pt[~pt.is_decoy]
    prots = prt[~prt.is_decoy & (prt.q_picked <= VIEWER_Q)].protein
    peps = set(pt.index[pt.q <= VIEWER_Q]) | set(tg.index[tg.protein.isin(prots) | tg.is_gt])
    peps |= set(pt[pt.is_decoy].nlargest(VIEWER_TOP_DECOYS, "s").index)
    keep = rows.peptide.isin(peps)
    if cfg["viewer_all_targets"]:
        keep |= ~rows.is_decoy
    if cfg["viewer_all_decoys"]:
        keep |= rows.is_decoy
    qi = np.unique(rows.qi[keep].to_numpy())
    scale = np.maximum(img[qi].max(axis=2), 1e-30)  # (n, 3); float16 holds the ratio, not raw counts (overflow)
    np.save(v / "images.npy", (img[qi] / scale[:, :, None]).astype(np.float16))
    grid, shape, tic, _ = tdf_meta(cfg["maldi_d"])
    tic_img = np.full(shape[0] * shape[1], np.nan, np.float32)
    tic = tic[tic.index < len(grid)]
    px = grid[tic.index.to_numpy()]
    tic_img[px[px >= 0]] = tic.to_numpy()[px >= 0]
    sw = Q.mob_switch.to_numpy()
    slot_qi = qi[sw[qi]]
    si = mob["img"][(np.cumsum(sw) - 1)[slot_qi]]
    slot_scale = np.maximum(si.max(1), 1e-30)
    np.save(v / "slot_images.npy", (si / slot_scale[:, None]).astype(np.float16))
    np.savez(v / "meta.npz", qi=qi, scale=scale.astype(np.float32), pix=layout[0], shape=np.array(layout[1]),
             tic=tic_img, prof=prof.astype(np.float32), prof2=prof2[qi].astype(np.float32), mob_kb=mob["kb"],
             mob_mg=mob["mg"][qi], mob_cen=mob["cen"][qi], mob_area=mob["area"][qi], mob_siso=mob["siso"][qi],
             mob_best=mob["best"][qi], slot_qi=slot_qi, slot_scale=slot_scale.astype(np.float32))
    return len(qi)


def run(cfg, log=print):
    from importlib.metadata import version
    from msi_picasso.candidates import read_targets, world
    from msi_picasso.extract import CEN_PPM, PW, extract, pair_r_fn, slot_r_fn, tdf_meta
    from msi_picasso.score import ids, mz_offset, profile_features, score_run
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    steps, t, t0 = [], time.time(), time.time()
    n_steps = 7 if cfg["figures"] else 6

    def step(name):  # wall time + peak RSS per step, for run.json
        nonlocal t
        steps.append(dict(step=name, seconds=round(time.time() - t, 1),
                          peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)))
        t = time.time()

    def say(i, name, msg):  # one line per step: [mm:ss] i/n Name   what happened
        m, sec = divmod(int(time.time() - t0), 60)
        log(f"[{m:02d}:{sec:02d}] {i}/{n_steps} {name:<11} {msg}")

    def more(msg):  # continuation line, aligned with the message column
        log(" " * 24 + msg)

    log(f"MSI-PICASSO {version('MSI-PICASSO')}: {Path(cfg['maldi_d']).name} x {Path(cfg['lcms_peptides']).name}")
    log(f"output: {out}")
    targets = read_targets(cfg["lcms_peptides"], cfg["psm_utils_reader"])
    step("targets")
    say(1, "Targets", f"{len(targets):,} peptides in the PSM file (hydroxyproline forms included)")
    mz_range = cfg["mz_range"] or tdf_meta(cfg["maldi_d"])[3]
    c = world(targets, cfg["decoy_seeds"], cfg["decoy_method"], cfg["decoy_min_shift_da"], mz_range)
    step("decoys")
    mzs = np.unique(c.feature_mz.to_numpy())
    say(2, "Decoys", f"{c.peptide[c.is_decoy].nunique():,} decoy peptides ({cfg['decoy_method']}, "
                     f"{len(cfg['decoy_seeds'])} seeds); {len(mzs):,} m/z to read")
    mask = np.load(cfg["tissue_mask"]) if cfg["tissue_mask"] else None
    say(3, "Evidence", f"reading {Path(cfg['maldi_d']).name}")
    Q, X, img, prof2, layout, mob = extract(cfg["maldi_d"], mzs, cfg["extraction_ppm"], cfg["tic_quantile"],
                                            cfg["tic_normalize"], mask, cfg["threads"], more)
    prof = prof2[:, 1, PW - CEN_PPM:PW + CEN_PPM + 1]  # M, -25..25 ppm (viewer, figures)
    c["qi"] = np.searchsorted(Q.mz.to_numpy(), c.feature_mz.to_numpy())
    assert (Q.mz.to_numpy()[c.qi] == c.feature_mz.to_numpy()).all(), "candidate without its own query m/z"
    pair_r = pair_r_fn(X, Q.valid.to_numpy())
    step("evidence")
    auto = cfg["mz_offset_ppm"] == "auto"
    o_auto = mz_offset(c, Q, pair_r) if auto else (None, None, None)
    o = o_auto[0] if auto else float(cfg["mz_offset_ppm"])
    step("mz_offset")
    if not auto:
        say(4, "m/z offset", f"{o:+.2f} ppm (set in the config)")
    else:
        say(4, "m/z offset", f"{o:+.2f} ppm (estimate {o_auto[1]:+.2f} ppm, z {o_auto[2]:.1f}, "
                             f"{'accepted' if o != 0 else 'not accepted: needs z > 6 and at least 1 ppm'})")
    Q, prof_run = profile_features(Q, prof2, o)  # needs the offset
    rows, pt, prt = score_run(c, Q, slot_r_fn(mob["XS"], Q.mob_switch.to_numpy(), X, Q.valid.to_numpy()), o)
    step("score")
    say(5, "Score", f"{len(rows):,} candidates scored; {int(rows.removed.sum()):,} lost their ion to a better target")
    gt = gt_set(cfg["gt_peptides"]) if cfg["gt_peptides"] else set()
    pt["is_gt"] = pt.index.isin(gt)
    rows.to_parquet(out / "rows.parquet")
    Q.to_parquet(out / "evidence.parquet")
    pt.drop(columns="qi").to_csv(out / "peptides.tsv", sep="\t")
    prt.drop(columns="qi").to_csv(out / "proteins.tsv", sep="\t")
    n_view = save_viewer(out, cfg, rows, pt, prt, Q, img, prof, layout, prof2, mob)
    info = dict(config=cfg, n_targets=int((~pt.is_decoy).sum()), n_decoys=int(pt.is_decoy.sum()),
                mz_offset=dict(used=o, estimate=o_auto[1], z=o_auto[2], accepted=o_auto[0]),
                profile=prof_run, mob_switched=int(Q.mob_switch.sum()),
                ids={f"{a:g}": ids(pt, a) for a in (.01, .05, .1)},
                proteins_picked={f"{a:g}": ids(prt, a, "q_picked") for a in (.01, .05, .1)},
                proteins_nc={f"{a:g}": ids(prt, a) for a in (.01, .05, .1)},
                gt={f"{a:g}": int(((pt.q <= a) & pt.is_gt).sum()) for a in (.01, .05, .1)} if gt else None)
    step("output")
    say(6, "Output", f"peptides.tsv, proteins.tsv, run.json; viewer/ with ion images of {n_view:,} m/z")
    if cfg["figures"]:
        from msi_picasso.figures import save_figures
        save_figures(out / "figures", rows, pt, prt, Q, img, prof, layout, gt, o)
        step("figures")
        say(7, "Figures", "PNG check figures in figures/")
    info["steps"] = steps
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    m, sec = divmod(int(time.time() - t0), 60)
    peak = max(x["peak_rss_gb"] for x in steps)
    fmt = lambda d: " / ".join(str(v) for v in d.values())
    log(f"Done in {m}:{sec:02d}. Peptides {fmt(info['ids'])} at 1 / 5 / 10% FDR; proteins (picked) "
        f"{fmt(info['proteins_picked'])}" + (f"; GT {fmt(info['gt'])}" if gt else "") + f". Peak memory {peak:.0f} GB.")
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
