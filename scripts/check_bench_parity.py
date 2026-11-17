"""Bench parity of the package (RD 2026-11-17: layer A exact, layer B explained).

Valid at tag port-parity-exact (run it from a worktree of that tag): later commits fix three bench behaviours on
purpose (exact target masses, exclusion of decoys whose parent is out of range, parent partners excluded by m/z),
so the candidate set no longer equals the bench's.

Layer A (method): the package's candidates and score, fed with the bench store's evidence (peaks.parquet columns and
corr_protein), must reproduce the bench's real world rep 0 exactly: same rows, same terms, same IDs, same proteins.
References: make_bench_ref.py (bench code, frozen package) -> /home/robbe/MALDI_MSI_score/port_ref/.
Layer B (extraction): the package's own evidence table (extract.py, cached by the run in <out>/evidence.parquet) per
term against the store; then the full package run's IDs against the bench. Run:
  python MSI-PICASSO/scripts/check_bench_parity.py A [ds ...]
  python MSI-PICASSO/scripts/check_bench_parity.py B <ds> <d1b|d1c_raw> <run output dir>
(run from outside the MALDI_MSI_score folder: its psm_utils checkout shadows the package by name).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from msi_picasso.candidates import read_targets, world
from msi_picasso.score import ids, score_run

R = Path("/home/robbe/MALDI_MSI_score")
REF = R / "port_ref"
# dataset -> PSM file, psm_utils reader, adopted store, decoy floor (Da; None = 10 ppm), (D1b id, D1C_RAW id), offset
DS = {
    "kidney": ("data/kidney/psm_Lme025_kidney.tsv", "fragpipe", "kidney_raw_seeds20_ppm10_isofit_sharp_hyplc_mzoff", 2.0,
               ("429c0e4a59", "df0c1354fe"), -1.13),
    "amyloidosis": ("data/amyloidosis/fragpipe_output_amyloidosis/Amyl_tissue_psm.tsv", "infer",
                    "amyloidosis_raw_seeds20_ppm10_isofit_flank_sharp_hyplc", None, ("7417ddd3e6", "771c8db220"), 0.0),
    "her2": ("data/her2/psm_Lme022_Her2pos.tsv", "fragpipe", "her2_raw_seeds20_ppm10_isofit_sharp_hyplc", 2.0,
             ("230058d2e4", "78ba6a068c"), 0.0),
}
BENCH_COLS = {"coloc": "protein_colocalization_weighted_s4", "iso": "isotope_image_colocalization_m1",
              "kl": "averagine_envelope_kl", "mm1": "isofit_mm1_r_hinge0.5", "flank": "peakfit_cen25_hipix_ppm_hinge2"}
SEEDS = [int(x) for x in np.random.default_rng(7919).choice(list(range(20)), 10, replace=False)]  # bench rep 0 draw


def store_evidence(store):
    """The bench store as a package evidence table Q, and pair_r from its packed corr_protein upper triangle."""
    d = R / "bench_store" / store
    P = pd.read_parquet(d / "peaks.parquet")
    env = np.array([np.asarray(e, float) for e in P.env])
    Q = pd.DataFrame({"mz": P.feature_mz, "iso_r": P.isotope_image_colocalization_m1, "env0": env[:, 0],
                      "env1": env[:, 1], "env2": env[:, 2], "p90": P.intensity_p90, "isum": P.intensity_sum,
                      "mm1_r": P.isofit_mm1_r, "cen": P.peakfit_cen25_hipix_ppm})
    vm = np.load(d / "corr_protein_mzs.npy")
    ix = pd.Series(np.arange(len(vm)), index=vm.astype(float)).reindex(Q.mz.to_numpy(float))
    Q["valid"] = ix.notna().to_numpy()
    idx = ix.fillna(-1).to_numpy(np.int64)
    tri = np.load(d / "corr_protein_tri.npy", mmap_mode="r")
    n = len(vm)

    def pair_r(i, j):
        a, b = np.minimum(idx[i], idx[j]), np.maximum(idx[i], idx[j])
        return np.asarray(tri[a * n - a * (a - 1) // 2 + b - a], np.float64)
    return Q, pair_r


def check_rows(c, ref):
    """The package's candidate set equals the bench's (same rows, same order)."""
    assert len(c) == len(ref), (len(c), len(ref))
    for col in ("peptide", "protein", "is_decoy", "parent_peptide", "feature_mz"):
        a, b = c[col].to_numpy(), ref[col].to_numpy()
        assert ((a == b) | (pd.isna(a) & pd.isna(b))).all(), col
    assert (c.seed.to_numpy() == ref.seed.to_numpy())[ref.is_decoy.to_numpy()].all(), "seed"


def compare(rows, pt, prt, ref, rpt, rprt):
    """Per-term max |diff| and correlation against the bench dump; asserts the ID sets and protein counts."""
    out = {}
    for t, b in BENCH_COLS.items():
        x, y = rows[t].to_numpy(float), ref[b].to_numpy(float)
        out[t] = (float(np.nanmax(np.abs(x - y))), float(np.corrcoef(np.nan_to_num(x), np.nan_to_num(y))[0, 1]))
    out["s"] = (float(np.max(np.abs(rows.s.to_numpy() - ref.s.to_numpy()))), 1.0)
    for a in (.01, .05, .1):
        got = set(pt.index[(pt.q <= a) & ~pt.is_decoy])
        exp = set(rpt.index[(rpt.q <= a) & ~rpt.is_decoy])
        out[f"ids{int(a * 100)}"] = (len(got), len(exp), len(got ^ exp))
        out[f"pids{int(a * 100)}"] = (ids(prt, a, "q_picked"), ids(rprt, a, "q_picked"))
    return out


def layer_a(ds):
    psm, reader, store, shift, tids, _ = DS[ds]
    t = read_targets(str(R / psm), reader)
    Q, pair_r = store_evidence(store)
    for method, tid in zip(("d1b", "d1c_raw"), tids):
        t0 = time.time()
        c = world(t, SEEDS, method, shift, (Q.mz.min(), Q.mz.max()))
        ref = pd.read_parquet(REF / f"{tid}_rep0_rows.parquet")
        check_rows(c, ref)
        c["qi"] = np.searchsorted(Q.mz.to_numpy(), c.feature_mz.to_numpy())
        assert (Q.mz.to_numpy()[c.qi] == c.feature_mz.to_numpy()).all()
        rows, pt, prt = score_run(c, Q, pair_r, 0.0)  # the store's centroid column already has the offset removed
        o = compare(rows, pt, prt, ref, pd.read_parquet(REF / f"{tid}_rep0_pep.parquet"),
                    pd.read_parquet(REF / f"{tid}_rep0_prot.parquet"))
        print(f"A {ds} {method}: {len(c)} rows = bench; " + "; ".join(f"{k} {v}" for k, v in o.items())
              + f" [{time.time() - t0:.0f} s]", flush=True)
        assert all(o[t][0] == 0 for t in (*BENCH_COLS, "s")), "terms differ"
        assert all(o[f"ids{a}"][2] == 0 for a in (1, 5, 10)), "ID sets differ"
        assert all(o[f"pids{a}"][0] == o[f"pids{a}"][1] for a in (1, 5, 10)), "protein counts differ"


MASK_TICQ_TICN = {"kidney": (0.25, False), "amyloidosis": (0.5, True), "her2": (0.25, False)}  # bench configs (E031)
D = {"kidney": "data/kidney/Kidney_MS1_ITO6_kidneyonly.d", "amyloidosis": "data/amyloidosis/Amy_TMA_MS1.d",
     "her2": "data/her2/ucsd_Upload_PDX_MS1/PDX_MS1_852.d"}


def bench_config(ds, method, out):
    """The package config that reproduces bench world rep 0: bench seeds, mask, m/z range and frozen offset."""
    from msi_picasso.cli import load_config
    psm, reader, store, shift, _, off = DS[ds]
    P = pd.read_parquet(R / "bench_store" / store / "peaks.parquet", columns=["feature_mz"]).feature_mz
    return load_config(None, maldi_d=str(R / D[ds]), lcms_peptides=str(R / psm), psm_utils_reader=reader,
                       output_dir=str(out), decoy_method=method, decoy_seeds=SEEDS, decoy_min_shift_da=shift,
                       tic_normalize=MASK_TICQ_TICN[ds][1], tic_quantile=MASK_TICQ_TICN[ds][0], mz_offset_ppm=off,
                       tissue_mask=str(R / "bench_store" / f"{ds}_minreg1" / "pixel_mask.npy"),
                       mz_range=[float(P.min()), float(P.max())], gt_peptides=str(R / "data" / ds / "GT_peptides.txt"))


def layer_b(ds, method, out):
    """Full package run (own extraction) against the bench world: per term max |diff| and r, IDs and their
    differences, peptide by peptide."""
    from msi_picasso.cli import run
    tid = DS[ds][4][method == "d1c_raw"]
    rows, pt, prt, Q, info = run(bench_config(ds, method, out))
    ref = pd.read_parquet(REF / f"{tid}_rep0_rows.parquet")
    check_rows(rows, ref)
    rpt = pd.read_parquet(REF / f"{tid}_rep0_pep.parquet")
    o = compare(rows, pt, prt, ref, rpt, pd.read_parquet(REF / f"{tid}_rep0_prot.parquet"))
    print(f"B {ds} {method}: " + "; ".join(f"{k} {v}" for k, v in o.items()), flush=True)
    for a in (.01, .05, .1):
        got, exp = set(pt.index[(pt.q <= a) & ~pt.is_decoy]), set(rpt.index[(rpt.q <= a) & ~rpt.is_decoy])
        for p in sorted(got ^ exp):
            print(f"  {a:g}: {p} {'package only' if p in got else 'bench only'}: q {pt.q.get(p, np.nan):.4f} vs "
                  f"{rpt.q.get(p, np.nan):.4f}, s {pt.s.get(p, np.nan):.4f} vs {rpt.s.get(p, np.nan):.4f}")
    return o, info


if __name__ == "__main__":
    if sys.argv[1] == "A":
        for ds in sys.argv[2:] or DS:
            layer_a(ds)
        print("layer A: exact")
    elif sys.argv[1] == "B":  # B <ds> <d1b|d1c_raw> <output dir>
        layer_b(*sys.argv[2:5])
