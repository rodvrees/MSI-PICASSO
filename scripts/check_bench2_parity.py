"""Bench 2 parity: the package's score reproduces the bench 2 reference (gates `ref` = x:r2_profile:m1cg) per draw.

    python scripts/check_bench2_parity.py A|B [<ds> ...]      (run from /tmp; bench2_store and bench2 results needed)

A: package score (profile_features, slot_gate, score_run) on the bench's cached evidence: Q, X, the bench's M-1 / M
   profiles (r2prof.npz), its mobility components (comp/: slots, slot iso, component rows C).
B: package extraction (extract.extract on the cache's query m/z) against the same caches, then the package score on
   the package's own evidence.
Both score the 20 real draws (bench2_store/<ds>_<tag>/draws.parquet, offset from draw 0 as gates.py) and compare
IDs at 1 / 5 / 10%, the 5% cut score and the 5% ID set with bench2/results/worlds/<ds>_ref_<tag>.tsv.
"""
import json
import sys

import numpy as np
import pandas as pd

from msi_picasso import score as S
from msi_picasso.extract import extract, pair_r_fn, slot_gate, slot_r_fn

ROOT = "/home/robbe/MALDI_MSI_score"
TAGS = {"amyloidosis": "sO", "kidney": "sS", "her2": "sHM"}


def bench_features(ds, st):
    """The bench's per-query features: flankx (dgrad, crat) and r2_profile (m1c, m1g)."""
    f = np.load(f"{st}/flankx_feat.npz")
    m = pd.read_parquet(f"{ROOT}/bench2/r2_profile/out/{ds}_m1c.parquet")
    g = m["m1g_s"].to_numpy()
    return dict(dgrad=f["dgrad"], crat=f["crat"], m1c=m["m1c_o_2.5_60"].to_numpy(), m1g=np.abs(g - np.nanmedian(g)))


def diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    nan = (np.isnan(a) != np.isnan(b)).sum()
    ok = ~np.isnan(a) & ~np.isnan(b)
    return f"max|diff| {np.abs(a[ok] - b[ok]).max():.2e}, NaN mismatches {nan}"


def score_draws(st, Q, X):
    d = pd.read_parquet(f"{st}/draws.parquet")
    d["qi"] = np.searchsorted(Q.mz.to_numpy(), d.feature_mz.to_numpy())
    draws = [x.drop(columns="draw").reset_index(drop=True) for _, x in d.groupby("draw")]
    cfg = json.load(open(f"{st}/cfg.json"))
    pr = pair_r_fn(X, Q.valid.to_numpy())
    o = S.mz_offset(draws[0], Q, pr)[0] if cfg["mz_offset_ppm"] == "auto" else float(cfg["mz_offset_ppm"])
    return o, draws, cfg


def compare(ds, tag, Qf, draws, o, slot_r):
    ref = pd.read_csv(f"{ROOT}/bench2/results/worlds/{ds}_ref_{tag}.tsv", sep="\t")
    ref = ref[ref.world == "real"].set_index("k")
    bad = 0
    for r, c in enumerate(draws):
        rows, pt, prt = S.score_run(c, Qf, slot_r, o)
        tg = pt[~pt.is_decoy]
        ids = [S.ids(pt, a) for a in (.01, .05, .1)]
        cut = float(tg.s[tg.q <= .05].min()) if (tg.q <= .05).any() else np.nan
        same = (ids == [ref.ids1[r], ref.ids5[r], ref.ids10[r]] and
                ",".join(sorted(tg.index[tg.q <= .05])) == (ref.id5_set[r] if isinstance(ref.id5_set[r], str) else ""))
        dc = abs(cut - ref["cut_0.05"][r]) if np.isfinite(cut) else 0.0
        bad += not same
        print(f"  draw {r:2d}: IDs {ids} ref {[int(ref.ids1[r]), int(ref.ids5[r]), int(ref.ids10[r])]}, "
              f"5% set {'same' if same else 'DIFFERENT'}, |cut diff| {dc:.1e}", flush=True)
    print(f"{ds}: {20 - bad}/20 draws identical", flush=True)


def main(mode, dss):
    for ds in dss:
        tag = TAGS[ds]
        st = f"{ROOT}/bench2_store/{ds}_{tag}"
        Q = pd.read_parquet(f"{st}/Q.parquet")
        X = np.load(f"{st}/X.npy")
        slots, siso = np.load(f"{st}/comp/slots.npy"), np.load(f"{st}/comp/iso.npy")
        sw_b, best_b = slot_gate(slots, siso, Q.iso_r.to_numpy())
        P_b = np.load(f"{st}/r2prof.npz")["P"][:, :2]
        o, draws, cfg = score_draws(st, Q, X)
        C = np.load(f"{st}/comp/C.npy", mmap_mode="r")
        print(f"{ds}: offset {o:+.4f} ppm, {int(sw_b.sum())} switched queries (bench comp/)", flush=True)
        if mode == "A":
            qs = np.flatnonzero(sw_b)
            XS = np.stack([C[q, best_b[q]] for q in qs])
            Qf = S.profile_features(Q.assign(mob_switch=sw_b), P_b, o)[0]
        else:
            mask = np.load(cfg["tissue_mask"]) if cfg.get("tissue_mask") else None
            Qp, Xp, img, P, layout, mob = extract(cfg["maldi_d"], Q.mz.to_numpy(), cfg["extraction_ppm"],
                                                 cfg["tic_quantile"], cfg["tic_normalize"], mask, 64, print)
            cols = [k for k in Q.columns if k in Qp]
            print("  Q columns", "identical" if Qp[cols].equals(Q[cols]) else
                  {k: diff(Qp[k], Q[k]) for k in cols if not Qp[k].equals(Q[k]) and k != "valid"})
            XS = mob["XS"]
            print("  X", diff(Xp, X), "| M-1 / M profiles vs r2prof:", diff(P.astype(np.float32), P_b))
            sw = Qp.mob_switch.to_numpy()
            print(f"  switched: package {int(sw.sum())}, bench {int(sw_b.sum())}, differ {int((sw != sw_b).sum())}")
            both = np.flatnonzero(sw & sw_b)
            srow = np.cumsum(sw) - 1
            print("  slot rows vs bench C:", diff(XS[srow[both]], np.stack([C[q, best_b[q]] for q in both])))
            Q, X, P_b = Qp, Xp, P
            Qf = S.profile_features(Q, P, o)[0]
        B = bench_features(ds, st)
        for k, v in B.items():
            print(f"  {k}: {diff(Qf[k], v)}")
        compare(ds, tag, Qf, draws, o, slot_r_fn(XS, Qf.mob_switch.to_numpy(), X, Qf.valid.to_numpy()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(TAGS))
