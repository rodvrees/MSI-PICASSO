"""The adopted score: five evidence terms, M1, per-ion competition, q-values, proteins.

Every term is read at the candidate's own query m/z from the evidence table Q (one row per query m/z, extract.py),
so a decoy gets its evidence from the same code as a target. 

Q columns: mz (sorted), iso_r (r of the M and M+1 images), env0..env2 (M, M+1, M+2 envelope), p90 and isum (the
M image's 90th percentile of non-zero pixels and its sum), mm1_r (r of the M-1 and M images), cen (signed
top-decile-pixel m/z centroid within +-25 ppm, ppm). pair_r(i, j) -> Pearson r of the M images of Q rows i and j
(NaN-free; only called for valid rows, Q.valid).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from msi_picasso.candidates import PROTON, averagine

TERMS = ["coloc", "iso", "kl", "mm1", "flank"]
SIGNS = np.array([1.0, 1.0, -1.0, -1.0, -1.0])
# profile terms (bench 2, 2026-11-21), used when Q carries profile_features' columns: name -> signed weight
EXTRA = {"crat": -0.5, "m1c": -0.25, "m1g": -0.1}
MM1_HINGE, FLANK_HINGE = 0.5, 2.0  # one-sided terms: max(0, x - hinge)
DGRAD_HINGE, M1C_HINGE, M1G_HINGE = 1.0, 0.5, 3.0
SLOPE_PPM, CRAT_PPM = 6.0, 2.5  # log-slope fit window, centre window of the ratio terms (ppm from the offset)
SHRINK = 4.0  # coloc shrunk to the median: (n x + 4 median) / (n + 4), n = partners
SELFWIT_PPM = 20.0  # partners within 2 x the +-10 ppm window read the same TOF peak: dropped
ION_PPM, ION_TAU = 0.05, 1.0  # per-ion competition window and soft penalty
OFFSET_Z, OFFSET_MIN_PPM = 6.0, 1.0  # the m/z offset is used only when this clear and this large


@lru_cache(maxsize=None)
def _dist3(c, h, n, o):
    from brainpy import isotopic_variants
    p = np.array([x.intensity for x in isotopic_variants(dict(zip("CHNOS", (c, h, n, o, 0))), npeaks=3, charge=0)])
    p = np.pad(p, (0, 3 - len(p)))[:3]
    return p / p.sum() if p.sum() >= 1e-12 else np.zeros(3)


def isotope_dist3(c, h, n, o):
    """(n, 3) sum-to-1 [M, M+1, M+2] of each composition (brainpy, S = 0)."""
    return np.array([_dist3(*k) for k in zip(c.tolist(), h.tolist(), n.tolist(), o.tolist())]).reshape(-1, 3)


def averagine_dist3(mz):
    """(n, 3) [M, M+1, M+2] of averagine at each [M+H]+ m/z (charge 1, no sulfur)."""
    a = averagine(np.asarray(mz) - PROTON)
    return isotope_dist3(a["C"], a["H"], a["N"], a["O"])


def averagine_kl(mz, env):
    """KL(observed envelope || averagine at mz - proton), both sum-to-1 and clipped at 1e-10; 0 without M signal."""
    ref = averagine_dist3(mz)
    s = env.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.clip(env / s[:, None], 1e-10, None)
        t = o * np.log(o / np.clip(ref, 1e-10, None))
        return np.where(s > 0, t[:, 0] + t[:, 1] + t[:, 2], 0.0)


def _filled(x):
    x = pd.Series(x)
    return x.fillna(x.median()).to_numpy(float)


def _z(X):
    """Column z-scores (population sd) after the per-column NaN median fill."""
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(med), 0.0, med), X)
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)


def partner_pairs(c, valid):
    """(feature_mz, protein) -> partner m/z pairs. A target pairs with the other targets of its protein; a decoy of
    protein X pairs with the targets of X except its parent. Partners within SELFWIT_PPM of the row are dropped."""
    base = c[["feature_mz", "protein", "peptide", "parent_peptide", "qi"]][valid[c.qi.to_numpy()]]
    is_d = base.protein.astype(str).str.startswith("DECOY_")
    tgt = base[~is_d].drop_duplicates(["feature_mz", "protein"])
    dec = base[is_d].drop_duplicates(["feature_mz", "protein"]).copy()
    dec["key"] = dec.protein.astype(str).str.removeprefix("DECOY_")
    dec["parent_mz"] = dec.parent_peptide.map(c[~c.is_decoy].set_index("peptide").feature_mz)
    tp = tgt.rename(columns={"feature_mz": "partner_mz", "qi": "pqi", "protein": "key", "peptide": "partner_peptide"})[
        ["key", "partner_mz", "pqi", "partner_peptide"]]
    dp = dec.merge(tp, on="key")
    # the parent's ion, not only the parent's name: an isobaric sibling target (I/L) carries the parent's image
    # (2026-11-17, RD; the bench excluded by name only)
    dp = dp[dp.partner_mz != dp.parent_mz]
    tt = tgt.merge(tgt.rename(columns={"feature_mz": "partner_mz", "qi": "pqi"})[["protein", "partner_mz", "pqi"]],
                   on="protein")
    cols = ["feature_mz", "protein", "qi", "partner_mz", "pqi"]
    p = pd.concat([tt[cols], dp[cols]], ignore_index=True)
    p = p[p.feature_mz != p.partner_mz]
    return p[np.abs(p.partner_mz - p.feature_mz) / p.feature_mz * 1e6 >= SELFWIT_PPM].copy()


def _log_slope(P, x, c):
    """Per row: slope of log(P / row max) against x - c over the sampled bins (P > 0) within SLOPE_PPM of c (NaN
    with fewer than 3)."""
    pn = P / np.maximum(P.max(1, keepdims=True), 1e-12)
    sel = np.abs(x - c) <= SLOPE_PPM
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        k = sel & (P[i] > 0)
        if k.sum() >= 3:
            out[i] = np.polyfit(x[k] - c, np.log(pn[i][k]), 1)[0]
    return out


def _interp(P, x, grid):
    """Per row: P linearly interpolated over its sampled bins (P > 0) at grid (NaN with fewer than 5)."""
    out = np.full((len(P), len(grid)), np.nan)
    for i in range(len(P)):
        k = P[i] > 0
        if k.sum() >= 5:
            out[i] = np.interp(grid, x[k], P[i, k])
    return out


def run_sigma(pn, iso):
    """Run peak width (ppm, Gaussian sigma) of the +-25 ppm M profiles, label-free: per profile of the 200 highest-iso
    queries, the sigma (5..20 ppm) whose best fit A G(x - mu; s) + a noise(x) + b (A >= 0, mu on a 0.25 ppm grid
    over +-35, sampled bins only) has the least squared error; the median. noise = the median normalised profile of
    the queries below the 30% iso quantile (box + background slope)."""
    x, mus = np.arange(-25, 26, dtype=float), np.arange(-35, 35.01, 0.25)
    m = (pn > 0).astype(float)
    lo = np.nan_to_num(iso, nan=0) < np.nanquantile(iso, 0.3)
    noise = np.nanmedian(np.where(m > 0, pn, np.nan)[lo], 0)
    noise = np.where(np.isfinite(noise), noise, np.nanmedian(noise))
    top = np.argsort(-np.nan_to_num(iso, nan=-1))[:200]
    pn, m = pn[top], m[top]
    yy = (m * pn ** 2).sum(1)
    sig = np.arange(5, 21, 1.0)
    best = []
    for s in sig:
        sse = np.full((len(pn), len(mus)), np.inf)
        for j, mu in enumerate(mus):
            D = np.stack([np.exp(-0.5 * ((x - mu) / s) ** 2), np.ones_like(x), noise], 1)
            M = np.einsum("qi,ij,ik->qjk", m, D, D) + 1e-9 * np.eye(3)
            r = np.einsum("qi,ij->qj", m * pn, D)
            b = np.linalg.solve(M, r[..., None])[..., 0]
            ok = b[:, 0] >= 0
            sse[ok, j] = (yy - (b * r).sum(1))[ok]
        best.append(sse.min(1))
    return float(np.median(sig[np.argmin(best, 0)]))


def profile_features(Q, prof2, offset):
    """Q plus four per-query profile features (bench 2, 2026-11-21), label-free, from the top-decile-pixel M-1 and M
    m/z profiles prof2 (n_q, 2, 121 bins, -60..60 ppm) and run-level constants only:
      dgrad  log-slope of the M profile at the offset (bins within 6 ppm) x sigma_run^2, minus its median over queries:
             how far the peak apex sits from the query (ppm; 0 for a centred peak). Replaces the centroid in flank.
      crat   best M sample within 2.5 ppm of the offset / the +-25 ppm profile max (1: the peak is centred on the query)
      m1c    best M-1 sample within 2.5 ppm of the exact M-1 position / the M-1 profile max within 60 ppm (1: an ion's
             apex sits at M-1, so the query reads the M+1 isotope of a lighter ion)
      m1g    |M-1 log-slope x sigma_run^2 at the run's M-1 apex shift - its median| (small: an M-1 apex where expected)
    sigma_run: run_sigma; M-1 apex shift: the shift (-15..15 ppm, 0.5 steps) of the M-1 profile that best matches the
    M profile shape (mean r over the 10% highest-iso queries, +-20 ppm). Returns (Q, {sigma_run, m1_shift} in ppm)."""
    x60 = np.arange(-60, 61, dtype=float)
    P1, P = prof2[:, 0].astype(float), prof2[:, 1].astype(float)
    p = P[:, 35:86]  # M, -25..25 ppm
    x = x60[35:86]
    pn = p / np.maximum(p.max(1, keepdims=True), 1e-12)
    iso = Q.iso_r.to_numpy()
    s = run_sigma(pn, iso)
    Q = Q.copy()
    dg = _log_slope(p, x, offset) * s ** 2
    Q["dgrad"] = dg - np.nanmedian(dg)
    cm = np.abs(x - offset) <= CRAT_PPM
    Q["crat"] = np.where((p[:, cm] > 0).any(1), pn[:, cm].max(1), np.nan)
    cm, lm = np.abs(x60 - offset) <= CRAT_PPM, np.abs(x60 - offset) <= 60
    Q["m1c"] = np.where((P1[:, cm] > 0).any(1), P1[:, cm].max(1) / np.maximum(P1[:, lm].max(1), 1e-12), np.nan)
    # run-level M-1 apex shift, label-free (the high-iso queries)
    good = np.nan_to_num(iso, nan=-1) >= np.quantile(np.nan_to_num(iso, nan=-1), 0.9)
    g = np.arange(-20, 20.01, 0.5)
    A = _interp(P[good], x60, g)
    A = A - np.nanmean(A, 1, keepdims=True)
    best, s1 = -np.inf, 0.0
    for sh in np.arange(-15, 15.01, 0.5):
        Y = _interp(P1[good], x60, g + sh)
        Y = Y - np.nanmean(Y, 1, keepdims=True)
        v = np.nanmean((A * Y).sum(1) / np.sqrt((A * A).sum(1) * (Y * Y).sum(1)))
        if v > best:
            best, s1 = v, sh
    m1g = _log_slope(P1, x60, offset + s1) * s ** 2
    Q["m1g"] = np.abs(m1g - np.nanmedian(m1g))
    return Q, dict(sigma_run=float(s), m1_shift=float(s1))


def terms(c, Q, pair_r, offset=0.0):
    """The term columns (raw values, before z-scoring) of candidates c (with qi = their Q row): TERMS, then the EXTRA
    profile terms when Q has profile_features' columns. Per-run (label-blind) fills: store NaNs -> the median over all
    rows; a no-signal row without coloc -> the pooled minimum coloc. flank: max(0, |dgrad| - 1) when Q has dgrad,
    else the centroid form (|centroid - offset| ppm, hinge 2; mz_offset needs it, before the profile features).
    Bench 2 coloc (isogate1): pass extract.slot_r_fn as pair_r."""
    q = c.qi.to_numpy()
    d = pd.DataFrame(index=c.index)
    bench2 = "dgrad" in Q
    d["iso"] = _filled(Q.iso_r.to_numpy()[q])
    d["kl"] = averagine_kl(Q.mz.to_numpy(), Q[["env0", "env1", "env2"]].to_numpy(float))[q]  # per query m/z
    d["mm1"] = np.maximum(0.0, _filled(Q.mm1_r.to_numpy()[q]) - MM1_HINGE)
    if bench2:
        d["flank"] = _filled(np.maximum(0.0, np.abs(Q.dgrad.to_numpy()) - DGRAD_HINGE)[q])
    else:
        d["flank"] = np.maximum(0.0, _filled(np.abs(Q.cen.to_numpy()[q] - offset)) - FLANK_HINGE)
    # coloc: partner pairs weighted by sqrt(p90_a p90_b) x the partner's evidence (percentile rank of the other four
    # terms' M1 sum over all rows, mean per m/z)
    sub = _z(d[["iso", "kl", "mm1", "flank"]].to_numpy(float)) @ SIGNS[1:]
    ev = pd.Series(sub).rank(pct=True).groupby(c.feature_mz.to_numpy()).mean()
    pairs = partner_pairs(c, Q.valid.to_numpy())
    pairs["r"] = pair_r(pairs.qi.to_numpy(), pairs.pqi.to_numpy())
    p90 = Q.p90.to_numpy(float)
    w = np.sqrt(np.nan_to_num(p90[pairs.qi.to_numpy()]).clip(0) * np.nan_to_num(p90[pairs.pqi.to_numpy()]).clip(0))
    w = w * np.nan_to_num(pairs.partner_mz.map(ev).to_numpy(float))
    key = ["feature_mz", "protein"]
    g = pairs.assign(w=w, wr=w * pairs.r).groupby(key)
    agg = pd.DataFrame({"x": g.wr.sum() / g.w.sum().where(g.w.sum() > 0), "n": g.size()}).reset_index()
    m = c[key].merge(agg, on=key, how="left")
    x = pd.Series(m.x.to_numpy(float), c.index)
    sig = Q.isum.to_numpy(float)[q] > 0
    if (~sig).any() and x.notna().any():
        x = x.where(sig | x.notna(), x.min())
    n = pd.Series(m.n.fillna(0).to_numpy(float), c.index)
    d["coloc"] = x.where(n < 1, (n * x + SHRINK * x.median()) / (n + SHRINK))
    d["n_partners"] = n
    extra = []
    if bench2:
        d["crat"] = _filled((1.0 - Q.crat.to_numpy())[q])
        d["m1c"] = _filled(np.maximum(0.0, Q.m1c.to_numpy() - M1C_HINGE)[q])
        d["m1g"] = _filled(np.maximum(0.0, M1G_HINGE - Q.m1g.to_numpy())[q])
        extra = list(EXTRA)
    return d[TERMS + extra + ["n_partners"]]


def term_weights(cols):
    """(names, weights) of the score terms present in cols: the five TERMS (+, +, -, -, -), then the EXTRA terms."""
    names = TERMS + [k for k in EXTRA if k in cols]
    return names, np.r_[SIGNS, [EXTRA[k] for k in names[len(TERMS):]]]


def m1(T):
    """Fixed weighted z-sum of the terms present in T (term_weights)."""
    names, w = term_weights(T)
    return _z(T[names].to_numpy(float)) @ w


def ion_competition(s, dec, mz):
    """(removed, penalised score). A row leaves when a target with a higher score reads its ion (within ION_PPM);
    decoys never remove. Survivors pay ION_TAU log(1 + sum_j exp((s_j - s_i) / ION_TAU)) over the other targets j on
    their ion (computed on the scores before removal)."""
    t = np.flatnonzero(~dec)
    t = t[np.argsort(mz[t])]  # the targets by query m/z; lo:hi = those within ION_PPM of each row
    lo = np.searchsorted(mz[t], mz * (1 - ION_PPM * 1e-6))
    hi = np.searchsorted(mz[t], mz * (1 + ION_PPM * 1e-6), "right")
    ts = s[t]
    best = np.array([ts[a:b].max() if b > a else -np.inf for a, b in zip(lo, hi)])
    pen = np.zeros(len(s))
    for i in np.flatnonzero(hi - lo > ~dec):  # rows with a target on their ion other than themselves
        j = t[lo[i]:hi[i]]
        x = s[j[j != i]]
        if x.size:
            pen[i] = ION_TAU * np.logaddexp.reduce(np.r_[0.0, (x - s[i]) / ION_TAU])
    return best > s, s - pen


def nc_qvalues(s, dec, k=1.0):
    """Non-competitive TDC: q = monotone (D(>=t) + 1) / (k T(>=t)); tied scores share the counts at the end of
    their tie block. k = decoys per target."""
    o = np.argsort(-s, kind="stable")
    ss, dd = s[o], dec[o]
    last = np.searchsorted(-ss, -ss, side="right") - 1
    D, T = np.cumsum(dd)[last], np.cumsum(~dd)[last]
    q = np.minimum.accumulate(((D + 1) / (k * np.maximum(T, 1)))[::-1])[::-1]
    out = np.empty(len(s))
    out[o] = q
    return out


def peptide_table(c, s, k):
    """Per peptide: best score over its rows and q (non-competitive, k decoys per target)."""
    best = pd.DataFrame({"peptide": c.peptide, "s": s, "is_decoy": c.is_decoy, "qi": c.qi, "protein": c.protein,
                         "seed": c.seed}).groupby("peptide").agg(
        s=("s", "max"), is_decoy=("is_decoy", "first"), qi=("qi", "first"), protein=("protein", "first"),
        seed=("seed", "first"))
    best["q"] = nc_qvalues(best.s.to_numpy(float), best.is_decoy.to_numpy(bool), k)
    return best


def protein_table(pt):
    """Proteins at controlled FDR (secondary output). Score = best peptide. A decoy protein = the decoys of one target
    protein from one decoy seed. q: non-competitive (D + 1) / (k T) over proteins; q_picked: picked protein FDR
    (Savitski et al. 2015), a target protein competes with its own decoy proteins and only the best counts."""
    key = pt.protein.astype(str) + np.where(pt.is_decoy, "|" + pt.seed.astype(str), "")
    t = pt.assign(key=key.to_numpy()).sort_values("s", ascending=False)
    pr = t.groupby("key").agg(s=("s", "first"), is_decoy=("is_decoy", "first"), qi=("qi", "first"),
                              protein=("protein", "first"))
    k = pr.is_decoy.sum() / max((~pr.is_decoy).sum(), 1)
    pr["q"] = nc_qvalues(pr.s.to_numpy(float), pr.is_decoy.to_numpy(bool), k)
    pr["grp"] = pr.protein.astype(str).str.removeprefix("DECOY_")
    pr["q_picked"] = _picked_qvalues(pr)
    return pr


def _picked_qvalues(best):
    """Max-competition TDC with K_g decoy proteins per group (Keich & Noble 2015): only the group's best protein
    competes (ties to the decoy); q = monotone (sum over decoy winners of 1 / K_g + 1 / mean K) / target winners.
    Decoy proteins whose best peptides share one ion count once. Losers and groups without decoys get q = 1."""
    full = best.index
    best = best[~(best.is_decoy & best.duplicated(["grp", "qi"]))]
    tg = best.index[~best.is_decoy]
    kg = best[best.is_decoy & best.grp.isin(tg)].groupby("grp").size()
    w = best[best.grp.isin(kg.index)].assign(d=lambda x: x.is_decoy.astype(int))
    w = w.sort_values(["s", "d"], ascending=False, kind="stable")
    w = w[~w.grp.duplicated()]
    s, dec = w.s.to_numpy(float), w.is_decoy.to_numpy(bool)
    last = np.searchsorted(-s, -s, side="right") - 1
    D = np.cumsum(np.where(dec, 1.0 / w.grp.map(kg).to_numpy(float), 0.0))[last]
    T = np.cumsum(~dec)[last]
    q = np.minimum.accumulate(((D + 1.0 / kg.mean()) / np.maximum(T, 1))[::-1])[::-1]
    out = pd.Series(1.0, index=full)
    out[w.index] = q
    return out


def score_run(c, Q, pair_r, offset=0.0):
    """The adopted score of one run: (rows with terms, score s0 and final s and the `removed` flag, peptide table,
    protein table). k = decoy / target peptides before the per-ion competition. Q with profile_features' columns and
    pair_r = extract.slot_r_fn give the bench 2 score; without them, the bench 1 score."""
    T = terms(c, Q, pair_r, offset)
    s0 = m1(T)
    dec, mz = c.is_decoy.to_numpy(bool), c.feature_mz.to_numpy(float)
    pk = c.groupby("peptide").is_decoy.first()
    k = pk.sum() / (~pk).sum()
    gone, s = ion_competition(s0, dec, mz)
    rows = pd.concat([c, T], axis=1).assign(s0=s0, s=s, removed=gone)
    pt = peptide_table(rows[~gone].reset_index(drop=True), s[~gone], k)
    return rows, pt, protein_table(pt)


def ids(pt, a, q="q"):
    return int(((pt[q] <= a) & ~pt.is_decoy).sum())


def mz_offset(c, Q, pair_r):
    """Label-free global m/z offset (ppm) for the flank term (BENCH "Label-free m/z offset"): the mode of the signed
    centroid under a +-0.5 ppm box, over the top 1% of all rows (targets and decoys pooled) by the M1 sum of the other
    four terms; accepted only if its excess over the 1.5-2.5 ppm shoulders has z > OFFSET_Z and |offset| >= OFFSET_MIN_PPM."""
    T = terms(c, Q, pair_r, 0.0)
    s4 = _z(T[TERMS[:4]].to_numpy(float)) @ SIGNS[:4]
    x = Q.cen.to_numpy()[c.qi.to_numpy()][s4 >= np.quantile(s4, 0.99)]
    x = x[np.isfinite(x)]
    o = _box_mode(x, 0.5)
    n, b = (np.abs(x - o) < 0.5).sum(), ((np.abs(x - o) >= 1.5) & (np.abs(x - o) < 2.5)).sum() * 0.5
    z = (n - b) / np.sqrt(max(b, 1))
    return (o if z > OFFSET_Z and abs(o) >= OFFSET_MIN_PPM else 0.0), o, z


def _box_mode(s, w):
    grid = np.arange(-5, 5.001, 0.05)
    s = np.sort(s[np.abs(s) < 10])
    n = np.searchsorted(s, grid + w, "right") - np.searchsorted(s, grid - w)
    o = float(np.mean(grid[n == n.max()]))
    for _ in range(20):  # mean shift: the median inside the window
        o = float(np.median(s[np.abs(s - o) < w]))
    return o


if __name__ == "__main__":  # self-checks
    s, dec, mz = np.array([0.5, 0.9, 0.3, 0.95]), np.array([True, False, False, True]), np.full(4, 1000.0)
    gone, _ = ion_competition(s, dec, mz)
    assert list(gone) == [True, False, True, False]  # target 0.9 removes the rows below it; decoys never remove
    assert np.allclose(nc_qvalues(np.array([3.0, 2.0, 1.0]), np.array([False, True, False]), 1.0), [1.0, 1.0, 1.0])
    assert abs(_box_mode(np.r_[np.random.default_rng(0).uniform(-10, 10, 20000), np.full(500, -1.5)], 2.0) + 1.5) < 0.3
    print("score ok")
