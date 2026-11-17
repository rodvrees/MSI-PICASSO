"""Targets and decoys (BENCH "Bench closed": the adopted candidate set).

Targets: the LC-MS/MS peptides of the PSM file plus their LC-observed HyP forms ('<seq>+h<n>').
Decoys: residue-substitution decoys of each target, chosen on sequence-only properties (D1C_RAW, or D1b), one
decoy set per seed; K seeds per run. HyP decoys are the base decoy plus the same n O.
Nothing here reads MALDI data, and no decoy is chosen with any MALDI evidence.
"""
import hashlib

import numpy as np
import pandas as pd
from pyteomics import mass as pmass

PROTON = 1.007276
NEUTRON = 1.003355
O_MASS = 15.994915
ALPHABET = list("ADEFGHILNPQSTVWY")  # substitutes: no K/R (cleavage), no C/M (sulfur neither in nor out)
PEPTIDE_FDR = 0.01  # PSM files with q-values: peptides at q <= this
FLOOR_PPM = 10.0  # minimum decoy mass shift without decoy_min_shift_da
EXCLUDE_PPM = 20.0  # parent-family exclusion window
# decoy method -> (match keys in priority order, variants per target); the closest mass-defect residual breaks ties
DECOYS = {"d1c_raw": (["arom", "pro", "acid", "basic"], 100), "d1b": (["arom"], 30)}
# Residue masses at 5 decimals, summed from water in sequence order: the targets' masses as the bench had them (the
# earlier package's Rust digest). They are up to 1.7e-4 Da off the exact masses (decoys use exact pyteomics masses).
# ponytail: kept for bench parity; switch to pmass.fast_mass once RD decides (open question of the port).
RESIDUE = dict(G=57.02146, A=71.03711, V=99.06841, L=113.08406, I=113.08406, P=97.05276, F=147.06841, W=186.07931,
               M=131.04049, S=87.03203, T=101.04768, C=103.00919, Y=163.06333, H=137.05891, D=115.02694, E=129.04259,
               N=114.04293, Q=128.05858, K=128.09496, R=156.10111)
AVERAGINE = (("C", 0.04443), ("H", 0.06981), ("N", 0.01221), ("O", 0.01329), ("S", 0.00037))


def averagine(m):
    """Averagine atom counts {C, H, N, O, S} for an array of masses (np.round, half to even)."""
    m = np.asarray(m, np.float64)
    return {el: np.round(m * k).astype(int) for el, k in AVERAGINE}


def mass_defect_residual(m):
    """Mass defect of m minus the mass defect of averagine at m."""
    m = np.asarray(m, float)
    a = averagine(m)
    am = a["C"] * 12.0 + a["H"] * 1.00782503207 + a["N"] * 14.00307400480 + a["O"] * 15.99491461956
    return (m - np.floor(m)) - (am - np.floor(am))


def read_targets(psm_path, reader="fragpipe"):
    """Target rows (peptide, protein, mass, mh_mz, hyp_base) from a psm_utils-readable PSM file.

    Base targets: one row per unmodified sequence, sorted by sequence; protein = the first accession of the first PSM
    in q-value order (pandas' default sort, as the earlier package). HyP forms: each (sequence, n) peptidoform of the
    file with n prolines carrying +15.995 (hydroxyproline), as '<seq>+h<n>'."""
    from psm_utils.io import read_file
    rec = [(p.peptidoform.sequence, _accession(p.protein_list[0]) if p.protein_list else "",
            float(p.qvalue) if p.qvalue is not None else np.nan,
            sum(a == "P" and any(abs(m.mass - 15.9949) < 0.01 for m in mods or ()) for a, mods in p.peptidoform.parsed_sequence))
           for p in read_file(psm_path, filetype=reader) if p.peptidoform is not None and not p.is_decoy]
    df = pd.DataFrame(rec, columns=["sequence", "protein", "q", "n_hyp"])
    pep = df.sort_values("q").groupby("sequence").agg(protein=("protein", "first"), q=("q", "min")).reset_index()
    if pep.q.notna().any():  # files without q-values (FragPipe psm.tsv is already filtered) keep every peptide
        pep, df = pep[pep.q <= PEPTIDE_FDR], df[df.q <= PEPTIDE_FDR]  # HyP forms too: only from passing PSMs
    t = pd.DataFrame({"peptide": sorted(pep.sequence)})
    t["protein"] = t.peptide.map(pep.set_index("sequence").protein)
    t["mass"] = [_mass5(s) for s in t.peptide]
    t["hyp_base"], t["mh_mz"] = None, t.mass + PROTON
    forms = set(zip(df.sequence[df.n_hyp > 0], df.n_hyp[df.n_hyp > 0]))
    hyp = []
    for k in sorted({k for _, k in forms}):
        r = t[t.peptide.isin({q for q, j in forms if j == k})].copy()
        r["hyp_base"], r["peptide"] = r.peptide, r.peptide + f"+h{k}"
        r["mass"], r["mh_mz"] = r.mass + k * O_MASS, r.mh_mz + k * O_MASS  # added after the proton, as the bench
        hyp.append(r)
    return pd.concat([t, *hyp], ignore_index=True)


def _mass5(seq):
    m = 18.010565
    for a in seq:
        m += RESIDUE[a]
    return m


def _accession(a):
    """sp|P12345|GENE_HUMAN -> P12345 (also tr|, ref|); anything else up to the first space (contam_sp|... stays)."""
    p = str(a).strip().lstrip(">").split("|")
    return p[1] if len(p) >= 2 and p[0] in ("sp", "tr", "ref") else str(a).strip().lstrip(">").split()[0]


def decoys(targets, seed, method="d1c_raw", min_shift_da=None):
    """One decoy set: for each base target, n_variants random substitutions of 2 interior non-K/R/C/M residues
    (replacement drawn from the target residue frequency; a variant must not be a target and must shift the mass by at
    least min_shift_da, else FLOOR_PPM of the mass), keeping the variant closest to its target on the method's match
    keys, then on the mass-defect residual. Two targets never share a decoy. A HyP target's decoy is its base target's
    decoy plus the same n O. Returns decoy rows with parent_peptide (the target they were made from)."""
    h = targets.hyp_base.notna().to_numpy()
    base, H = targets[~h].reset_index(drop=True), targets[h]
    match, n_var = DECOYS[method]
    seqs = base.peptide.astype(str)
    counts = pd.Series(list("".join(seqs))).value_counts()
    w = np.array([counts.get(a, 0) for a in ALPHABET], float)
    w /= w.sum()
    sub = {}  # residue -> (replacements, their probabilities)
    for r in ALPHABET:
        choices = [a for a in ALPHABET if a != r]
        p = np.array([w[ALPHABET.index(a)] for a in choices])
        sub[r] = (choices, p / p.sum())
    target_set = set(seqs)
    var = []
    for i, pep in enumerate(seqs):
        rng = np.random.default_rng([seed, int.from_bytes(hashlib.md5(pep.encode()).digest()[:8], "little")])
        elig = [k for k in range(1, len(pep) - 1) if pep[k] not in "KRCM"]
        if len(elig) < 2:
            continue
        m0 = pmass.fast_mass(pep)
        floor = min_shift_da if min_shift_da is not None else FLOOR_PPM * (m0 + 1.00728) / 1e6  # 1.00728: the bench's proton
        for _ in range(n_var):
            s = list(pep)
            for k in rng.choice(elig, 2, replace=False):
                choices, p = sub[pep[k]]
                s[k] = rng.choice(choices, p=p)
            s = "".join(s)
            if s not in target_set and abs(pmass.fast_mass(s) - m0) >= floor:
                var.append((i, s))
    v = pd.DataFrame(var, columns=["src", "peptide"]).drop_duplicates()
    v["mass"] = [pmass.fast_mass(q) for q in v.peptide]
    v["d_res"] = np.abs(mass_defect_residual(v.mass) - mass_defect_residual(base.mass.to_numpy())[v.src])
    for key, pat in (("arom", "[FWYH]"), ("pro", "P"), ("acid", "[DE]"), ("basic", "H")):
        if key in match:
            v[f"d_{key}"] = (v.peptide.str.count(pat) - v.src.map(base.peptide.str.count(pat))).abs()
    best = v.sort_values(["src"] + [f"d_{k}" for k in match] + ["d_res"]).drop_duplicates("src")
    best = best.drop_duplicates("peptide")
    out = base.loc[best.src].reset_index(drop=True)
    out["parent_peptide"], out["peptide"] = out.peptide, best.peptide.to_numpy()
    out["protein"] = "DECOY_" + out.protein.astype(str)
    out["mass"] = best.mass.to_numpy()
    out["mh_mz"] = out.mass + PROTON
    H = H[H.hyp_base.isin(out.parent_peptide)]  # a base without a decoy gives its HyP forms none either
    d = out.set_index("parent_peptide").loc[H.hyp_base].reset_index()
    n = H.peptide.str.rsplit("+h", n=1).str[1].astype(int).to_numpy()
    d["peptide"] = d.peptide + pd.Series([f"+h{k}" for k in n])
    d["parent_peptide"], d["mass"], d["mh_mz"] = H.peptide.to_numpy(), d.mass + n * O_MASS, d.mh_mz + n * O_MASS
    out = pd.concat([out, d[out.columns]], ignore_index=True)
    out["is_decoy"], out["hyp_base"] = True, None
    return out


def world(targets, seeds, method="d1c_raw", min_shift_da=None, mz_range=(0, np.inf)):
    """The candidate set of one run: targets once, then the decoys of each seed in seed order, a decoy peptide that two
    seeds share kept once (with the first seed's parent). Only candidates whose [M+H]+ lies in mz_range (the acquired
    range). feature_mz = [M+H]+ rounded to 1e-6 (the query m/z). Rows sorted as the bench's matcher left them: by query
    m/z, then exact m/z, targets before decoys within one seed. Then the parent-family exclusion (excluded_decoys)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    t = targets.assign(is_decoy=False, parent_peptide=None, seed=seeds[0])
    with ProcessPoolExecutor(len(seeds), mp_context=mp.get_context("fork")) as pool:  # one process per seed
        ds = list(pool.map(partial(decoys, targets, method=method, min_shift_da=min_shift_da), seeds))
    parts = [_in_range(pd.concat([t, d.assign(seed=s)], ignore_index=True), mz_range) for s, d in zip(seeds, ds)]
    c = pd.concat([parts[0][~parts[0].is_decoy]] + [p[p.is_decoy] for p in parts], ignore_index=True)
    c = c[~c.is_decoy | ~c.peptide.duplicated()].reset_index(drop=True)
    c = c[~excluded_decoys(c)].reset_index(drop=True)
    c["is_decoy"] = c.is_decoy.astype(bool)
    return c


def _in_range(db, mz_range):
    db = db.assign(feature_mz=np.round(db.mh_mz.to_numpy(), 6))
    db = db[(db.feature_mz >= mz_range[0]) & (db.feature_mz <= mz_range[1])]
    return db.iloc[np.lexsort((np.arange(len(db)), db.mh_mz.to_numpy(), db.feature_mz.to_numpy()))]


def excluded_decoys(c, ppm=EXCLUDE_PPM):
    """Decoys within ppm of their parent's [M+H]+ + n x 1.00336 (n = 0, +-1, +-2) + k O (k = 0, +-1, +-2, +-3): there
    they would read an ion the parent family owns (its isotopes, its HyP forms).
    ponytail: as the bench, the parent mass comes from the targets in c, so a decoy whose parent lies outside the m/z
    range is never excluded (311-512 decoys per dataset); open question for RD (fix: parent masses from all targets)."""
    tmass = c[~c.is_decoy].groupby("peptide").mh_mz.first()
    dm = c.mh_mz - c.parent_peptide.map(tmass)
    tol = ppm * 1e-6 * c.mh_mz
    hit = np.any([(dm - n * NEUTRON - k * O_MASS).abs() <= tol for n in (0, 1, -1, 2, -2) for k in (0, 1, -1, 2, -2, 3, -3)],
                 axis=0)
    return c.is_decoy.to_numpy(bool) & hit


if __name__ == "__main__":  # self-checks
    t = pd.DataFrame({"peptide": ["AGLPGEVGAK", "DEFGHIPSTWK"], "protein": ["P1", "P2"], "hyp_base": None})
    t["mass"] = [pmass.calculate_mass(sequence=s) for s in t.peptide]
    t = pd.concat([t, t.iloc[:1].assign(hyp_base="AGLPGEVGAK", peptide="AGLPGEVGAK+h1", mass=t.mass[0] + O_MASS)],
                  ignore_index=True)
    t["mh_mz"] = t.mass + PROTON
    d = decoys(t, 0)
    assert set(d.parent_peptide) == {"AGLPGEVGAK", "DEFGHIPSTWK", "AGLPGEVGAK+h1"}
    hd = d.set_index("parent_peptide")
    assert hd.loc["AGLPGEVGAK+h1", "peptide"] == hd.loc["AGLPGEVGAK", "peptide"] + "+h1"
    assert abs(hd.loc["AGLPGEVGAK+h1", "mass"] - hd.loc["AGLPGEVGAK", "mass"] - O_MASS) < 1e-9
    c = pd.DataFrame({"peptide": ["T", "D1", "D2"], "is_decoy": [False, True, True], "parent_peptide": [None, "T", "T"],
                      "mh_mz": [1000.0, 1000.0 + NEUTRON + 1e-3, 1010.0]})
    assert list(excluded_decoys(c)) == [False, True, False]
    print("candidates ok")
