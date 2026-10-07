"""FASTA digest, decoy generation, and MALDI m/z matching."""

import bisect
import collections
import hashlib
import logging
import random

import numpy as np
import pandas as pd
from pyteomics import fasta, mass, parser

from msi_picasso.utils import PROTON

logger = logging.getLogger(__name__)

# Monoisotopic residue masses for the 18 non-K/R standard amino acids.
# K (128.09496) and R (156.10111) are excluded: introducing them would add
# tryptic cleavage sites.  I and L are listed separately (both 113.08406)
# so the dict covers all 20 standard AAs for lookup, but the substitution
# alphabet excludes K, R, and any AA isobaric with the residue being replaced.
_AA_RESIDUE_MASSES: dict[str, float] = {
    "G": 57.02146, "A": 71.03711, "V": 99.06841, "L": 113.08406,
    "I": 113.08406, "P": 97.05276, "F": 147.06841, "W": 186.07931,
    "M": 131.04049, "S": 87.03203, "T": 101.04768, "C": 103.00919,
    "Y": 163.06333, "H": 137.05891, "D": 115.02694, "E": 129.04259,
    "N": 114.04293, "Q": 128.05858,
}
_SUB_ALPHABET: tuple[str, ...] = tuple(sorted(_AA_RESIDUE_MASSES))  # 18 AAs, no K/R

# The two sulfur-carrying residues. Sulfur dominates isotope-envelope shape, so it is
# the composition axis the ranker is most sensitive to (PROGRESS.md F-036, F-042).
_SULFUR_RESIDUES: frozenset[str] = frozenset("CM")


def target_rows(peptide_db):
    """The target rows every decoy generator is fed. Drop decoys, change nothing else.

    Use this rather than re-filtering by hand. Decoy placement consumes one RNG draw
    per row and avoids m/z already taken, so altering the row set -- de-duplicating
    peptides, restricting to a peak m/z range -- shifts the entire decoy stream, not
    just the rows removed. ``scripts/prefilter_peaklist.py`` re-implemented this with
    two extra filters and silently produced a different decoy set (PROGRESS.md F-049).
    """
    return peptide_db[~peptide_db["is_decoy"].astype(bool)].reset_index(drop=True)


def _residue_frequencies(peptides) -> dict[str, float] | None:
    """Empirical frequency of each substitutable residue in a peptide set.

    H-decoy-15. `rng.choice(pool)` draws uniformly over the 18-letter alphabet, so
    every residue is produced 5.6% of the time whatever its real abundance. Cys and
    Met together are 11.1% of a uniform draw against 1.4-1.8% of the residues in
    these datasets' own target peptides, which is a 6-8x over-production and the
    whole source of the decoys' doubled sulfur content (PROGRESS.md F-042). Drawing
    from these weights instead removes the asymmetry at its source, and does so on
    every composition axis at once rather than only on sulfur.

    Returns None when the peptide set carries no substitutable residue at all, in
    which case the caller falls back to the uniform draw.
    """
    counts: collections.Counter = collections.Counter()
    for pep in peptides:
        counts.update(pep)
    total = sum(counts[aa] for aa in _SUB_ALPHABET)
    if total <= 0:
        return None
    return {aa: counts[aa] / total for aa in _SUB_ALPHABET}


def _draw_replacement(rng, pool: list[str], weights: dict[str, float] | None) -> str:
    """Pick one replacement residue from `pool`, weighted by `weights` if given.

    `pool` is already filtered to the residues that satisfy this substitution's mass
    constraints, so the weights are renormalised over what survived. Falls back to a
    uniform draw when every surviving residue has zero weight, which keeps a decoy
    that would otherwise be lost.
    """
    if weights is None:
        return str(rng.choice(pool))
    w = np.array([weights.get(aa, 0.0) for aa in pool], dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return str(rng.choice(pool))
    return str(rng.choice(pool, p=w / w.sum()))


def _replacement_pool(current_aa: str, alphabet, up: bool) -> list[str]:
    """Residues in `alphabet` heavier (`up`) or lighter than `current_aa`.

    Strict inequality excludes `current_aa` itself and any isobaric residue (L/I).
    """
    m = _AA_RESIDUE_MASSES[current_aa]
    return [aa for aa in alphabet if (_AA_RESIDUE_MASSES[aa] > m if up else _AA_RESIDUE_MASSES[aa] < m)]


def _cleave(seq, enzyme, missed_cleavages, min_length, max_length) -> list[str]:
    """Sorted unique cleavage products of `seq` within [min_length, max_length]."""
    return [
        pep for pep in sorted(parser.cleave(
            seq, parser.expasy_rules.get(enzyme, enzyme), missed_cleavages=missed_cleavages,
        ))
        if min_length <= len(pep) <= max_length
    ]


def _assign_mass_columns(df, sequences=None, log=False):
    """Compute mass + elemental composition and assign the 7 columns onto ``df``
    in place (``mass``, ``mh_mz``, ``n_C``, ``n_H``, ``n_N``, ``n_O``, ``n_S``).

    Uses the Rust ``compute_peptide_masses`` backend if importable, else the
    pyteomics fallback. When ``log`` is True, logs which backend was used.
    """
    if sequences is None:
        sequences = df["peptide"].tolist()
    try:
        from ms1rescore_rs import compute_peptide_masses

        masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss = compute_peptide_masses(sequences)
        cols = {"mass": masses, "mh_mz": mh_mzs, "n_C": n_cs, "n_H": n_hs,
                "n_N": n_ns, "n_O": n_os, "n_S": n_ss}
        if log:
            logger.info("  (used Rust backend for mass computation)")
    except ImportError:
        if log:
            logger.info("  (using pyteomics for mass computation)")
        masses_list = []
        for seq in sequences:
            try:
                comp = mass.Composition(sequence=seq)
                pep_mass = mass.calculate_mass(composition=comp)
                masses_list.append({
                    "mass": pep_mass, "mh_mz": pep_mass + PROTON,
                    "n_C": comp.get("C", 0), "n_H": comp.get("H", 0),
                    "n_N": comp.get("N", 0), "n_O": comp.get("O", 0),
                    "n_S": comp.get("S", 0),
                })
            except Exception:
                masses_list.append({
                    "mass": 0, "mh_mz": 0, "n_C": 0, "n_H": 0,
                    "n_N": 0, "n_O": 0, "n_S": 0,
                })
        mass_df = pd.DataFrame(masses_list)
        cols = {col: mass_df[col].values for col in mass_df.columns}
    for col, vals in cols.items():
        df[col] = vals


def _add_protein_count_features(result, target_candidates):
    """Add ``n_candidates``, ``protein_n_features`` and (when available)
    ``protein_tryptic_count`` to a combined target+decoy frame, in place.

    ``protein_tryptic_count`` is the full-digest peptide count per protein;
    decoys carry a ``DECOY_``-prefixed protein, so the prefix is stripped to
    inherit the source protein's count (keeps ``protein_coverage`` symmetric
    between a protein and its decoy). Shared by the mz_shuffle / substitution paths.
    """
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")
    prot_feat_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(prot_feat_count).fillna(0).astype(int)
    if "protein_tryptic_count" in target_candidates.columns:
        prot_tryptic = (
            target_candidates.drop_duplicates(subset=["protein"])
            .set_index("protein")["protein_tryptic_count"]
            .to_dict()
        )
        _base_prot = result["protein"].astype(str).str.replace(r"^DECOY_", "", regex=True)
        result["protein_tryptic_count"] = (
            _base_prot.map(prot_tryptic).fillna(0).astype(int)
        )


def _shuffle_protein(seq: str, random_state: int = 42) -> str:
    """
    Shuffle non-K/R residues of a protein sequence randomly while keeping
    K and R at their original positions.

    Keeping K/R in place ensures the decoy protein is digested at the same
    tryptic cleavage sites as the target, preserving peptide length and charge
    distributions. Shuffling (rather than reversing) the non-K/R residues
    gives decoy peptides different amino acid compositions from targets of the
    same mass/length, making isotope envelope features (theo_isotope_cosine,
    theo_isotope_chi2, theo_isotope_kl) genuinely discriminative. K/R-fixed
    reversal creates isobaric peptides with near-identical isotope patterns.
    """
    kr_positions = {i for i in range(len(seq)) if seq[i] in "KR"}
    non_kr = [seq[i] for i in range(len(seq)) if seq[i] not in "KR"]
    rng = random.Random(random_state)
    rng.shuffle(non_kr)
    result = list(seq)
    j = 0
    for i in range(len(result)):
        if i not in kr_positions:
            result[i] = non_kr[j]
            j += 1
    return "".join(result)


def digest_fasta(
    fasta_path: str,
    enzyme: str = "trypsin",
    missed_cleavages: int = 1,
    min_length: int = 7,
    max_length: int = 30,
    generate_decoys: bool = True,
) -> pd.DataFrame:
    """
    In-silico tryptic digest of a FASTA file with decoy generation.

    Decoy strategy: reverse each protein keeping K/R at original positions,
    then digest. Produces 1:1 paired decoys with identical mass distribution.

    Returns DataFrame with columns:
        peptide, protein, mass, mh_mz, is_decoy, n_C, n_H, n_N, n_O, n_S
    """

    # Phase 1: Cleave all proteins (pyteomics), collect sequences
    rows = []  # (peptide, protein, is_decoy)
    for desc, seq in fasta.read(fasta_path):
        protein_id = desc.split("|")[1] if "|" in desc else desc.split()[0]
        rows += [(pep, protein_id, False)
                 for pep in _cleave(seq, enzyme, missed_cleavages, min_length, max_length)]
        if generate_decoys:
            rows += [(pep, f"DECOY_{protein_id}", True)
                     for pep in _cleave(_shuffle_protein(seq), enzyme, missed_cleavages,
                                        min_length, max_length)]

    df = pd.DataFrame(rows, columns=["peptide", "protein", "is_decoy"])
    df = df.drop_duplicates(subset=["peptide", "is_decoy"])

    # Remove decoys whose sequence is identical to a target (arises when all
    # non-K/R residues in a peptide are identical or there is only one — the
    # K/R-preserving shuffle is then a no-op for that peptide).
    target_seqs = set(df.loc[~df["is_decoy"], "peptide"])
    n_before = df["is_decoy"].sum()
    df = df[~(df["is_decoy"] & df["peptide"].isin(target_seqs))].reset_index(drop=True)
    n_removed = n_before - df["is_decoy"].sum()
    if n_removed > 0:
        logger.debug("  Removed %d decoy sequences identical to a target peptide", n_removed)

    # Phase 2: Compute masses + elemental composition (Rust if available, else pyteomics)
    _assign_mass_columns(df, log=True)

    # Remove peptides with unknown amino acids (mass=0)
    df = df[df["mass"] > 0].reset_index(drop=True)
    df["is_decoy"] = df["is_decoy"].astype(bool)
    logger.info(
        f"Digested {fasta_path}: {(~df['is_decoy']).sum()} target, "
        f"{df['is_decoy'].sum()} decoy peptides"
    )
    return df


def match_to_maldi_features(
    maldi_mzs: np.ndarray,
    peptide_db: pd.DataFrame,
    ppm_tolerance: float = 20.0,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Match MALDI m/z features to digest candidates within ppm tolerance.

    Uses Rust (ms1rescore_rs) if available for the m/z matching step.

    Parameters
    ----------
    maldi_intensities
        Per-feature intensity array aligned with ``maldi_mzs``.  Prefer
        passing ``maldi_intensities_p90`` (90th-percentile of nonzero pixels)
        rather than mean-of-nonzero, as p90 decouples intensity magnitude from
        spatial coverage.  If only this argument is supplied it is used for
        ``feature_intensity`` (backwards compatibility).
    maldi_intensities_p90
        90th-percentile intensity of nonzero pixels per feature.  Robust
        estimate of peak intensity that is not confounded by spatial coverage
        (``fraction_detected`` handles that separately).  Preferred over
        mean-of-nonzero.  Computed by ``compute_spatial_features`` as
        ``intensity_p90``.
    maldi_intensities_sum
        Sum of nonzero pixel intensities per feature.  Computed by
        ``compute_spatial_features`` as ``intensity_sum``.

    Returns candidate table with columns from peptide_db plus:
        feature_mz, feature_idx, ppm_error, ppm_error_abs,
        protein_n_features, n_candidates
    """
    peptide_mzs = peptide_db["mh_mz"].values

    def _assign_intensities(df: pd.DataFrame, idx) -> None:
        if maldi_intensities_p90 is not None:
            df["feature_intensity_p90"] = maldi_intensities_p90[idx]
        if maldi_intensities_sum is not None:
            df["feature_intensity_sum"] = maldi_intensities_sum[idx]
        if maldi_intensities is not None:
            df["feature_intensity"] = maldi_intensities[idx]

    try:
        from ms1rescore_rs import match_mz

        feat_idx, pep_idx, ppm_errors = match_mz(
            maldi_mzs.tolist(), peptide_mzs.tolist(), ppm_tolerance
        )
        if len(feat_idx) == 0:
            logger.warning("No candidates matched any MALDI features")
            return pd.DataFrame()

        feat_idx = np.array(feat_idx, dtype=np.int64)
        pep_idx = np.array(pep_idx, dtype=np.int64)
        ppm_errors = np.array(ppm_errors)

        result = peptide_db.iloc[pep_idx].copy()
        result = result.reset_index(drop=True)
        result["feature_mz"] = maldi_mzs[feat_idx]
        result["feature_idx"] = feat_idx
        result["ppm_error"] = ppm_errors
        result["ppm_error_abs"] = np.abs(ppm_errors)
        _assign_intensities(result, feat_idx)
        logger.info("  (used Rust backend for m/z matching)")

    except ImportError:
        matches = []
        db_mz = peptide_mzs
        sort_idx = np.argsort(db_mz)
        db_mz_sorted = db_mz[sort_idx]

        for i, mz in enumerate(maldi_mzs):
            tol = mz * ppm_tolerance / 1e6
            lo = np.searchsorted(db_mz_sorted, mz - tol, side="left")
            hi = np.searchsorted(db_mz_sorted, mz + tol, side="right")
            if lo >= hi:
                continue
            candidate_idx = sort_idx[lo:hi]
            candidates = peptide_db.iloc[candidate_idx].copy()
            candidates["feature_mz"] = mz
            candidates["feature_idx"] = i
            candidates["ppm_error"] = (mz - candidates["mh_mz"]) / candidates["mh_mz"] * 1e6
            candidates["ppm_error_abs"] = candidates["ppm_error"].abs()
            _assign_intensities(candidates, i)
            matches.append(candidates)

        if not matches:
            logger.warning("No candidates matched any MALDI features")
            return pd.DataFrame()
        result = pd.concat(matches, ignore_index=True)

    # Protein-level consistency: count distinct MALDI features per protein
    # Computed over ALL candidates (targets + decoys) — symmetric
    protein_feature_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(protein_feature_count).fillna(0).astype(int)

    # Full tryptic digest count per protein (from peptide_db, before m/z filtering).
    # Used by compute_protein_consistency_features to compute protein_coverage correctly.
    protein_tryptic_counts = peptide_db.groupby("protein")["peptide"].nunique()
    result["protein_tryptic_count"] = result["protein"].map(protein_tryptic_counts).fillna(0).astype(int)

    # Candidates per feature
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")

    # A10 — Kendrick mass defect (CH₂ reference unit: 14 / 14.01565).
    # KMD = KM − round(KM) ∈ [-0.5, 0.5).
    kendrick_mass = result["feature_mz"] * (14.0 / 14.01565)
    result["kendrick_mass_defect"] = kendrick_mass - np.round(kendrick_mass)

    logger.info(
        f"Matched {result['feature_mz'].nunique()}/{len(maldi_mzs)} features → "
        f"{(~result['is_decoy']).sum()} target + {result['is_decoy'].sum()} decoy candidates"
    )
    return result


def generate_mz_shuffle_candidates(
    target_df: pd.DataFrame,
    feature_mzs: np.ndarray,
    matching_ppm: float = 20.0,
    random_state: int = 42,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate m/z-assignment-shuffle decoys: a derangement of the target
    peptide -> feature assignment.

    Each unique target peptide is matched to its representative MALDI feature
    (the matched feature with the smallest ``ppm_error_abs``).  Decoys are formed
    by permuting which peptide is assigned to which feature, so every decoy is a
    REAL target peptide relocated onto a DIFFERENT real feature (the one belonging
    to another peptide).  Consequences, which make this a good TDC null:

    - Decoy features are the SAME set as target features (1 target + 1 decoy per
      feature, co-located on the identical ion image), so feature-quality features
      (intensity, fraction_detected, spatial autocorrelation, colocalization) are
      *identical* between the target and the decoy at a feature and contribute
      nothing to the target/decoy separation.  Discrimination is forced onto the
      peptide-specific predicted-vs-observed match (CCS, isotope pattern).
    - The permutation is built on a mass-sorted rotation, so a peptide is never
      assigned to its own feature (no fixed point) and never to a near-isobaric
      feature (the rotation magnitude spans a large mass-rank gap).

    ``ppm_error`` on decoy rows is copied from the peptide's best target match
    (non-discriminative) — in raw-query mode it is later recomputed from the
    observed peak centroid at the assigned feature (symmetric, ~0).  Mass accuracy
    must NOT be computed against the decoy peptide's own mass: that mismatch does
    not exist for real false positives (which match a feature within tolerance), so
    using it would make the null anti-conservative.

    ``decoy_delta_da`` stores assigned_feature_mz - peptide_mh_mz (diagnostic).
    ``source = "decoy_mz_shuffle"``.  Returns a combined target+decoy DataFrame
    with the same schema as ``match_to_maldi_features()`` plus ``decoy_delta_da``.
    """
    feature_mzs = np.asarray(feature_mzs, dtype=np.float64)

    # --- Match targets against MALDI features (normal path) ---
    target_candidates = match_to_maldi_features(
        feature_mzs, target_df, matching_ppm,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    if len(target_candidates) == 0:
        logger.warning("mz_shuffle: no target candidates matched — returning empty")
        return target_candidates
    target_candidates["decoy_delta_da"] = np.nan
    if "source" not in target_candidates.columns:
        target_candidates["source"] = "target"

    # Representative feature per unique target peptide = its best (lowest |ppm|) match.
    best_idx = target_candidates.groupby("peptide")["ppm_error_abs"].idxmin()
    best = target_candidates.loc[best_idx].reset_index(drop=True)
    n = len(best)
    if n < 2:
        logger.warning("mz_shuffle: <2 unique target peptides — returning target-only")
        return target_candidates

    mh = best["mh_mz"].to_numpy(dtype=np.float64)
    feat_mz = best["feature_mz"].to_numpy(dtype=np.float64)
    feat_idx = best["feature_idx"].to_numpy()

    # Mass-sorted rotation derangement: in mass-rank space assign each peptide to
    # the one `k` ranks away (cyclic).  k in [n/4, 3n/4) guarantees both no fixed
    # point and a large mass gap (never near-isobaric).
    rng = np.random.default_rng(random_state)
    order = np.argsort(mh)
    if n > 3:
        k = int(rng.integers(max(1, n // 4), max(2, 3 * n // 4)))
    else:
        k = 1
    rolled = np.roll(order, k)
    sigma = np.empty(n, dtype=np.int64)
    sigma[order] = rolled  # sigma[i] = index of the peptide whose feature i is assigned to

    # ppm inherited from each peptide's own best target match (non-discriminative).
    pep_ppm = best["ppm_error"].to_numpy(dtype=np.float64)

    # --- Build decoy rows: peptide i relocated onto feature of peptide sigma[i] ---
    decoy_df = best.copy()
    decoy_df["is_decoy"] = True
    decoy_df["source"] = "decoy_mz_shuffle"
    # Separate protein namespace so protein-level features are computed within class
    # (a decoy must not be pooled with its source target's protein peptides).
    decoy_df["protein"] = "DECOY_" + decoy_df["protein"].astype(str)
    decoy_df["feature_mz"] = feat_mz[sigma]
    decoy_df["feature_idx"] = feat_idx[sigma]
    decoy_df["decoy_delta_da"] = feat_mz[sigma] - mh
    decoy_df["ppm_error"] = pep_ppm
    decoy_df["ppm_error_abs"] = np.abs(pep_ppm)

    fi = feat_idx[sigma]
    if maldi_intensities_p90 is not None:
        decoy_df["feature_intensity_p90"] = maldi_intensities_p90[fi.astype(int)]
    if maldi_intensities_sum is not None:
        decoy_df["feature_intensity_sum"] = maldi_intensities_sum[fi.astype(int)]
    if maldi_intensities is not None:
        decoy_df["feature_intensity"] = maldi_intensities[fi.astype(int)]

    kendrick = decoy_df["feature_mz"].to_numpy() * (14.0 / 14.01565)
    decoy_df["kendrick_mass_defect"] = kendrick - np.round(kendrick)

    # --- Combine and recompute per-feature / per-protein statistics ---
    # Use the representative-feature target set (`best`, one row per unique
    # peptide), NOT the full multiplicity `target_candidates`. A target peptide
    # whose m/z falls within `matching_ppm` of several MALDI peaks otherwise
    # yields multiple target rows while its single decoy yields one, producing a
    # ~(mean features/peptide):1 target:decoy imbalance (e.g. 5901:2895) and
    # leaving most target rows without a co-located decoy. Deduplicating to
    # `best` realises the mz_shuffle design — exactly 1 target + 1 decoy per
    # peptide, co-located on the identical feature — and loses no unique peptide
    # identifications (only redundant near-isobaric secondary matches; the
    # lowest-|ppm| match is kept).
    result = pd.concat([best, decoy_df], ignore_index=True)
    result["is_decoy"] = result["is_decoy"].astype(bool)
    _add_protein_count_features(result, target_candidates)

    logger.info(
        "mz_shuffle: %d features → %d target + %d decoy candidates "
        "(every decoy co-located with a target on a real feature)",
        result["feature_mz"].nunique(),
        int((~result["is_decoy"]).sum()),
        int(result["is_decoy"].sum()),
    )
    return result


# Retry budget per peptide when the collision filter rejects a candidate decoy.
_SUBSTITUTION_MAX_ATTEMPTS = 200


def generate_substitution_candidates(
    target_df: pd.DataFrame,
    feature_mzs: np.ndarray,
    matching_ppm: float = 20.0,
    n_residues: int = 1,
    random_seed: int = 42,
    mass_shift_min_da: float | None = None,
    mass_shift_max_da: float | None = None,
    collision_filter: bool = True,
    collision_ppm: float | None = None,
    snap_to_features: bool = False,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
    residue_weighting: str = "uniform",
    preserve_sulfur: bool = False,
) -> pd.DataFrame:
    """
    Generate sequence-space substitution decoys and return a combined
    target + decoy candidates DataFrame.

    One decoy p′ is generated per unique target peptide p by substituting
    n_residues interior non-K/R residues.  In raw-query mode (snap_to_features=False)
    each decoy is queried at its own theoretical [M+H]+, giving it a genuine on-demand
    ion image at a mass distinct from all targets.  This makes the null:

    - Size-fair for protein-level features: |DECOY_X| == |X| by construction.
    - CCS-safe: mass shift is ~1-50 Da (0.1% of ~1000 Da), far below the mz_shuffle
      gap of 50-500 Da; no m/z-gap CCS artifact; _MZ_SHUFFLE_CCS_LEAK_FEATURES
      exclusion does NOT apply.
    - Spatial-ranker-compatible: each decoy has its own real on-demand ion image.
    - Target-independent: decoy(p) = f(p, seed) with no cross-peptide dependence
      (each peptide uses its own MD5-seeded RNG).

    Sign symmetry (~50% up / ~50% down mass shifts) is enforced via the upper 32 bits
    of the MD5 hash of the peptide sequence.  K and R are excluded from both the
    substitution alphabet and the substitution targets to preserve tryptic cleavage.
    The L/I isobaric pair (both 113.084 Da) is handled automatically by excluding
    any replacement with the same replacement mass as the current residue.

    **Elemental composition is NOT preserved, and that has consequences** — see
    invariant 3 in CLAUDE.md, and PROGRESS.md F-003, F-036 and F-042. Two options
    control how far the decoys' composition is allowed to drift from the targets',
    both off by default so that every result predating them reproduces exactly:

    ``residue_weighting``
        ``"uniform"`` (default) draws the replacement residue uniformly over the
        18-letter alphabet, so each residue appears 5.6% of the time whatever its
        real abundance.  ``"target_frequency"`` draws it from the empirical residue
        frequency of the target peptides instead (H-decoy-15).  This matters because
        Cys and Met are 2 of 18 letters but only 1.4-1.8% of the residues in real
        peptides, so the uniform draw over-produces sulfur by 6-8x and gives decoys
        roughly twice the sulfur content of targets.  Features that read composition
        then partially read the target/decoy label with no spectral evidence
        involved: F-036 measured that for the isotope-envelope family, F-042 for
        four more ranker features.  Weighting acts on every composition axis, not
        only sulfur.

    ``preserve_sulfur``
        When True, Cys and Met are removed from the substitution alphabet *and*
        positions already holding one are removed from the eligible set, so a decoy
        carries exactly its source target's sulfur count.  Narrower than the
        weighting above but exact on the axis that dominates isotope-envelope shape.
        The two compose, and using both is the strongest available guarantee.

    Composition symmetry is a property of the decoys and is cheap to check directly:
    compare mean ``n_S`` between the classes on the returned frame, or run
    ``scripts/audit_composition_leak.py`` on a finished run.

    LC-MS/MS evidence columns are wiped for decoy rows: p′ is a fictional sequence
    not present in the LC-MS/MS run, so inheriting evidence would break TDC symmetry.

    ppm_error is initialized to 0.0 and overwritten by the pipeline's
    _recompute_ppm_from_centroids call in raw-query mode.  feature_mz on decoy rows
    is p′'s own [M+H]+ — this is load-bearing for raw-query extraction.

    Returns a DataFrame with the same schema as match_to_maldi_features() plus
    decoy_delta_da (NaN for targets, p'_mhz - p_mhz for decoys).
    """
    feature_mzs = np.asarray(feature_mzs, dtype=np.float64)
    n_features = len(feature_mzs)
    tol_frac = (matching_ppm if collision_ppm is None else collision_ppm) * 1e-6

    unique_pep = (
        target_df[~target_df["is_decoy"].astype(bool)]
        .drop_duplicates(subset="peptide")
        .reset_index(drop=True)
    )
    n_unique = len(unique_pep)
    target_mzs_sorted = np.sort(unique_pep["mh_mz"].values.astype(np.float64))

    # Composition-symmetry options (H-decoy-15 / F-042). Both narrow what may be
    # substituted, so they are resolved once here rather than per peptide.
    if residue_weighting not in ("uniform", "target_frequency"):
        raise ValueError(
            f"residue_weighting must be 'uniform' or 'target_frequency', "
            f"got {residue_weighting!r}"
        )
    sub_alphabet = tuple(
        aa for aa in _SUB_ALPHABET if not (preserve_sulfur and aa in _SULFUR_RESIDUES)
    )
    residue_weights = (
        _residue_frequencies(unique_pep["peptide"].astype(str))
        if residue_weighting == "target_frequency" else None
    )
    if residue_weighting == "target_frequency" and residue_weights is None:
        logger.warning(
            "substitution: residue_weighting='target_frequency' requested but the "
            "target peptides carry no substitutable residue; falling back to uniform"
        )
    if residue_weights is not None:
        _sulfur_freq = sum(residue_weights.get(aa, 0.0) for aa in _SULFUR_RESIDUES)
        logger.info(
            "substitution: drawing replacements from the target residue frequency "
            "(C+M %.2f%% of target residues against %.2f%% under a uniform draw)",
            100.0 * _sulfur_freq, 100.0 * len(_SULFUR_RESIDUES) / len(_SUB_ALPHABET),
        )
    if preserve_sulfur:
        logger.info(
            "substitution: preserving sulfur — C and M are neither substituted in "
            "nor out, so every decoy carries its source target's sulfur count"
        )

    used_decoy_mz: list[float] = []
    next_decoy_idx = n_features
    n_skipped = 0
    n_collisions = 0
    n_relaxed = 0

    def _collides_target(mz: float) -> bool:
        lo = np.searchsorted(target_mzs_sorted, mz * (1.0 - tol_frac), side="left")
        hi = np.searchsorted(target_mzs_sorted, mz * (1.0 + tol_frac), side="right")
        return lo < hi

    def _collides_used(mz: float, relax: bool = False) -> bool:
        if relax or not used_decoy_mz:
            return False
        j = bisect.bisect_left(used_decoy_mz, mz * (1.0 - tol_frac))
        return j < len(used_decoy_mz) and used_decoy_mz[j] <= mz * (1.0 + tol_frac)

    # Accepted decoys: list of (src_idx, p_prime, net_delta, approx_mhz, feature_idx)
    accepted: list[tuple] = []

    for i in range(n_unique):
        peptide = unique_pep.at[i, "peptide"]
        orig_mhz = float(unique_pep.at[i, "mh_mz"])
        L = len(peptide)

        # Per-peptide hash-based direction and RNG (independent of all other peptides)
        _digest = hashlib.md5(peptide.encode()).digest()
        _pep_hash_int = int.from_bytes(_digest[:8], "little")
        upshift = bool(int.from_bytes(_digest[8:12], "little") % 2 == 0)
        rng = np.random.default_rng(random_seed ^ _pep_hash_int)

        # Eligible positions: interior (index 1..L-2), non-K/R, and with
        # preserve_sulfur also not already holding a C or M, so sulfur cannot be
        # substituted out any more than it can be substituted in.
        eligible = [
            pos for pos in range(1, L - 1)
            if peptide[pos] not in "KR"
            and not (preserve_sulfur and peptide[pos] in _SULFUR_RESIDUES)
        ]
        if len(eligible) < n_residues:
            logger.debug(
                "substitution: skipping '%s' — %d eligible positions, need %d",
                peptide, len(eligible), n_residues,
            )
            n_skipped += 1
            continue

        eligible_arr = list(eligible)
        rng.shuffle(eligible_arr)

        # Apply n substitutions sequentially at distinct positions, preferred
        # direction first (pass 0), fallback direction second (pass 1). The sign
        # constraint on the net delta is best-effort; log when unsatisfied.
        # Retry with a different draw when the decoy m/z is rejected (shift too
        # small or too large, or colliding with a target / an already-placed decoy).
        # Without this the peptide is dropped on the first rejection, which costs
        # ~60% of decoys once the collision filter is active.
        accepted_here = False
        for attempt in range(_SUBSTITUTION_MAX_ATTEMPTS):
            if attempt > 0:
                # Attempt 0 preserves the original RNG draw order, so runs with an
                # inert collision filter produce byte-identical decoys.
                rng.shuffle(eligible_arr)
            seq = list(peptide)
            net_delta = 0.0
            used_positions: set[int] = set()
            applied = 0

            for pass_num in range(2):
                if applied >= n_residues:
                    break
                for pos in eligible_arr:
                    if applied >= n_residues:
                        break
                    if pos in used_positions:
                        continue
                    current_aa = seq[pos]
                    current_mass = _AA_RESIDUE_MASSES.get(current_aa)
                    if current_mass is None:
                        continue
                    pool = _replacement_pool(current_aa, sub_alphabet, upshift == (pass_num == 0))
                    if not pool:
                        continue
                    replacement = _draw_replacement(rng, pool, residue_weights)
                    seq[pos] = replacement
                    net_delta += _AA_RESIDUE_MASSES[replacement] - current_mass
                    used_positions.add(pos)
                    applied += 1

            if applied < n_residues:
                # Structural: too few eligible positions. Retrying cannot help.
                logger.debug(
                    "substitution: could not apply %d substitutions to '%s' (applied %d)",
                    n_residues, peptide, applied,
                )
                break

            if (upshift and net_delta < 0) or (not upshift and net_delta > 0):
                logger.debug(
                    "substitution: net sign mismatch for '%s' (wanted %s, got %.4f Da)",
                    peptide, "up" if upshift else "down", net_delta,
                )

            approx_mhz = orig_mhz + net_delta
            min_shift = (
                mass_shift_min_da if mass_shift_min_da is not None
                else matching_ppm * orig_mhz / 1e6
            )
            if abs(net_delta) < min_shift:
                logger.debug(
                    "substitution: '%s' net shift %.4f Da < min %.4f Da — retrying",
                    peptide, abs(net_delta), min_shift,
                )
                continue
            # Upper bound: H-fdr-10. A large shift leaves the decoy with a composition
            # far from its source target's, and the isotope-envelope features read
            # composition, so those decoys separate from targets for a construction
            # reason rather than a spectral one (PROGRESS.md F-036: AUC 0.58-0.66 above
            # ~120 Da, against 0.50-0.53 below). It is the NET shift across all
            # substituted residues that matters. Rejecting resamples, it does not drop.
            if mass_shift_max_da is not None and abs(net_delta) > mass_shift_max_da:
                logger.debug(
                    "substitution: '%s' net shift %.4f Da > max %.4f Da — retrying",
                    peptide, abs(net_delta), mass_shift_max_da,
                )
                continue

            if collision_filter:
                if _collides_target(approx_mhz):
                    n_collisions += 1
                    continue
                _relax = attempt >= _SUBSTITUTION_MAX_ATTEMPTS // 2
                if not snap_to_features and _collides_used(approx_mhz, _relax):
                    n_collisions += 1
                    continue
                if _relax:
                    n_relaxed += 1

            p_prime = "".join(seq)
            cand_idx = -1
            if not snap_to_features:
                cand_idx = next_decoy_idx
                next_decoy_idx += 1
                bisect.insort(used_decoy_mz, approx_mhz)
            accepted.append((i, p_prime, net_delta, approx_mhz, cand_idx))
            accepted_here = True
            break

        if not accepted_here:
            n_skipped += 1

    n_accepted = len(accepted)
    logger.info(
        "substitution: %d/%d target peptides → valid decoys "
        "(%d skipped, %d colliding draws rejected across retries, "
        "%d placed only after relaxing decoy-vs-decoy proximity)",
        n_accepted, n_unique, n_skipped, n_collisions, n_relaxed,
    )
    # With retries, n_collisions counts rejected *draws*, not peptides, so it is not
    # a rate. What matters downstream is how many peptides ended up without a decoy:
    # _tdc_qvalues assumes a 1:1 target:decoy ratio and applies no correction.
    skip_rate = n_skipped / n_unique if n_unique > 0 else 0.0
    if skip_rate > 0.05:
        logger.warning(
            "substitution: %.1f%% of peptides (%d/%d) got no decoy after %d attempts each; "
            "target:decoy is %.2f, and the TDC q-value assumes 1:1 — q-values are "
            "anti-conservative by ~%.0f%%. Lower collision_ppm or raise the retry budget.",
            100.0 * skip_rate, n_skipped, n_unique, _SUBSTITUTION_MAX_ATTEMPTS,
            n_accepted / max(n_unique, 1),
            100.0 * (n_unique / max(n_accepted, 1) - 1.0),
        )

    # Match targets against MALDI features (normal path)
    target_candidates = match_to_maldi_features(
        feature_mzs, target_df, matching_ppm,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    target_candidates["decoy_delta_da"] = np.nan
    if "source" not in target_candidates.columns:
        target_candidates["source"] = "target"

    if n_accepted == 0:
        logger.warning("substitution: no valid decoys — returning target-only candidates")
        if "is_decoy" not in target_candidates.columns:
            target_candidates["is_decoy"] = False
        target_candidates["is_decoy"] = target_candidates["is_decoy"].astype(bool)
        return target_candidates

    if snap_to_features:
        # Feature-list mode: match substituted peptides against detected features,
        # exactly as targets are matched. Decoys whose m/z hits no peak are dropped.
        dec_pep_rows = []
        for (src_i, p_prime, _, _, _) in accepted:
            src_row = unique_pep.iloc[src_i]
            dec_pep_rows.append({
                "peptide": p_prime,
                "protein": "DECOY_" + str(src_row["protein"]),
                "is_decoy": True,
            })
        dec_pep_db = pd.DataFrame(dec_pep_rows)
        _assign_mass_columns(dec_pep_db)
        dec_pep_db = dec_pep_db[dec_pep_db["mass"] > 0].reset_index(drop=True)
        dec_cands = match_to_maldi_features(
            feature_mzs, dec_pep_db, matching_ppm,
            maldi_intensities=maldi_intensities,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
        n_dec = len(dec_cands)
        if n_dec < 0.10 * max(1, len(target_candidates)):
            logger.warning(
                "substitution (feature-list mode): only %d decoy candidates matched "
                "MALDI features (vs %d target). The TDC q-value assumes about 1:1, so "
                "q-values will be anti-conservative. Check the peak list and matching_ppm.",
                n_dec, len(target_candidates),
            )
        if n_dec > 0:
            dec_cands["is_decoy"] = True
            dec_cands["source"] = "decoy_substitution"
            dec_cands["decoy_delta_da"] = np.nan
            result = pd.concat([target_candidates, dec_cands], ignore_index=True)
        else:
            logger.warning("substitution: no decoy candidates matched MALDI features (snap_to_features=True)")
            result = target_candidates
        result["is_decoy"] = result["is_decoy"].astype(bool)
        _add_protein_count_features(result, target_candidates)
        logger.info(
            "substitution: %d features → %d target + %d decoy candidates",
            result["feature_mz"].nunique(),
            int((~result["is_decoy"]).sum()),
            int(result["is_decoy"].sum()),
        )
        return result

    # Raw-query mode: build decoy rows from accepted substitutions
    src_indices = [row[0] for row in accepted]
    p_primes = [row[1] for row in accepted]
    feat_indices = [row[4] for row in accepted]

    # Batch-compute accurate masses for all p′ sequences
    decoy_mass_df = pd.DataFrame({"peptide": p_primes})
    _assign_mass_columns(decoy_mass_df)

    src_rows = unique_pep.iloc[src_indices].reset_index(drop=True)
    decoy_df = src_rows.copy()
    decoy_df["peptide"] = p_primes
    decoy_df["protein"] = "DECOY_" + decoy_df["protein"].astype(str)
    decoy_df["is_decoy"] = True
    decoy_df["source"] = "decoy_substitution"

    # Overwrite mass and composition with p′ values
    for col in ["mass", "mh_mz", "n_C", "n_H", "n_N", "n_O", "n_S"]:
        if col in decoy_mass_df.columns:
            decoy_df[col] = decoy_mass_df[col].values

    # feature_mz = p′ [M+H]+  — load-bearing: raw-query extracts at this m/z
    decoy_df["feature_mz"] = decoy_mass_df["mh_mz"].values
    decoy_df["feature_idx"] = feat_indices
    decoy_df["ppm_error"] = 0.0  # overwritten by _recompute_ppm_from_centroids
    decoy_df["ppm_error_abs"] = 0.0

    src_mhzs = unique_pep.iloc[src_indices]["mh_mz"].values
    decoy_df["decoy_delta_da"] = decoy_mass_df["mh_mz"].values - src_mhzs

    kendrick = decoy_df["feature_mz"].values * (14.0 / 14.01565)
    decoy_df["kendrick_mass_defect"] = kendrick - np.round(kendrick)

    # Wipe LC-MS/MS evidence: p′ is a fictional sequence not in the LC-MS/MS run
    for col in list(decoy_df.columns):
        if col.startswith("lcms_") or col == "n_psms":
            decoy_df[col] = np.nan

    result = pd.concat([target_candidates, decoy_df], ignore_index=True)
    result["is_decoy"] = result["is_decoy"].astype(bool)
    _add_protein_count_features(result, target_candidates)

    logger.info(
        "substitution: %d features → %d target + %d decoy candidates",
        result["feature_mz"].nunique(),
        int((~result["is_decoy"]).sum()),
        int(result["is_decoy"].sum()),
    )
    return result


def _digest_shuffled_pseudo_protein(
    pseudo_protein: str,
    seed: int,
    enzyme: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
    exclude_seqs: set,
) -> list[str]:
    """Shuffle pseudo_protein, digest, filter length and exact-sequence exclusions."""
    shuffled = _shuffle_protein(pseudo_protein, random_state=seed)
    rule = parser.expasy_rules.get(enzyme, enzyme)
    seen: set[str] = set()
    kept = []
    for pep in parser.cleave(shuffled, rule, missed_cleavages=missed_cleavages):
        if (
            min_length <= len(pep) <= max_length
            and pep not in exclude_seqs
            and pep not in seen
        ):
            seen.add(pep)
            kept.append(pep)
    return kept


def _contamination_filter(
    candidates: pd.DataFrame,
    reference_mzs: np.ndarray,
    matching_ppm: float,
    label: str,
) -> pd.DataFrame:
    """Remove rows from candidates whose mh_mz is within matching_ppm of reference_mzs."""
    if len(candidates) == 0:
        return candidates
    cand_mzs = candidates["mh_mz"].values.astype(np.float64)
    n_total = len(candidates)
    collided_idx: set[int] = set()
    try:
        from ms1rescore_rs import match_mz
        _f, pep_idx, _e = match_mz(reference_mzs.tolist(), cand_mzs.tolist(), matching_ppm)
        collided_idx = set(int(i) for i in pep_idx)
    except ImportError:
        sorted_idx = np.argsort(cand_mzs)
        sorted_mzs = cand_mzs[sorted_idx]
        for tmz in reference_mzs:
            tol = tmz * matching_ppm / 1e6
            lo = np.searchsorted(sorted_mzs, tmz - tol, side="left")
            hi = np.searchsorted(sorted_mzs, tmz + tol, side="right")
            for j in range(lo, hi):
                collided_idx.add(int(sorted_idx[j]))
    n_collided = len(collided_idx)
    collision_rate = n_collided / n_total if n_total else 0.0
    logger.info(
        "%s: contamination filter removed %d/%d peptides (%.1f%% isobaric)",
        label, n_collided, n_total, 100.0 * collision_rate,
    )
    if collision_rate > 0.10:
        logger.warning(
            "%s: collision rate %.1f%% > 10%% — m/z-space overlap with reference "
            "is high; the null may be biased.",
            label, 100.0 * collision_rate,
        )
    return candidates[~candidates.index.isin(collided_idx)].reset_index(drop=True)


def generate_entrapment_from_lcms_ids(
    lcms_ids,
    matching_ppm: float = 20.0,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    enzyme: str = "trypsin",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate entrapment pseudo-target candidates and their paired decoys from
    shuffled LC-MS/MS peptides.

    Each identified protein's confirmed peptide sequences are sorted and
    concatenated into a per-protein pseudo-protein, then shuffled twice with
    ``_shuffle_protein`` (K/R-preserving):

    * First shuffle (seed = ``random_state + hash(accession)``):
      pseudo-targets — ``is_decoy=False``, ``source="entrapment_shuffled"``,
      ``protein="ENTRAPMENT_{accession}"``.
    * Second shuffle (seed XOR ``0xDEADBEEF``):
      paired decoys — ``is_decoy=True``, ``source="entrapment_decoy"``,
      ``protein="ENTRAPMENT_DECOY_{accession}"``.

    Both sets are required so that TDC competition stays balanced (Wen et al.).
    The pseudo-targets that survive TDC are the false-positive estimates;
    the decoys serve only to keep the TDC denominator correct.

    Both sets are filtered: exact-sequence matches with any LC-MS/MS confirmed
    peptide are removed, then isobaric matches (within ``matching_ppm``) against
    confirmed peptide m/z values are removed; a collision rate > 10% triggers a
    warning.

    Returns a combined peptide-DB DataFrame (pseudo-targets + decoys); the
    caller matches it to MALDI features via ``match_to_maldi_features``.
    """
    peps_df = lcms_ids.peptides
    if "protein" not in peps_df.columns or "sequence" not in peps_df.columns:
        raise ValueError(
            "lcms_ids.peptides must have 'sequence' and 'protein' columns"
        )

    target_seqs: set[str] = set(peps_df["sequence"].dropna())

    tgt_rows: list[pd.DataFrame] = []
    dec_rows: list[pd.DataFrame] = []

    for prot_acc, grp in peps_df.groupby("protein"):
        sorted_seqs = sorted(set(grp["sequence"].dropna()))
        if not sorted_seqs:
            continue
        pseudo_protein = "".join(sorted_seqs)
        prot_seed = (random_state + hash(str(prot_acc))) & 0xFFFFFFFF
        dec_seed  = (prot_seed ^ 0xDEADBEEF) & 0xFFFFFFFF

        tgt_peps = _digest_shuffled_pseudo_protein(
            pseudo_protein, prot_seed, enzyme, missed_cleavages,
            min_length, max_length, target_seqs,
        )
        dec_peps = _digest_shuffled_pseudo_protein(
            pseudo_protein, dec_seed, enzyme, missed_cleavages,
            min_length, max_length, target_seqs,
        )

        if tgt_peps:
            df = pd.DataFrame({"peptide": tgt_peps})
            df["protein"] = f"ENTRAPMENT_{prot_acc}"
            tgt_rows.append(df)
        if dec_peps:
            df = pd.DataFrame({"peptide": dec_peps})
            df["protein"] = f"ENTRAPMENT_DECOY_{prot_acc}"
            dec_rows.append(df)

    if not tgt_rows:
        logger.warning(
            "generate_entrapment_from_lcms_ids: no entrapment peptides generated "
            "(all shuffled digest sequences were exact matches with LC-MS/MS IDs)"
        )
        return pd.DataFrame()

    tgt_db = pd.concat(tgt_rows, ignore_index=True).drop_duplicates(subset="peptide").reset_index(drop=True)
    dec_db = pd.concat(dec_rows, ignore_index=True).drop_duplicates(subset="peptide").reset_index(drop=True) if dec_rows else pd.DataFrame()

    for db in [tgt_db, dec_db]:
        if len(db):
            _assign_mass_columns(db)

    tgt_db = tgt_db[tgt_db.get("mass", pd.Series(0, index=tgt_db.index)) > 0].reset_index(drop=True)
    if len(dec_db):
        dec_db = dec_db[dec_db["mass"] > 0].reset_index(drop=True)

    if len(tgt_db) == 0:
        return pd.DataFrame()

    # Contamination filter against confirmed LC-MS/MS peptide m/z values.
    _target_db_tmp = pd.DataFrame({"peptide": sorted(target_seqs)})
    _assign_mass_columns(_target_db_tmp)
    target_mzs = _target_db_tmp["mh_mz"].dropna().values.astype(np.float64)

    tgt_db = _contamination_filter(tgt_db, target_mzs, matching_ppm, "entrapment_shuffled")
    if len(dec_db):
        dec_db = _contamination_filter(dec_db, target_mzs, matching_ppm, "entrapment_decoy")

    if len(tgt_db) == 0:
        logger.warning("entrapment_shuffled: all pseudo-targets removed by contamination filter")
        return pd.DataFrame()

    tgt_db["is_decoy"] = False
    tgt_db["source"] = "entrapment_shuffled"

    parts = [tgt_db]
    if len(dec_db):
        dec_db["is_decoy"] = True
        dec_db["source"] = "entrapment_decoy"
        parts.append(dec_db)
    else:
        logger.warning(
            "entrapment_decoy: no paired decoys generated; TDC denominator will "
            "not account for entrapment peptides"
        )

    result = pd.concat(parts, ignore_index=True)
    logger.info(
        "entrapment: %d pseudo-targets + %d paired decoys across %d ENTRAPMENT proteins",
        len(tgt_db),
        len(dec_db) if len(dec_db) else 0,
        tgt_db["protein"].nunique(),
    )
    return result


def digest_identified_proteins(
    fasta_path: str | None,
    lcms_ids,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    generate_decoys: bool = True,
) -> pd.DataFrame:
    """
    Strategy C hybrid candidate generation.

    Builds a candidate DataFrame from:
    1. In-silico digest of LC-MS/MS-identified proteins (target + K/R-preserving
       shuffled decoy, via ``_shuffle_protein``). Skipped when ``fasta_path`` is
       ``None`` — all confirmed peptides are then treated as novel (LC-only mode).
    2. Directly identified LC-MS/MS peptides (``source="lcms_confirmed"``), even
       if also present in the protein digest.

    A peptide present in both the protein digest and ``lcms_ids.peptides`` is
    labelled ``"lcms_confirmed"``. Novel directly-identified peptides (not
    reachable by digesting the identified proteins) are added as targets with
    K/R-preserving peptide-level decoys.

    LC-MS/MS evidence columns are joined onto target rows and set to NaN for
    all decoy rows: ``lcms_q_value``, ``lcms_pep``, ``lcms_score``,
    ``n_psms``, ``lcms_charge``, ``lcms_rt_mean``, ``lcms_intensity``.

    Parameters
    ----------
    fasta_path
        Path to the protein FASTA file, or ``None`` to skip protein digestion
        and use only the LC-MS/MS identified peptides as candidates.
    lcms_ids
        ``LCMSIds`` namedtuple returned by ``parse_lcms_ids()``.
    enzyme
        Enzyme name recognised by ``pyteomics.parser.expasy_rules``.
    missed_cleavages
        Maximum allowed missed cleavages.
    min_length, max_length
        Peptide length range (inclusive).

    Returns
    -------
    DataFrame with the same columns as ``digest_fasta()`` plus
    ``source``, ``lcms_q_value``, ``lcms_pep``, ``lcms_score``,
    ``n_psms``, ``lcms_charge``, ``lcms_rt_mean``, ``lcms_intensity``.
    """
    _EV_COLS = {
        "q_value": "lcms_q_value",
        "pep": "lcms_pep",
        "score": "lcms_score",
        "n_psms": "n_psms",
        "charge": "lcms_charge",
        "rt_mean": "lcms_rt_mean",
        "lcms_intensity": "lcms_intensity",
        "lcms_ccs": "lcms_ccs",
    }
    _BASE_COLS = ["peptide", "protein", "is_decoy", "mass", "mh_mz",
                  "n_C", "n_H", "n_N", "n_O", "n_S", "source"]

    if fasta_path is not None:
        from msi_picasso.lcms_ids import filter_fasta_to_proteins

        # --- Step 1: Filter FASTA to identified proteins ---
        protein_seqs = filter_fasta_to_proteins(fasta_path, lcms_ids.proteins)

        if not protein_seqs:
            logger.warning(
                "No identified proteins found in FASTA — check accession format. "
                "Continuing with LC-MS/MS confirmed peptides only."
            )
            df = pd.DataFrame(columns=_BASE_COLS)
        else:
            # --- Step 2: Digest identified proteins (target + shuffled decoy) ---
            rows = []  # (peptide, protein, is_decoy)
            for acc, seq in sorted(protein_seqs.items()):
                rows += [(pep, acc, False)
                         for pep in _cleave(seq, enzyme, missed_cleavages, min_length, max_length)]
                if generate_decoys:
                    rows += [(pep, f"DECOY_{acc}", True)
                             for pep in _cleave(_shuffle_protein(seq), enzyme, missed_cleavages,
                                                min_length, max_length)]

            df = pd.DataFrame(rows, columns=["peptide", "protein", "is_decoy"])
            df = df.drop_duplicates(subset=["peptide", "is_decoy"])
    else:
        # LC-only mode: no FASTA digestion — all confirmed peptides will be added
        # as novel targets in Step 5 below.
        logger.info("  No FASTA provided — using LC-MS/MS identified peptides only as candidates.")
        df = pd.DataFrame(columns=_BASE_COLS)

    # --- Step 3: Compute masses (Rust if available, else pyteomics) ---
    sequences = df["peptide"].tolist()
    if sequences:
        _assign_mass_columns(df, sequences=sequences, log=True)
        df = df[df["mass"] > 0].reset_index(drop=True)

    # --- Step 4: Label source (only rows from protein digest; novel rows set in Step 5) ---
    if len(df) > 0:
        df["source"] = np.where(df["is_decoy"], "decoy", "protein_digest")

    # --- Step 5: Union with directly identified peptides ---
    lcms_pep_df = lcms_ids.peptides
    if len(lcms_pep_df) > 0:
        confirmed_seqs = set(lcms_pep_df["sequence"].values)

        # Mark digest targets that are LC-MS/MS confirmed
        target_mask = ~df["is_decoy"]
        df.loc[target_mask & df["peptide"].isin(confirmed_seqs), "source"] = "lcms_confirmed"

        # Find novel confirmed peptides not reachable from the protein digest
        existing_targets = set(df.loc[target_mask, "peptide"].values)
        novel_seqs = sorted(s for s in confirmed_seqs if s not in existing_targets)

        if novel_seqs:
            logger.info(
                f"  {len(novel_seqs)} LC-MS/MS-confirmed peptides not in protein digest "
                f"— adding as lcms_confirmed targets"
            )
            novel_rows = []

            # Target rows (always the same)
            for seq in novel_seqs:
                prot_row = lcms_pep_df[lcms_pep_df["sequence"] == seq].iloc[0]
                prot = str(prot_row.get("protein", ""))
                novel_rows.append((seq, prot, False))

            if generate_decoys:
                if fasta_path is None:
                    # Concatenated pseudo-protein decoy strategy for LC-only mode.
                    # Per-peptide shuffle produces decoys with identical elemental
                    # composition to their target (same residue multiset, just reordered)
                    # — making isotope envelope features non-discriminative. By
                    # concatenating all target peptides into a pseudo-protein and
                    # shuffling at that level, non-K/R residues are redistributed
                    # across tryptic boundaries, breaking composition conservation.
                    sorted_seqs = sorted(novel_seqs)  # fixed order for reproducibility
                    pseudo_protein = "".join(sorted_seqs)
                    shuffled_pseudo = _shuffle_protein(pseudo_protein, random_state=42)
                    target_set = set(novel_seqs)
                    decoy_peptides = [
                        p for p in _cleave(shuffled_pseudo, enzyme, missed_cleavages,
                                           min_length, max_length)
                        if p not in target_set
                    ]
                    n_targets = len(novel_seqs)
                    if len(decoy_peptides) < n_targets:
                        logger.warning(
                            "Concatenated pseudo-protein decoy digest produced %d decoys "
                            "for %d target peptides — TDC ratio will be < 1:1. Consider "
                            "increasing --missed-cleavages or --max-length.",
                            len(decoy_peptides), n_targets,
                        )
                    elif len(decoy_peptides) > n_targets:
                        decoy_peptides = random.Random(42).sample(decoy_peptides, n_targets)
                    for dec in decoy_peptides:
                        novel_rows.append((dec, "DECOY_concat", True))
                else:
                    # Per-peptide K/R-preserving shuffle for novel sequences that are
                    # not reachable from the protein digest (Strategy C hybrid).
                    for seq in novel_seqs:
                        prot_row = lcms_pep_df[lcms_pep_df["sequence"] == seq].iloc[0]
                        prot = str(prot_row.get("protein", ""))
                        dec = _shuffle_protein(seq, random_state=42)
                        if dec != seq:
                            novel_rows.append((dec, f"DECOY_{prot}", True))

            novel_df = pd.DataFrame(novel_rows, columns=["peptide", "protein", "is_decoy"])
            novel_df = novel_df.drop_duplicates(subset=["peptide", "is_decoy"])

            # Compute masses for novel sequences
            novel_seqs_list = novel_df["peptide"].tolist()
            _assign_mass_columns(novel_df, sequences=novel_seqs_list)

            novel_df = novel_df[novel_df["mass"] > 0].reset_index(drop=True)
            novel_df["is_decoy"] = novel_df["is_decoy"].astype(bool)
            novel_df.loc[~novel_df["is_decoy"], "source"] = "lcms_confirmed"
            novel_df.loc[novel_df["is_decoy"], "source"] = "decoy"
            df = pd.concat([df, novel_df], ignore_index=True)

    # --- Step 6: Join LC-MS/MS evidence columns ---
    for new_col in _EV_COLS.values():
        df[new_col] = np.nan

    if len(lcms_pep_df) > 0:
        ev = lcms_pep_df.drop_duplicates(subset="sequence").set_index("sequence")
        for old_col, new_col in _EV_COLS.items():
            if old_col in ev.columns:
                df[new_col] = df["peptide"].map(ev[old_col])

    # --- Step 7: Wipe evidence for decoys (symmetric TDC requirement) ---
    df["is_decoy"] = df["is_decoy"].astype(bool)
    decoy_mask = df["is_decoy"].values
    for new_col in _EV_COLS.values():
        df.loc[decoy_mask, new_col] = np.nan

    n_confirmed = (df["source"] == "lcms_confirmed").sum()
    n_digest = (df["source"] == "protein_digest").sum()
    n_decoy = (df["source"] == "decoy").sum()
    logger.info(
        f"Strategy C candidates: {n_confirmed} lcms_confirmed + "
        f"{n_digest} protein_digest + {n_decoy} decoy"
    )
    return df
