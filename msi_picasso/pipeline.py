"""End-to-end symmetric MALDI-MSI rescoring pipeline."""

import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from msi_picasso.candidates import (
    generate_substitution_candidates,
    target_rows,
    digest_fasta,
    digest_identified_proteins,
    generate_mz_shuffle_candidates,
    match_to_maldi_features,
)
from msi_picasso.feature_generator import (
    FEATURE_NAN_FILL,
    MAIN_FEATURES,
    MALDI_INTRINSIC_FEATURES,
    MOB_QUALITY_FEATURES,
    MZ_SHUFFLE_MASSNORM_ISOTOPE_FEATURES,
    COSINE_COLOCALIZATION_FEATURES,
    PROTEIN_LEVEL_FEATURES,
    SPATIAL_RANKER_FEATURES,
    compute_all_features,
)
from msi_picasso.utils import values_at_mz

logger = logging.getLogger(__name__)

# Features that use the candidate's PREDICTED CCS/mobility to gate or compare against
# the observed feature. For mz_shuffle (peptide relocated far in mass; CCS/1-K0 ∝ m/z)
# these leak the m/z baseline rather than testing identity, so they are dropped from
# the ranker.
#
# This used to be an enumerated list covering only the isotope_*_mob and adduct_*_mob
# families, and it went stale: PROGRESS.md F-016 measured protein_colocalization_mob_max at
# target/decoy AUC 0.0003 on kidney E005 (top-importance ranker feature, 5070/5070 targets
# passing at 1% FDR), plus fraction_detected_mob, log_mean_intensity_mob, spatial_morans_i_mob
# and intensity_cv_mob at 0.001-0.010. Mechanism: a mz_shuffle decoy keeps its own predicted
# CCS but sits on a feature at an unrelated m/z, so the predicted 1/K0 gate selects an empty
# slice and every mobility-gated column becomes a label proxy by construction.
#
# The same measurement also refutes the old "keep only the m/z-detrended *_resid CCS
# features" rule: im2deep_ccs_rank_resid and im2deep_abs_delta_ccs_pct_resid measured AUC
# 0.76-0.86. Detrending removes a linear m/z trend, but CCS-vs-m/z is not linear and
# mz_shuffle relocates decoys far enough in mass that the residual still carries the baseline.
#
# So the rule is now BY CONSTRUCTION rather than by enumeration: under mz_shuffle, exclude
# every column that is either mobility-gated or derived from predicted CCS. Matching on
# suffix/prefix means a newly added mobility-gated feature is covered on the day it is added,
# which an explicit list demonstrably does not achieve.
_MZ_SHUFFLE_LEAK_SUFFIXES = ("_mob", "_mob_max", "_mob_n_partners")
_MZ_SHUFFLE_LEAK_PREFIXES = ("im2deep_",)

# Retained only as a documented floor, so the by-construction rule can be asserted to be a
# superset of what was previously excluded. Not used for matching.
_MZ_SHUFFLE_CCS_LEAK_FEATURES = frozenset([
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank",
    "isotope_colocalization_m1_mob", "isotope_colocalization_m2_mob",
    "isotope_colocalization_mean_mob",
    "adduct_colocalization_na_mob", "adduct_colocalization_k_mob",
    "adduct_colocalization_chca_mob",
])


def _mz_shuffle_leaking_features(columns) -> set[str]:
    """Columns that leak the m/z baseline under ``mz_shuffle`` decoys.

    Any mobility-gated column (a predicted-1/K0 gate applied to a decoy sitting on an
    unrelated m/z selects an empty slice), any predicted-CCS-derived column (``*_resid``
    variants included), or any own-mass-vs-observed-envelope isotope feature (see
    ``maldi_features.MZ_SHUFFLE_OWN_MASS_ENVELOPE_FEATURES``: theo_isotope_kl and siblings
    build their theoretical envelope from the candidate's own mass and compare it against
    the observed envelope at the assigned feature, which under mz_shuffle's mass-sorted
    derangement is deliberately far away — PROGRESS.md F-020 measured Spearman 0.70-0.80
    between that mass gap and the isotope-feature difference within a co-located pair,
    confirmed on all three ground-truth datasets and via a label-permutation test). See the
    module comment above and PROGRESS.md F-016/F-020.
    """
    from msi_picasso.maldi_features import MZ_SHUFFLE_OWN_MASS_ENVELOPE_FEATURES

    columns = set(columns)
    return (
        {
            c for c in columns
            if c.endswith(_MZ_SHUFFLE_LEAK_SUFFIXES) or c.startswith(_MZ_SHUFFLE_LEAK_PREFIXES)
        }
        | (columns & MZ_SHUFFLE_OWN_MASS_ENVELOPE_FEATURES)
    )


def _observed_ccs_by_feature_idx(
    candidates: pd.DataFrame,
    maldi_mzs: np.ndarray,
    ccs_arr: np.ndarray,
) -> dict | None:
    """Build an ``observed_ccs_per_feature`` dict keyed by candidate ``feature_idx``.

    ``ccs_arr`` is aligned with ``maldi_mzs`` (the queried m/z grid).  In raw-query
    mode a candidate's ``feature_idx`` indexes the digest grid, not ``maldi_mzs``,
    so the mapping is bridged via ``feature_mz`` (1:1 with ``feature_idx``).  This
    matches how ``compute_im2deep_features`` consumes the dict
    (``df["feature_idx"].map(observed_ccs_per_feature)``).  Non-finite CCS values
    are dropped.  Returns ``None`` when no feature has a finite CCS, so downstream
    ``is not None`` guards behave like the no-mobility path.
    """
    pairs = candidates[["feature_idx", "feature_mz"]].drop_duplicates()
    ccs = values_at_mz(ccs_arr, maldi_mzs, pairs["feature_mz"])
    found = np.isfinite(ccs)
    ccs_map = {
        int(fi): float(c) for fi, c in zip(pairs["feature_idx"].to_numpy()[found], ccs[found])
    }
    return ccs_map or None


def _recompute_ppm_from_centroids(
    feature_mz: np.ndarray,
    maldi_mzs: np.ndarray,
    centroid_mz: np.ndarray,
    worst_case_ppm: float | None = None,
) -> np.ndarray:
    """Symmetric raw-query ``ppm_error``: observed peak centroid vs the candidate anchor.

    In raw-query mode every candidate (target or decoy) is matched against the
    *theoretical* digest grid, so the usual ``(feature_mz - mh_mz)`` ppm is 0 by
    construction and decoys inherit 0.  Instead, for each candidate row compute the
    mass accuracy of the observed peak in its own extraction window:
    ``(observed_centroid - feature_mz) / feature_mz * 1e6``, where ``feature_mz`` is
    the candidate's queried anchor (the peptide's own [M+H]+ for a target or a
    substitution decoy).  Identical treatment for targets and decoys (no
    inheritance), bounded by ±extraction_ppm, and non-leaking (it never references
    the peptide mass for a decoy).

    A window with no observed peak has unmeasurable mass accuracy.  When
    ``worst_case_ppm`` is given, such rows are set to that worst-case value (the
    extraction window edge — a real peak's centroid is always within
    ±extraction_ppm of the anchor, so this is the worst in-distribution value) so
    empty-signal candidates (e.g. decoys whose m/z lands in empty space)
    are penalised on ppm rather than median-imputed to an average value.  When
    ``worst_case_ppm`` is ``None`` those rows are left ``NaN``.

    ``centroid_mz`` is aligned with ``maldi_mzs``; the result is aligned with the
    per-row ``feature_mz`` input.
    """
    fmz = np.asarray(feature_mz, dtype=np.float64)
    obs = values_at_mz(centroid_mz, maldi_mzs, fmz)
    with np.errstate(invalid="ignore"):
        ppm = (obs - fmz) / fmz * 1e6
    if worst_case_ppm is not None:
        ppm = np.where(np.isfinite(ppm), ppm, float(worst_case_ppm))
    return ppm


# Protein colocalization features are Pearson r aggregates (higher = more target-like
# protein co-distribution). A candidate whose MALDI feature has *no signal* (constant
# ion image) has an undefined correlation -> NaN, which the scoring imputer would fill
# with the column median, i.e. an *average* coloc value, silently rewarding a
# zero-evidence candidate. Mirror the ppm worst-case fill in _recompute_ppm_from_centroids:
# set those NaNs to the worst in-distribution value (the pooled finite minimum) so a
# no-signal candidate is penalised on coloc rather than imputed up to average.
_PROTEIN_COLOC_WORST_PREFIXES = ("protein_colocalization",)


def _fill_nosignal_coloc_worst_case(features_df: pd.DataFrame) -> pd.DataFrame:
    """Worst-case fill of protein-colocalization NaNs for zero-signal candidates only.

    Symmetry: the no-signal mask is read from ``feature_intensity_sum`` (the ion image
    alone, an ``is_decoy``-blind quantity that a co-located target/decoy pair share under
    mz_shuffle), and the fill constant is the pooled finite minimum over all candidates,
    so both the mask and the value are label-blind and no target/decoy asymmetry is
    introduced. Only no-signal rows are touched; NaNs from a single-feature protein (no
    within-protein partner) are left for the downstream median imputer, since those are
    "coloc undefined", not "no evidence".
    """
    if "feature_intensity_sum" not in features_df.columns:
        return features_df
    no_signal = ~(features_df["feature_intensity_sum"] > 0)  # True for 0, NaN, negative
    if not no_signal.any():
        return features_df
    cols = [
        c
        for c in features_df.columns
        if c.startswith(_PROTEIN_COLOC_WORST_PREFIXES) and not c.endswith("_n_partners")
    ]
    n_filled = 0
    for c in cols:
        finite = features_df[c][np.isfinite(features_df[c])]
        if finite.empty:
            continue
        worst = float(finite.min())
        fill_mask = no_signal & ~np.isfinite(features_df[c])
        n = int(fill_mask.sum())
        if n:
            features_df.loc[fill_mask, c] = worst
            n_filled += n
    if cols:
        logger.info(
            f"  No-signal coloc fill: set {n_filled} NaN entries across {len(cols)} "
            f"protein-colocalization columns to the worst-case (pooled min) for "
            f"{int(no_signal.sum())} zero-signal candidates (symmetric, is_decoy-blind)."
        )
    return features_df


def _apply_nan_fill(
    X: np.ndarray,
    feature_names: list[str],
    fill_spec: dict[str, "float | str"],
) -> np.ndarray:
    """Apply feature-specific NaN fills before the generic median imputer.

    Operates in-place on *X* (caller should pass a copy if the original must
    be preserved).  Only columns present in both *feature_names* and
    *fill_spec* are touched; remaining NaN values are left for the downstream
    ``SimpleImputer`` (or ``fillna``) to handle.
    """
    for fname, fill_val in fill_spec.items():
        if fname not in feature_names:
            continue
        j = feature_names.index(fname)
        col = X[:, j]
        nan_mask = np.isnan(col)
        n_nan = int(nan_mask.sum())
        if n_nan == 0:
            continue
        if isinstance(fill_val, str):
            if fill_val == "col_max":
                value = float(np.nanmax(col)) if np.isfinite(col[~nan_mask]).any() else 0.0
            elif fill_val == "col_min":
                value = float(np.nanmin(col)) if np.isfinite(col[~nan_mask]).any() else 0.0
            else:
                raise ValueError(f"Unknown fill_spec value {fill_val!r} for feature {fname!r}")
        else:
            value = float(fill_val)
        col[nan_mask] = value
        logger.debug(
            "  _apply_nan_fill: %s — filled %d NaN with %.4f (%s)",
            fname, n_nan, value, fill_val if isinstance(fill_val, str) else "constant",
        )
    return X


def _log_imputation_debug(
    label: str,
    X_fit: np.ndarray,
    fit_names: list[str],
    is_target: np.ndarray,
    is_decoy: np.ndarray,
    pipe,
) -> None:
    """Log per-feature NaN counts and imputation values, split by target/decoy.

    Gated on DEBUG level so it is a no-op in normal runs.

    Columns:
      nan_tgt / nan_dec  — how many rows of each group had NaN and were imputed
      tgt% / dec%        — NaN rate per group
      imputed            — value filled in (train-set median from SimpleImputer)
      tgt_med            — median of real (non-NaN) target values
      dec_med            — median of real (non-NaN) decoy values
      bias               — imputed − dec_med  (positive = decoys with NaN got
                           pulled above their natural median toward target territory)

    A large positive bias on a feature with high dec% is a sign that imputation
    may be inflating decoy scores for that feature.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    nan_mask = np.isnan(X_fit)
    has_nan = nan_mask.any(axis=0)
    if not has_nan.any():
        logger.debug("  %s imputation: no NaN values in any feature", label)
        return

    n_tgt = int(is_target.sum())
    n_dec = int(is_decoy.sum())
    imp_vals = pipe["imputer"].statistics_

    header = (
        f"  {'feature':<42s}  {'nan_tgt':>7}  {'nan_dec':>7}"
        f"  {'tgt%':>6}  {'dec%':>6}  {'imputed':>9}  {'tgt_med':>9}  {'dec_med':>9}  {'bias':>9}"
    )
    rows = [header]
    for j, fname in enumerate(fit_names):
        if not has_nan[j]:
            continue
        n_nan_tgt = int(nan_mask[is_target, j].sum())
        n_nan_dec = int(nan_mask[is_decoy, j].sum())
        pct_tgt = 100.0 * n_nan_tgt / n_tgt if n_tgt else 0.0
        pct_dec = 100.0 * n_nan_dec / n_dec if n_dec else 0.0

        tgt_real = X_fit[is_target & ~nan_mask[:, j], j]
        dec_real = X_fit[is_decoy & ~nan_mask[:, j], j]
        tgt_med = float(np.median(tgt_real)) if len(tgt_real) else float("nan")
        dec_med = float(np.median(dec_real)) if len(dec_real) else float("nan")
        bias = imp_vals[j] - dec_med if np.isfinite(dec_med) else float("nan")

        rows.append(
            f"  {fname:<42s}  {n_nan_tgt:>7d}  {n_nan_dec:>7d}"
            f"  {pct_tgt:>5.1f}%  {pct_dec:>5.1f}%  {imp_vals[j]:>9.3f}"
            f"  {tgt_med:>9.3f}  {dec_med:>9.3f}  {bias:>+9.3f}"
        )

    logger.debug(
        "  %s imputation stats (n_tgt=%d, n_dec=%d, imputed=train-set median):\n%s",
        label, n_tgt, n_dec, "\n".join(rows),
    )


# Features excluded from best-feature initialization.  These measure amino acid
# composition rather than spectral quality.  Since decoys are K/R-preserving
# shuffles from the same protein pool, composition features can have arbitrary
# systematic differences that produce spurious pseudo-positives.
_BEST_FEAT_SKIP: frozenset[str] = frozenset({
    # Basic sequence properties
    "peptide_length", "n_missed_cleavages",
    # Peptide composition (C-group)
    "has_oxidized_met", "has_cys", "n_proline", "acidic_residue_density",
    # Ionization / physicochemistry
    "n_arginine", "n_basic_residues", "n_aromatic", "gravy_score", "charge_proxy",
})


def _encode_labels(is_decoy, positive_mask):
    """Three-valued semi-supervised labels: -1 for decoys, +1 for positive
    targets (``positive_mask`` True), 0 for unlabelled targets. int8."""
    return np.where(
        is_decoy, np.int8(-1),
        np.where(positive_mask, np.int8(1), np.int8(0)),
    ).astype(np.int8)


def _seed_pass_mask(
    scores: np.ndarray,
    is_decoy: np.ndarray,
    init_fdr: float,
    protein: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Positive-target mask and its size for one seed-search scoring attempt.

    Without ``protein`` this is the original row-level TDC count: q-values over
    every row, positives are targets at ``q <= init_fdr``.

    With ``protein`` (H-fdr-11, from F-051): two of the six allowlisted seed
    features are exact per-protein constants and the rest are heavily blocked
    (a mean of 3.6-5.5 distinct values per protein against ~10 rows/protein), so
    row-level TDC lets one lucky protein's entire row count pass as if each row
    were independent evidence -- F-051 traced a reported 255-target seed back to
    4 distinct proteins. This rolls TDC up to one row per protein first: for each
    (protein, decoy) group, keep only the best-scoring row under *this* score
    (the same best-of-k rule ``_peptide_level_qvalues`` uses for F-029's
    peptide-level rollup, one level up -- picked fresh per score, since the best
    row for one feature need not be the best row for another), then run TDC over
    just those representative rows.

    This departs from ``_peptide_level_qvalues`` in one place, deliberately:
    that function broadcasts a peptide's rolled-up q-value back onto every one
    of its matching rows, because its result is a final report (which rows to
    print) and every match is legitimately part of the answer. Here the result
    is a training label fed straight to the semi-supervised loop, so
    broadcasting a passing protein's q-value onto its other rows would hand the
    classifier the exact per-protein row multiplicity this rollup exists to
    remove. Only the single representative row of a passing protein is marked
    positive; its other rows get label 0 (unlabeled, still scored).

    Returns ``(positive_mask, n_pass)``, both counted at the protein level when
    ``protein`` is given.
    """
    scores = np.asarray(scores)
    is_decoy = np.asarray(is_decoy, dtype=bool)
    if protein is None:
        q = _tdc_qvalues(scores, is_decoy)
        mask = (~is_decoy) & (q <= init_fdr)
        return mask, int(mask.sum())

    frame = pd.DataFrame({"score": scores, "decoy": is_decoy, "protein": np.asarray(protein)})
    rep_pos = frame.groupby(["protein", "decoy"])["score"].idxmax().to_numpy()
    q_rep = _tdc_qvalues(scores[rep_pos], is_decoy[rep_pos])
    passing = rep_pos[(~is_decoy[rep_pos]) & (q_rep <= init_fdr)]
    mask = np.zeros(len(scores), dtype=bool)
    mask[passing] = True
    return mask, len(passing)


def _find_best_feature_labels(
    X: np.ndarray,
    is_decoy: np.ndarray,
    feature_names: list[str],
    init_fdr: float = 0.2,
    min_seed_positives: int = 50,
    seed_features: list[str] | None = None,
    protein: np.ndarray | None = None,
) -> tuple[np.ndarray, str, int] | None:
    """
    Mokapot-style best-feature seed initialization.

    For each column in X and each ranking direction (ascending/descending),
    run TDC q-value computation and count target candidates at q <= init_fdr.
    Select the (feature, direction) pair that yields the most such targets.

    When the best single-feature result has fewer than ``min_seed_positives``
    targets, all unique pairwise sums and differences of eligible features are
    tried. The composite score that yields the most targets is used if it beats
    the single-feature result.

    When the pairwise result is *still* below ``min_seed_positives``, a shallow
    decision tree (depth 3, target vs. decoy) is fit on all eligible features at
    once, so weakly-correlated evidence can combine beyond what raw sums/pairs
    reach. Its leaf-probability score is used if it beats the pairwise result.

    Columns whose names appear in _BEST_FEAT_SKIP are excluded — they measure
    amino acid composition rather than spectral quality and can yield spurious
    pseudo-positives due to composition differences between shuffled decoys and
    targets.

    When ``seed_features`` is a non-empty list, the single-feature, pairwise, and
    tree sweeps are restricted to *only* those feature names (still intersected
    with the _BEST_FEAT_SKIP guard). This lets a run seed from a chosen,
    tissue-independent subset (e.g. CCS-error / ppm / isotope features) instead of
    whatever scores marginally highest — useful when the otherwise-dominant
    colocalization features are non-discriminative (heterogeneous tissue). ``None``
    or ``[]`` (default) uses every eligible feature, i.e. unchanged behaviour.

    NaN values in X are filled with the column median before ranking.

    ``protein`` (H-fdr-11): when given, every sweep below counts passes via
    ``_seed_pass_mask``'s per-protein rollup instead of raw rows -- see that
    function. ``None`` (default) reproduces the original row-level counting
    exactly, which existing callers rely on.

    Returns
    -------
    (labels, best_feature_name, n_passing) or None when n_passing == 0.
        labels: int8 array aligned to the X rows.
            +1  — pseudo-positive: representative row of a protein (or, without
                  ``protein``, a row) at q <= init_fdr under the best score
            -1  — pseudo-negative: decoy
             0  — excluded: q > init_fdr, or a non-representative row of a
                  passing protein
    """
    is_decoy = np.asarray(is_decoy, dtype=bool)

    # Optional allowlist: restrict seeding to a chosen feature subset (R2).
    _seed_allow = set(seed_features) if seed_features else None
    if _seed_allow is not None:
        _matched = [f for f in feature_names if f in _seed_allow and f not in _BEST_FEAT_SKIP]
        if _matched:
            logger.info("  Seed restricted to %d feature(s): %s", len(_matched), ", ".join(_matched))
        else:
            logger.warning(
                "  seed_features matched no eligible ranker features (%s); "
                "falling back to unrestricted seeding.", ", ".join(sorted(_seed_allow)),
            )
            _seed_allow = None

    def _seed_eligible(fname: str) -> bool:
        if fname in _BEST_FEAT_SKIP:
            return False
        return _seed_allow is None or fname in _seed_allow

    X_imp = np.where(np.isfinite(X), X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
        col_median = np.nanmedian(X_imp, axis=0)
    X_imp = np.where(np.isnan(X_imp), np.nan_to_num(col_median, nan=0.0), X_imp)

    # Sub-ULP random noise to break ties in argsort.  A stable sort on a feature
    # with many tied values preserves the original DataFrame row order, which
    # places targets before decoys (digest_fasta output order) and assigns them
    # artificially low q-values.
    rng = np.random.default_rng(0)
    tiebreak = rng.uniform(-1e-9, 1e-9, X_imp.shape[0])

    best_n = 0
    best_j = -1
    best_asc = True
    best_mask: np.ndarray | None = None

    for j, fname in enumerate(feature_names):
        if not _seed_eligible(fname):
            continue
        col = X_imp[:, j]
        if col.std() == 0.0:
            continue
        for ascending in (True, False):
            scores = (col if ascending else -col) + tiebreak
            mask, n_pass = _seed_pass_mask(scores, is_decoy, init_fdr, protein=protein)
            if n_pass > best_n:
                best_n = n_pass
                best_j = j
                best_asc = ascending
                best_mask = mask

    result: tuple[np.ndarray, str, int] | None = None
    if best_j >= 0:
        assert best_mask is not None
        result = (_encode_labels(is_decoy, best_mask), feature_names[best_j], best_n)

    eligible = [
        j for j, fname in enumerate(feature_names)
        if _seed_eligible(fname) and X_imp[:, j].std() > 0
    ]

    # --- Pairwise sweep when single-feature result is weak ---
    if best_n < min_seed_positives and eligible:
        # Scale to zero mean, unit variance so both features contribute equally.
        col_means = X_imp[:, eligible].mean(axis=0)
        col_stds = X_imp[:, eligible].std(axis=0)
        col_stds[col_stds == 0] = 1.0
        X_sc = (X_imp[:, eligible] - col_means) / col_stds

        pair_best_n = best_n
        for ii in range(len(eligible)):
            for jj in range(ii + 1, len(eligible)):
                gi, gj = eligible[ii], eligible[jj]
                for sign in (+1, -1):
                    composite = X_sc[:, ii] + sign * X_sc[:, jj]
                    for ascending in (True, False):
                        scores = (composite if ascending else -composite) + tiebreak
                        mask, n_pass = _seed_pass_mask(scores, is_decoy, init_fdr, protein=protein)
                        if n_pass > pair_best_n:
                            pair_best_n = n_pass
                            sign_str = "+" if sign == +1 else "-"
                            pair_best_name = (
                                f"{feature_names[gi]} {sign_str} {feature_names[gj]}"
                            )
                            result = (_encode_labels(is_decoy, mask), pair_best_name, n_pass)

        if pair_best_n > best_n:
            logger.info(
                "  Selected pair (%s) with %d PSMs at q<=%g",
                result[1], pair_best_n, init_fdr,
            )
            best_n = pair_best_n

    # --- Shallow-tree sweep when the pairwise result is still weak ---
    # A depth-3 tree on all eligible spectral-quality features at once lets
    # weakly-correlated evidence combine beyond what raw sums/pairs reach.
    is_target = ~is_decoy
    if best_n < min_seed_positives and eligible and is_decoy.any() and is_target.any():
        from sklearn.tree import DecisionTreeClassifier

        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)
        tree.fit(X_imp[:, eligible], is_target.astype(int))
        scores = tree.predict_proba(X_imp[:, eligible])[:, 1] + tiebreak
        mask, n_pass = _seed_pass_mask(scores, is_decoy, init_fdr, protein=protein)
        if n_pass > best_n:
            tree_name = f"tree(depth=3, n_features={len(eligible)})"
            logger.info(
                "  Selected shallow-tree seed (%s) with %d PSMs at q<=%g",
                tree_name, n_pass, init_fdr,
            )
            result = (_encode_labels(is_decoy, mask), tree_name, n_pass)
            best_n = n_pass
        else:
            logger.info(
                "  Shallow-tree seed (depth=3, %d features) gave %d PSMs at q<=%g — "
                "did not beat pairwise result (%d), keeping pairwise",
                len(eligible), n_pass, init_fdr, best_n,
            )

    if result is None or result[2] == 0:
        return None
    return result


def _find_best_feature_labels_escalating(
    X: np.ndarray,
    is_decoy: np.ndarray,
    feature_names: list[str],
    init_fdr: float,
    min_seed_positives: int = 50,
    seed_features: list[str] | None = None,
    protein: np.ndarray | None = None,
    escalate: bool = False,
    step: float = 0.005,
    ceiling: float = 0.5,
) -> tuple[tuple[np.ndarray, str, int] | None, float]:
    """``_find_best_feature_labels`` with training-FDR escalation (H-fdr-2).

    If ``init_fdr`` yields nothing (``None``), retries at increasing thresholds
    (steps of ``step``, capped at ``ceiling``) until a non-empty seed is found
    or the ceiling is reached. Freestone et al. (2025) escalate from 0.01 in
    steps of 0.005; here it escalates from whichever ``init_fdr`` was configured.
    A no-op when ``escalate`` is False or the configured threshold already succeeds.

    Returns ``(bf_result, fdr_used)`` — ``fdr_used`` equals ``init_fdr`` unless
    escalation actually fired.
    """
    fdr = init_fdr
    result = _find_best_feature_labels(
        X, is_decoy, feature_names, fdr,
        min_seed_positives=min_seed_positives, seed_features=seed_features,
        protein=protein,
    )
    while result is None and escalate and fdr < ceiling:
        fdr += step
        result = _find_best_feature_labels(
            X, is_decoy, feature_names, fdr,
            min_seed_positives=min_seed_positives, seed_features=seed_features,
            protein=protein,
        )
    return result, fdr


def _make_fold_ids(
    is_decoy: np.ndarray, cv_folds: int, random_state: int = 0
) -> np.ndarray | None:
    """Stratified fold assignment for out-of-fold scoring.

    Returns an int array (row → fold) or ``None`` when there are too few targets
    or decoys for ``cv_folds``-fold CV (caller then scores in-sample).

    ``random_state`` was fixed at 0 until F-030 measured how much rides on it:
    twelve values gave kidney anywhere from 0 to 63 peptides at 5% FDR with
    nothing else changed. It is now the replicate index under ``model_repeats``
    (see ``_rescore_linear``); 0 reproduces every result predating that.
    """
    is_decoy = np.asarray(is_decoy, dtype=bool)
    n = len(is_decoy)
    if int(is_decoy.sum()) < cv_folds * 2 or int((~is_decoy).sum()) < cv_folds * 2:
        return None
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_ids = np.empty(n, dtype=np.int64)
    for k, (_, test) in enumerate(skf.split(np.zeros(n), is_decoy.astype(int))):
        fold_ids[test] = k
    return fold_ids


def _cv_semisup_scores(X_fit, labels, fold_ids, make_pipe):
    """Semi-supervised scores with **out-of-fold** cross-validation.

    Trains a discriminant on ``label==1`` (positives) vs ``label==-1`` (decoys);
    ``label==0`` rows (unlabelled targets) are scored but never trained on.

    Returns ``(scores, pipe_full)``:
    - ``scores`` — when ``fold_ids`` is given, each row is scored by a model trained
      on the *other* folds' pos/neg rows (no row is scored by a model that trained
      on it), so the discriminant cannot manufacture target/decoy separation by
      overfitting.  ``None`` fold_ids (too few pos/neg) → in-sample scores.
    - ``pipe_full`` — a model fit on ALL current pos/neg rows, used only for
      reporting feature importances / structure coefficients (never for FDR).
    """
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == -1
    train = pos | neg
    pipe_full = make_pipe()
    pipe_full.fit(X_fit[train], pos[train].astype(float))
    if fold_ids is None:
        return pipe_full.decision_function(X_fit).ravel(), pipe_full

    oof = np.full(len(labels), np.nan)
    for k in np.unique(fold_ids):
        test = fold_ids == k
        tr = train & ~test
        ytr = pos[tr].astype(float)
        # Need both classes in the training partition; otherwise fall back in-sample.
        if ytr.sum() < 1 or (len(ytr) - ytr.sum()) < 1:
            return pipe_full.decision_function(X_fit).ravel(), pipe_full
        p = make_pipe()
        p.fit(X_fit[tr], ytr)
        oof[test] = p.decision_function(X_fit[test]).ravel()
    if not np.isfinite(oof).all():
        return pipe_full.decision_function(X_fit).ravel(), pipe_full
    return oof, pipe_full


# How many replicate fits contribute to the reported permutation importance when
# model_repeats > 1. See _rescore_linear for why this is not all of them.
_PERM_IMPORTANCE_REPLICATES = 5


def _permutation_importance(
    pipe, X_fit, n_repeats: int = 5, random_state: int = 0, max_samples: int = 5000
):
    """How much of the score's ranking one feature carries (H-model-2, F-041).

    A structure coefficient says how strongly a feature *correlates* with the
    score. A feature can correlate without contributing (it moves with one that
    does) and contribute without correlating (its effect is conditional on
    another feature), so it is not an attribution. Shuffling a column and
    re-scoring measures the contribution directly, and it needs nothing from the
    estimator but ``decision_function`` — which is what the kernel backends need,
    since they have no ``coef_``.

    Returned value per feature: ``1 - spearman(score with that column shuffled,
    the unshuffled score)``, averaged over ``n_repeats`` shuffles. 0 means the
    ranking is unchanged, so the model does not use the feature; 1 means the
    ranking is destroyed.

    Scored against the model's own output rather than against the labels on
    purpose. The obvious alternative, the drop in target-versus-pseudo-positive
    ROC AUC, is saturated here: the fitted SVC separates its own training rows at
    AUC 0.9995-1.0000 on all three datasets, and with 22 pseudo-positives against
    6245 decoys (kidney) the remaining features still reach 1.0 after any single
    column is shuffled, so that measure returns exactly 0 for almost every
    feature. The ranking is also the quantity that decides the identification
    count, which the training-set AUC is not.

    Reporting only: these values never enter a score or an FDR estimate. Rows are
    subsampled to ``max_samples`` to bound the cost, which is
    ``n_features * n_repeats`` scoring passes.
    """
    from scipy.stats import spearmanr
    from sklearn.inspection import permutation_importance

    def _agreement(est, Xq, yq):
        return float(spearmanr(est.decision_function(Xq), yq).statistic)

    r = permutation_importance(
        pipe, X_fit, pipe.decision_function(X_fit), scoring=_agreement,
        n_repeats=n_repeats, random_state=random_state,
        max_samples=min(max_samples, len(X_fit)),
    )
    return r.importances_mean


def _rescore_linear_once(
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    init_ppm_threshold: float,
    seed_mask: np.ndarray | None = None,
    init_fdr: float = 0.2,
    train_fdr: float = 0.05,
    max_iter: int = 5,
    r1_seed_percentile: float = 0.10,
    min_seed_positives: int = 50,
    seed_features: list[str] | None = None,
    cv_folds: int = 3,
    fold_seed: int = 0,
    make_clf=None,
    clf_name: str = "lda",
    fitted_out: dict | None = None,
    train_fdr_escalate: bool = False,
    pseudo_label_growth_cap: float | None = None,
    perm_importance: bool = True,
) -> np.ndarray:
    """
    Semi-supervised rescoring on MALDI-intrinsic features with a
    ``decision_function``-based classifier (LDA by default).
    ``make_clf`` is a zero-arg factory returning the final pipeline estimator
    (see ``_estimator_factory``); ``clf_name`` is its pipeline step key and the
    user-facing log/importance tag.

    Pre-processing: ±inf replaced with NaN, then median imputation and
    StandardScaler inside a sklearn Pipeline.

    Seed (when ``seed_mask`` is None): calls ``_find_best_feature_labels``
    to pick the single feature that yields the most targets at q <= ``train_fdr``.
    Falls back to ppm_error_abs < ``init_ppm_threshold`` OR n_candidates == 1
    (with a top-percentile fallback) if no feature yields any targets.

    Explicit seed (``seed_mask`` provided): the boolean mask is converted to the
    same three-valued label scheme (+1 / -1 / 0) for the iteration loop.

    Pseudo-label iteration (up to ``max_iter`` rounds): trains on +1 vs -1 rows
    only (label-0 targets are excluded from training but still scored); updates
    labels by running TDC q-values on the new scores and marking targets with
    q <= ``train_fdr`` as +1. Stops when the positive count changes by < 1%.

    Scoring is **cross-validated** (``cv_folds`` out-of-fold splits, stratified by
    ``is_decoy``): at each iteration every candidate is scored by a model trained
    on the other folds, so the LDA cannot manufacture target/decoy separation by
    overfitting (which would make the TDC FDR anti-conservative).  Falls back to
    in-sample scoring only when there are too few targets/decoys for CV.  The
    returned importances/structure coefficients come from a model fit on all
    pos/neg rows (reporting only — never used for the FDR scores).

    ``train_fdr_escalate`` (H-fdr-2, opt-in): if the seed search or an iteration
    would otherwise yield zero pseudo-positives at the configured ``init_fdr``/
    ``train_fdr``, retries at increasing thresholds (steps of 0.005, capped at
    0.5) until a non-empty discovery set is found or the cap is reached, instead
    of falling back to the weaker ppm-based heuristic or stopping early. A no-op
    whenever the configured threshold already succeeds.

    ``perm_importance`` (H-model-2): report permutation importance rather than
    ``|structure coefficient|`` when the estimator exposes neither ``coef_`` nor
    ``feature_importances_``. See ``_permutation_importance``. Costs one refit-free
    pass of ``5 x n_features`` scoring calls (3 to 68 s on the three datasets), so
    ``_rescore_linear`` asks only the first few replicates for it.

    ``pseudo_label_growth_cap`` (H-fdr-5, opt-in): if an iteration's pseudo-
    positive count exceeds this multiple of the initial seed size, the loop
    stops *without* accepting that iteration's label update — the returned
    scores come from the model trained on the last iteration within the cap.
    Guards against the self-training loop amplifying a leak or a noisy seed
    into a runaway positive set (F-020's amplification mechanism, independent
    of whether the seed itself is trustworthy). ``None`` (default) disables it,
    reproducing the unbounded-growth behaviour exactly.

    Returns ``(scores, importances, feature_names_used)``.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _tag = clf_name.upper()
    if make_clf is None:
        make_clf = _estimator_factory("lda")

    df = features_df.reset_index(drop=True)
    present = [f for f in intrinsic_feature_names if f in df.columns]
    X_raw = df[present].values.astype(np.float64)
    X = np.where(np.isfinite(X_raw), X_raw, np.nan)  # ±inf → nan for imputer
    _apply_nan_fill(X, present, FEATURE_NAN_FILL)

    is_decoy = df["is_decoy"].values.astype(bool)
    is_target = ~is_decoy
    protein = df["protein"].to_numpy() if "protein" in df.columns else None

    # --- Initial label assignment ---
    n_init_positives: int | None = None  # for post-loop comparison

    if seed_mask is None:
        bf_result, _bf_fdr = _find_best_feature_labels_escalating(
            X, is_decoy, present, init_fdr, min_seed_positives=min_seed_positives,
            seed_features=seed_features, protein=protein,
            escalate=train_fdr_escalate,
        )
        if bf_result is not None:
            labels, _best_feat, _n_init = bf_result
            n_init_positives = _n_init
            _esc_note = f" (escalated from {init_fdr:.3g})" if _bf_fdr != init_fdr else ""
            logger.info(
                f"  {_tag}: best-feature init on '{_best_feat}', "
                f"{_n_init} targets at q≤{_bf_fdr:.3g}{_esc_note}"
            )
        else:
            logger.warning(
                f"  {_tag}: best-feature init yielded 0 targets at q≤{init_fdr:.3g} "
                "— falling back to ppm-based seeding"
            )
            ppm_col = df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
            n_cand_col = df.get("n_candidates", pd.Series(np.inf, index=df.index))
            init_mask = (
                is_target & ((ppm_col < init_ppm_threshold) | (n_cand_col == 1))
            ).values
            if not init_mask.any():
                logger.warning(f"  {_tag}: no ppm-based seed positives — falling back to top-ppm init")
                init_mask = (
                    is_target & (ppm_col < ppm_col[is_target].quantile(r1_seed_percentile))
                ).values
            labels = _encode_labels(is_decoy, init_mask)
    else:
        seed_arr = seed_mask.values if hasattr(seed_mask, "values") else np.asarray(seed_mask)
        labels = _encode_labels(is_decoy, seed_arr)

    n_seed = int((labels == 1).sum())
    logger.info(f"  {_tag}: seed positives = {n_seed}, decoys = {is_decoy.sum()}")

    def _make_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (clf_name, make_clf()),
        ])

    # Out-of-fold cross-validation prevents the semi-supervised LDA from
    # manufacturing target/decoy separation by overfitting (each candidate is
    # scored by a model trained on other folds).  Folds are fixed and stratified
    # by is_decoy; falls back to in-sample scoring if there are too few pos/neg.
    fold_ids = _make_fold_ids(is_decoy, cv_folds, fold_seed)
    if fold_ids is None:
        logger.warning(
            f"  {_tag}: too few targets/decoys for {cv_folds}-fold CV — scoring "
            "in-sample (overfitting risk)"
        )
    else:
        logger.info(f"  {_tag}: {cv_folds}-fold cross-validated (out-of-fold) scoring")

    scores = np.zeros(len(df))
    prev_pos_size = -1
    pipe = None
    from threadpoolctl import threadpool_limits

    for iteration in range(max_iter):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == -1)[0]

        if len(pos_idx) == 0:
            logger.warning(f"  {_tag} iter {iteration + 1}: no positives — stopping early")
            break

        if len(neg_idx) == 0:
            logger.warning(f"  {_tag} iter {iteration + 1}: no negatives (decoys) — cannot train, stopping early")
            break

        with threadpool_limits(limits=1, user_api="blas"):
            scores, pipe = _cv_semisup_scores(X, labels, fold_ids, _make_pipe)

        q = _tdc_qvalues(scores, is_decoy)
        # Training-FDR escalation (H-fdr-2): retry at increasing thresholds
        # rather than giving up the moment a strict train_fdr yields nothing.
        _iter_fdr = train_fdr
        new_labels = _encode_labels(is_decoy, q <= _iter_fdr)
        n_new = int((new_labels == 1).sum())
        while n_new == 0 and train_fdr_escalate and _iter_fdr < 0.5:
            _iter_fdr += 0.005
            new_labels = _encode_labels(is_decoy, q <= _iter_fdr)
            n_new = int((new_labels == 1).sum())
        _esc_note = f" (escalated to {_iter_fdr:.3g})" if _iter_fdr != train_fdr else ""

        logger.info(
            f"  {_tag} iter {iteration + 1}: pseudo-positives = {n_new} "
            f"(prev = {prev_pos_size}){_esc_note}"
        )

        if n_new == 0:
            logger.warning(f"  {_tag}: no pseudo-positives at q≤{_iter_fdr:.3g} — stopping early")
            break

        # H-fdr-5: cap self-training growth relative to the initial seed size —
        # stop (keeping THIS iteration's scores, from the last trustworthy label
        # set) rather than accept a pseudo-positive set that ran away regardless
        # of convergence. Guards the same amplify-whatever-the-seed-hands-it
        # behaviour F-020 found exploiting a leak.
        if (
            pseudo_label_growth_cap is not None
            and n_seed > 0
            and n_new > pseudo_label_growth_cap * n_seed
        ):
            logger.warning(
                f"  {_tag}: pseudo-positive growth capped — {n_new} > "
                f"{pseudo_label_growth_cap}x seed ({n_seed}); stopping, keeping previous model"
            )
            break

        change = abs(n_new - prev_pos_size) / max(prev_pos_size, 1)
        if prev_pos_size >= 0 and change < 0.01:
            logger.info(f"  {_tag}: converged")
            break

        prev_pos_size = n_new
        labels = new_labels

    if n_init_positives is not None and 0 <= prev_pos_size < n_init_positives:
        logger.warning(
            f"  {_tag}: final iteration positives ({prev_pos_size}) < "
            f"best-feature init ({n_init_positives}). Using model anyway."
        )

    if pipe is not None:
        _log_imputation_debug(_tag, X, present, is_target, is_decoy, pipe)

    importances = None
    struct_coefs: np.ndarray | None = None
    feature_names_out = present
    if pipe is not None:
        try:
            est = pipe[clf_name]
            if hasattr(est, "coef_"):
                importances = est.coef_[0]
        except Exception:
            pass
        try:
            # Structure coefficients: correlation of each (imputed+scaled) original
            # feature with the discriminant score.  Unlike raw LDA coefficients,
            # these are unaffected by collinearity between features.
            X_imp = pipe["imputer"].transform(X)
            X_sc = pipe["scaler"].transform(X_imp)
            struct_coefs = np.array([
                float(np.corrcoef(X_sc[:, j], scores)[0, 1])
                for j in range(X_sc.shape[1])
            ])
            struct_coefs = np.nan_to_num(struct_coefs, nan=0.0)
        except Exception:
            pass
        # Kernel models (the rbf_svm backend) have no coef_: the decision function
        # lives in kernel space, not per-feature. Report permutation importance
        # instead, which is a contribution rather than a correlation (H-model-2,
        # F-041). Falls back to |structure coefficient| if that fails, so the
        # importance TSV is always populated.
        if importances is None and perm_importance:
            try:
                importances = _permutation_importance(pipe, X)
            except Exception as exc:
                logger.warning(f"  {_tag}: permutation importance failed ({exc})")
        if importances is None and struct_coefs is not None:
            importances = np.abs(struct_coefs)
    # Expose the fitted pipeline + raw feature matrix for downstream SHAP debug
    # explanations (populated only when the caller passes a mutable dict).
    if fitted_out is not None and pipe is not None:
        fitted_out["pipe"] = pipe
        fitted_out["X"] = X
        fitted_out["feature_names"] = present
    return scores, importances, struct_coefs, present, feature_names_out


def _rescore_linear(*args, model_repeats: int = 1, **kwargs):
    """``_rescore_linear_once`` averaged over ``model_repeats`` replicate fits.

    Why this exists (H-fdr-6, F-030). The semi-supervised loop does not converge:
    it is a chaotic map from the seed labels, and every arbitrary internal choice
    resamples its output. Re-running kidney's E016 round 1 under twelve CV
    partitions — no data, hyperparameter or label change — gave q-value floors of
    0.024 to 0.071 and 0 to 63 peptides at 5% FDR, with the published run among
    the worst three. Its reported "0 IDs at 5%" was a draw, not a property of the
    tissue.

    Each replicate is an *independent* trajectory (its own CV partition), so the
    mean over replicates converges where perturbing a single trajectory does not:
    measured on kidney, two disjoint sets of 30 replicates agree at 50 and 55
    peptides at 5% FDR, against the 0-to-63 single-trajectory spread.

    Averaging is over standardised scores, since replicates share a ranking but
    not a scale. Both classes go through the identical code path in every
    replicate, so the TDC null is untouched — checked by permuting the final
    report's labels (0 passing in 60 of 60 trials, all three datasets).

    Reported coefficients and structure coefficients come from the first
    replicate: they describe one fitted model and averaging them across
    replicates would describe none of them. **Permutation importances are the
    exception and are averaged** over the first ``_PERM_IMPORTANCE_REPLICATES``
    replicates, because a single replicate's ranking is not reproducible on every
    dataset — measured on E021, replicate-to-replicate Spearman is 0.95 on
    amyloidosis and 0.97 on her2 but 0.44 on kidney, with a worst pair of 0.16
    (F-041). That is the same instability the scores have, so it gets the same
    treatment. Five is a compromise: it covers most of the noise, and the cost is
    a fifth of a full 20-replicate measurement, which would add 23 minutes to a
    117 minute amyloidosis run to sharpen a ranking that was already stable
    there.

    ``model_repeats=1`` is exactly ``_rescore_linear_once``.
    """
    if model_repeats <= 1:
        return _rescore_linear_once(*args, **kwargs)

    # Intercepted so each replicate's fitted pipeline can be inspected here; the
    # caller's dict still ends up holding the last replicate's, as before.
    kwargs = dict(kwargs)
    caller_fitted_out = kwargs.pop("fitted_out", None)
    fitted: dict = {}

    scores_acc = []
    perm_acc = []
    first: tuple | None = None
    for r in range(model_repeats):
        out = _rescore_linear_once(
            *args, fold_seed=r, fitted_out=fitted,
            perm_importance=(r < _PERM_IMPORTANCE_REPLICATES), **kwargs,
        )
        s = np.asarray(out[0], dtype=np.float64)
        scores_acc.append((s - s.mean()) / (s.std() or 1.0))
        # out[1] is a permutation importance only for an estimator without
        # coef_; a coef_ is reported from replicate 0 unaveraged.
        est = fitted.get("pipe", [None])[-1]
        if (
            r < _PERM_IMPORTANCE_REPLICATES
            and out[1] is not None
            and est is not None
            and not hasattr(est, "coef_")
        ):
            perm_acc.append(np.asarray(out[1], dtype=np.float64))
        if first is None:
            first = out
    if caller_fitted_out is not None:
        caller_fitted_out.update(fitted)
    logger.info(f"  Averaged {model_repeats} replicate fits (differing CV partition)")
    importances = np.mean(perm_acc, axis=0) if len(perm_acc) > 1 else first[1]
    return (np.mean(scores_acc, axis=0), importances) + tuple(first[2:])


# Final pipeline step of each scoring backend. All three expose
# ``decision_function``, so they share ``_rescore_linear``: seed, pseudo-label
# iteration, out-of-fold CV, winner selection, TDC and PEP. ``lda`` and ``svm``
# report ``coef_[0]`` as importance; ``rbf_svm`` has no ``coef_`` (its weights
# live in kernel space), so it reports permutation importance. SVC training is
# ~O(N^2). On standardized features, ``rbf_svm_gamma`` of 0.01-0.03 with
# ``rbf_svm_c`` of 5-10 typically outperforms the ``"scale"`` default.
_ESTIMATOR_MODELS = ("lda", "svm", "rbf_svm")


def _estimator_factory(model: str, svm_c: float = 1.0, rbf_svm_c: float = 1.0, rbf_svm_gamma="scale"):
    """Zero-argument factory for the final pipeline estimator of ``model``.

    ``rbf_svm_gamma`` accepts ``"scale"``/``"auto"``, a float, or a float as a string.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC, LinearSVC

    gamma = rbf_svm_gamma
    if isinstance(gamma, str) and gamma not in ("scale", "auto"):
        gamma = float(gamma)
    return {
        "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]),
        "svm": lambda: LinearSVC(penalty="l2", loss="squared_hinge", C=svm_c, dual="auto", max_iter=2000),
        "rbf_svm": lambda: SVC(kernel="rbf", C=rbf_svm_c, gamma=gamma),
    }[model]


def _tdc_qvalues(scores: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """
    Compute per-candidate target-decoy q-values (Storey/Käll TDC).

    Sort by descending score with a stable sort, compute cumulative
    FDR = (1 + n_decoy) / max(n_target, 1) at each position (the +1
    correction is the standard Storey/Käll adjustment for small-N), then
    take the minimum FDR seen at or below each score (rolling min from
    the tail).
    """
    scores = np.asarray(scores)
    is_decoy = np.asarray(is_decoy).astype(bool)
    # Ties must be broken against the targets, not by row order. A stable sort on
    # score alone inherits the candidates frame's order, which is every target
    # followed by every decoy -- so if scoring degenerates and returns one constant
    # value, every target ranks above every decoy and the run reports essentially
    # all of them at q = 1/n_targets. Measured on kidney: a seed failure returned
    # all-zero scores and 2284 of 2798 target peptides "passed" at q <= 0.001.
    # Ordering decoys first inside a tie group makes that failure conservative
    # (nothing passes) instead of silently perfect. Real fitted scores are
    # continuous -- E024 has zero ties on all three datasets -- so this changes no
    # result that was not already meaningless.
    order = np.lexsort((~is_decoy, -scores))
    if scores.size and np.ptp(scores[np.isfinite(scores)] if np.isfinite(scores).any()
                              else np.zeros(1)) == 0:
        logger.warning(
            "TDC q-values: every score is identical (%g). The model almost "
            "certainly failed to train; reported q-values are meaningless.",
            float(scores.flat[0]),
        )
    n_target_cum = np.cumsum(~is_decoy[order]).astype(float)
    n_decoy_cum = np.cumsum(is_decoy[order]).astype(float)

    fdr = (n_decoy_cum + 1.0) / np.maximum(n_target_cum, 1.0)

    # q-value: minimum FDR at or below this score (monotone from the tail)
    qval_ordered = np.minimum.accumulate(fdr[::-1])[::-1]

    # Map back to original order
    q_values = np.empty_like(qval_ordered)
    q_values[order] = qval_ordered
    return np.clip(q_values, 0.0, 1.0)


def estimate_pep(
    scores: np.ndarray,
    is_decoy: np.ndarray,
) -> np.ndarray:
    """
    Estimate posterior error probability (PEP) via a two-component Gaussian mixture.

    Model
    -----
    f0 : null distribution fitted to decoy scores.
    f1 : signal distribution fitted to target scores above the target median.
         Using only the right tail avoids contamination from incorrect target
         matches that overlap with the null.
    pi0 : n_decoy / n_total.

    PEP(s) = pi0 * f0(s) / (pi0 * f0(s) + (1 - pi0) * f1(s)), clipped to [0, 1].

    Returns NaN for all entries when fewer than 2 decoys or fewer than 2 targets
    are present (mixture is unidentifiable).
    """
    scores = np.asarray(scores, dtype=float)
    is_decoy = np.asarray(is_decoy, dtype=bool)

    n_decoy = int(is_decoy.sum())
    n_target = int((~is_decoy).sum())
    if n_decoy < 2 or n_target < 2:
        return np.full(len(scores), np.nan)

    pi0 = n_decoy / len(scores)

    decoy_scores = scores[is_decoy]
    target_scores = scores[~is_decoy]

    median_t = float(np.median(target_scores))
    high_t = target_scores[target_scores > median_t]
    if len(high_t) < 2:
        high_t = target_scores

    from scipy.stats import norm
    mu0 = float(np.mean(decoy_scores))
    sigma0 = max(float(np.std(decoy_scores)), 1e-6)
    mu1 = float(np.mean(high_t))
    sigma1 = max(float(np.std(high_t)), 1e-6)
    f0 = norm.pdf(scores, mu0, sigma0)
    f1 = norm.pdf(scores, mu1, sigma1)

    numer = pi0 * f0
    denom = numer + (1.0 - pi0) * f1
    with np.errstate(invalid="ignore", divide="ignore"):
        pep = np.where(denom > 0.0, numer / denom, 1.0)

    return np.clip(pep, 0.0, 1.0)


def _pep_qvalues(pep: np.ndarray) -> np.ndarray:
    """
    Convert PEP values to q-values using the cumulative-mean estimator.

    PSMs are sorted by ascending PEP; q(k) = mean(PEP_1 ... PEP_k).  This is
    the BH-style q-value interpretation of PEP.  NaN entries (non-winners) are
    propagated as NaN.
    """
    pep = np.asarray(pep, dtype=float)
    q = np.full(len(pep), np.nan)
    finite = np.isfinite(pep)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = np.argsort(pep[idx])
    sorted_pep = pep[idx][order]
    cumavg = np.cumsum(sorted_pep) / (np.arange(len(sorted_pep)) + 1.0)
    result = np.empty(len(idx))
    result[order] = cumavg
    q[idx] = result
    return q


def _select_calibration_peptides(
    candidates: pd.DataFrame,
    percentile: float = 0.10,
) -> np.ndarray:
    """
    Select the best target candidates to finetune/calibrate IM2Deep on.

    Returns a boolean array aligned with ``candidates`` rows: True for the top
    ``percentile`` fraction of TARGET rows ranked by spectral quality (low
    ``ppm_error_abs`` and high ``theo_isotope_cosine``).  Decoys are never
    eligible — a decoy has no observed RT and assigning a feature's observed CCS
    to a decoy is a false (peptide, label) pair, so decoys cannot supply a valid
    calibration anchor.

    The ranking is computed only over targets and uses features that are blind to
    ``is_decoy``, so the selection introduces no target/decoy asymmetry into the
    downstream ranker.  No target/decoy competition or q-value is used here: this
    is a quality filter, not an FDR estimate.  Unlike the previous
    ``n_candidates == 1`` heuristic, the size of this set does not shrink when
    decoys are paired onto target features (e.g. under ``mz_shuffle``), and it
    selects likely-correct peptides rather than merely mass-unambiguous ones.
    """
    n = len(candidates)
    keep = np.zeros(n, dtype=bool)
    if n == 0 or percentile <= 0:
        return keep

    is_target = ~candidates["is_decoy"].to_numpy(dtype=bool)
    if not is_target.any():
        return keep

    from scipy.stats import zscore

    def _quality_z(values: np.ndarray) -> np.ndarray:
        """Standardised quality contribution; NaN rows and constant columns contribute 0."""
        x = np.asarray(values, dtype=float)
        finite = np.isfinite(x)
        z = np.zeros_like(x)
        if finite.sum() >= 2 and np.std(x[finite]) >= 1e-12:
            z[finite] = zscore(x[finite])
        return z

    tgt = candidates.loc[is_target]
    ppm = tgt["ppm_error_abs"].to_numpy(dtype=float) if "ppm_error_abs" in tgt else np.full(len(tgt), np.nan)
    iso = (
        tgt["theo_isotope_cosine"].to_numpy(dtype=float)
        if "theo_isotope_cosine" in tgt
        else np.full(len(tgt), np.nan)
    )
    # Low ppm error is good; high theoretical isotope cosine is good.
    score = -_quality_z(ppm) + _quality_z(iso)

    thr = float(np.quantile(score, 1.0 - percentile))
    tgt_keep = score >= thr

    target_pos = np.flatnonzero(is_target)
    keep[target_pos[tgt_keep]] = True
    return keep


def _select_feature_winners(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    feature_col: str,
    winner_percentile: float = 0.02,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    For each MALDI feature select the candidate with the highest score, then
    drop features whose winner score falls below the ``winner_percentile``
    quantile of all winner scores.  Filtered features receive
    ``is_tdc_winner=False`` and NaN winner scores in the final result.

    Returns
    -------
    winner_pos : np.ndarray[int]
        Integer positions (iloc-style) in ``features_df`` of the retained winners.
    winners_df : pd.DataFrame
        Subset of ``features_df`` with one row per retained feature, reset index.
    """
    score_series = pd.Series(scores, index=features_df.index)
    winner_idx = score_series.groupby(features_df[feature_col].values).idxmax().values
    winner_pos = features_df.index.get_indexer(winner_idx)
    winners_df = features_df.loc[winner_idx].copy().reset_index(drop=True)

    winner_scores = scores[winner_pos]
    q1 = np.quantile(winner_scores, winner_percentile)
    keep = winner_scores >= q1
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info(
            f"  Winner filter: dropped {n_dropped} features with score < Quantile({winner_percentile}) ({q1:.4f})"
        )
    winner_pos = winner_pos[keep]
    winners_df = winners_df[keep].reset_index(drop=True)
    return winner_pos, winners_df


def _peptide_level_qvalues(
    scores: np.ndarray,
    is_decoy: np.ndarray,
    peptides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TDC q-values at peptide level: best match per peptide, then TDC over those.

    In feature-list extraction a peptide is matched to every detected feature
    within tolerance, so it contributes one row per (peptide, feature) pair --
    measured at 4.8 rows per peptide on amyloidosis (max 15) and 2.1 on kidney.
    Computing TDC over those rows counts the same peptide as many separate
    discoveries on both the target and the decoy side, which inflates the
    reported count and breaks the exchangeability TDC assumes across discoveries
    (Savitski et al. 2015: count the pair once, not both members independently).

    This is the peptide-level analogue of :func:`_select_feature_winners`, and it
    runs *after* it so feature-level target-decoy competition still happens first:
    a candidate must win its feature, and only then does its peptide compete for
    a place in the reported set.

    Aggregating a peptide's features *before* scoring would instead destroy that
    competition, since each peptide would occupy its own aggregated pseudo-feature
    with nothing to compete against.

    The representative is the best-scoring match, the same rule Percolator and
    mokapot use to roll PSMs up to peptides. That is a maximum over k draws, so
    it is only unbiased while targets and decoys have the same multiplicity
    distribution -- checked and warned about below rather than assumed.

    :returns: ``(peptide_q, is_representative)`` aligned with the input rows.
        ``peptide_q`` is broadcast to every row of a peptide;
        ``is_representative`` marks the single row that carries it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_decoy = np.asarray(is_decoy, dtype=bool)
    frame = pd.DataFrame({"score": scores, "decoy": is_decoy, "peptide": np.asarray(peptides)})

    # Grouping on (peptide, decoy) rather than peptide alone only keeps a target
    # and a decoy that happen to share a sequence from collapsing into each other.
    # Both classes are reduced by the identical rule, so the rollup introduces no
    # asymmetry -- it does not branch on the label, it only avoids merging across it.
    positions = np.arange(len(frame))
    rep = frame.assign(_pos=positions).groupby(["peptide", "decoy"])["score"].idxmax()
    rep_pos = np.sort(rep.to_numpy())

    is_representative = np.zeros(len(frame), dtype=bool)
    is_representative[rep_pos] = True

    # Multiplicity must be symmetric or the best-of-k maximum favours one class.
    mult = frame.groupby(["peptide", "decoy"]).size()
    mult_t = mult[mult.index.get_level_values("decoy") == False]  # noqa: E712
    mult_d = mult[mult.index.get_level_values("decoy") == True]  # noqa: E712
    if len(mult_t) and len(mult_d):
        ratio = mult_d.mean() / mult_t.mean() if mult_t.mean() else 1.0
        logger.info(
            "  Peptide-level rollup: %d rows -> %d peptides "
            "(%.2f matches/target-peptide, %.2f per decoy)",
            len(frame), len(rep_pos), mult_t.mean(), mult_d.mean(),
        )
        if not 0.8 <= ratio <= 1.25:
            logger.warning(
                "  Target and decoy match multiplicity differ by %.2fx "
                "(%.2f vs %.2f matches per peptide). The peptide-level rollup takes a "
                "maximum over matches, so asymmetric multiplicity biases the null. "
                "Check the peak list and the matching tolerance before trusting "
                "peptide-level q-values.",
                ratio, mult_t.mean(), mult_d.mean(),
            )

    q_rep = _tdc_qvalues(scores[rep_pos], is_decoy[rep_pos])

    # broadcast each peptide's q back onto all of its rows
    key = pd.MultiIndex.from_arrays([frame["peptide"], frame["decoy"]])
    peptide_q = pd.Series(q_rep, index=key[rep_pos]).reindex(key).to_numpy()
    return peptide_q, is_representative


def _report_entrapment(result_df: "pd.DataFrame", features_df: "pd.DataFrame", output_dir: str) -> None:
    """Count entrapment pseudo-target survivals and write entrapment_result.tsv."""
    if "source" not in features_df.columns:
        return
    result_df = result_df.copy()
    result_df["source"] = features_df["source"].values
    ent = result_df["source"] == "entrapment_shuffled"
    if not ent.any():
        return
    winner = result_df["is_tdc_winner"].fillna(False).astype(bool)
    q = result_df["q_value"].fillna(np.inf)
    n_sub = int(ent.sum())
    is_decoy = result_df["is_decoy"].astype(bool)
    # Denominator = real targets only (exclude entrapment pseudo-targets and all decoys).
    real_target = (~is_decoy) & (~ent)
    lines = ["\nEntrapment validation results:"]
    lines.append(f"  Entrapment peptides submitted: {n_sub}")
    for fdr in [0.01, 0.05, 0.10]:
        mask = winner & (q <= fdr)
        n_ent = int((ent & mask).sum())
        n_all = int((real_target & mask).sum())
        frac = n_ent / n_all if n_all else 0.0
        lines.append(
            f"  Entrapment IDs at {fdr*100:.0f}% FDR: {n_ent} "
            f"({frac*100:.1f}% of {n_all} total IDs; expected ≤{fdr*100:.0f}%)"
        )
    print("\n".join(lines))
    out = os.path.join(output_dir, "entrapment_result.tsv")
    result_df[ent].to_csv(out, sep="\t", index=False)
    logger.info("entrapment: results written to %s", out)


def ccs_threshold_pct(single_ccs, multiplier, fixed_pct):
    """The CCS filter's threshold, and the sentence explaining where it came from.

    Returns ``(threshold_pct | None, p95 | None, message)``. ``None`` for the threshold
    means no filter is applied.

    ``fixed_pct`` wins over ``multiplier`` and needs no calibration set, which is the
    point of it: the p95 is measured on whichever calibration peptides the run happens
    to have, and a denser peak list adds calibration peptides sitting on noise peaks.
    Measured on amyloidosis, the p95 went 3.18% at ``min_regions=2`` to 4.06% at 1, so
    the same multiplier loosened the window exactly where it needed tightening and two
    runs being compared did not share a threshold (PROGRESS.md F-050).
    """
    n = len(single_ccs)
    p95 = float(np.percentile(single_ccs, 95)) if n >= 10 else None
    if fixed_pct is not None:
        seen = f"{p95:.2f}%." if p95 is not None else "not measurable."
        return float(fixed_pct), p95, (
            f"CCS filter: fixed threshold {float(fixed_pct):.2f}% (--ccs-window-pct; the "
            f"multiplier is ignored). p95 |delta_CCS%| on {n} calibration peptides is {seen}"
        )
    if p95 is not None:
        return float(multiplier * p95), p95, (
            f"CCS filter: p95 |delta_CCS%| on {n} calibration peptides = {p95:.2f}%. "
            f"Threshold = {multiplier}× = {multiplier * p95:.2f}%."
        )
    return None, None, (
        f"CCS filter: only {n} single-candidate matches with observed CCS — too few for a "
        "reliable data-driven threshold. Skipping CCS filter. Set --ccs-window-pct for a "
        "fixed window."
    )


def rescore(
    fasta_path: str | None,
    maldi_mzs: np.ndarray,
    spatial_features: pd.DataFrame | None = None,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    extra_ion_images: dict | None = None,
    maldi_envelopes: dict | None = None,
    maldi_query_raw: bool = False,
    maldi_d_path: str | None = None,
    raw_query_cache: dict | None = None,
    raw_query_cache_dir: str | None = None,
    extraction_ppm: float = 25.0,
    mob_quality_mz_window_ppm: float = 25.0,
    mob_quality_k0_tol: float = 0.02,
    ppm_tolerance: float = 20.0,
    init_fdr: float = 0.2,
    train_fdr: float = 0.05,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    model: str = "lda",
    svm_c: float = 1.0,
    rbf_svm_c: float = 1.0,
    rbf_svm_gamma: str = "scale",
    init_ppm_threshold: float = 5.0,
    only_main_features: bool = False,
    lcms_ids=None,
    lcms_proteins_path: str | None = None,
    lcms_peptides_path: str | None = None,
    lcms_psms_path: str | None = None,
    lcms_id_format: str = "percolator",
    psm_utils_reader: str | None = None,
    protein_fdr: float = 0.01,
    peptide_fdr: float = 0.01,
    extra_fasta_path: str | None = None,
    use_protein_level_features: bool = False,
    use_spatial_ranker_features: bool = False,
    verbose: bool = False,
    output_dir: str = "ms1rescore_output",
    debug_dir: str | None = None,
    n_debug: int = 50,
    debug_seed: int = 42,
    observed_ccs_per_feature: dict | None = None,
    im2deep_calibration: str = "linear",
    digest: bool = False,
    gt_peptides: list[str] | None = None,
    decoy_method: str = "substitution",
    features_preset: str = "all",
    features_exclude: list[str] | None = None,
    r1_seed_percentile: float = 0.10,
    max_iter: int = 5,
    min_seed_positives: int = 50,
    seed_features: list[str] | None = None,
    im2deep_kwargs: dict | None = None,
    calibration_percentile: float = 0.10,
    matching_ppm: float = 20.0,
    winner_percentile: float = 0.02,
    match_ccs: bool = False,
    ccs_window_multiplier: float = 2.0,
    ccs_window_pct: float | None = None,
    tdf_path: str | None = None,
    mob_coloc: bool = False,
    mob_protein_coloc: bool = False,
    mob_window_multiplier: float = 2.0,
    coloc_tic_quantile: float = 0.0,
    coloc_measured_pixel_mask: "np.ndarray | None" = None,
    coloc_tic_normalize: bool = False,
    coloc_common_mode: bool = False,
    cosine_coloc: bool = False,
    train_fdr_escalate: bool = False,
    pseudo_label_growth_cap: float | None = None,
    model_repeats: int = 1,
    drop_zero_signal: bool = False,
    entrapment: bool = False,
    substitution_n_residues: int = 1,
    substitution_seed: int = 42,
    substitution_collision_filter: bool = True,
    substitution_collision_ppm: float | None = None,
    substitution_mass_shift_min_da: float | None = None,
    substitution_mass_shift_max_da: float | None = None,
    substitution_residue_weighting: str | None = None,
    protein_size_residualize: bool = True,
    tic_image: np.ndarray | None = None,
    tic_n_features: int | None = None,
    substitution_preserve_sulfur: bool | None = None,
):
    """
    End-to-end symmetric MALDI-MSI rescoring pipeline.

    Parameters
    ----------
    fasta_path
        Path to protein FASTA file (forward sequences only). Used only with
        ``digest=True``.
    maldi_mzs
        Array of MALDI feature m/z values.
    spatial_features
        Pre-computed spatial features DataFrame aligned with ``maldi_mzs``
        (optional; produced by ``compute_spatial_features``).
    ion_images
        MALDI ion images, shape ``(n_features, H, W)`` float32 (optional).
        Required for colocalization and spatial autocorrelation features.
    ion_image_mzs
        m/z values aligned with ``ion_images`` rows (optional).
    extra_ion_images
        Dict of ion images extracted at shifted m/z positions for direct
        colocalization without requiring those peaks in the feature list
        (optional). Keys: ``"m1"``, ``"m2"`` (M+1/M+2 isotopologues) and
        ``"na"``, ``"k"``, ``"chca"`` (adducts). Each value is
        ``(n_features, H, W)`` float32.
    maldi_envelopes
        MALDI isotope envelopes: ``feature_mz → normalized envelope array``
        (optional).
    ppm_tolerance
        m/z half-window (ppm) for the mobility-colocalization peak extraction.
    train_fdr
        FDR threshold for the per-iteration pseudo-label update.
    max_iter
        Maximum pseudo-label iterations.
    missed_cleavages, min_length, max_length
        Tryptic in-silico digest settings.
    model
        Rescoring backend, a key of ``_ESTIMATOR_MODELS``: ``"lda"`` (default,
        LinearDiscriminantAnalysis), ``"svm"`` (LinearSVC, C=``svm_c``), or
        ``"rbf_svm"`` (RBF-kernel SVC; tuned by ``rbf_svm_c`` / ``rbf_svm_gamma``).
        All train on the MALDI-intrinsic ranker features only.
    init_ppm_threshold
        ppm_error_abs threshold for the ppm-fallback seed, used only when the
        best-feature initialization yields no passing targets. Targets below this
        threshold (or with ``n_candidates == 1``) become the initial pseudo-positives.
    only_main_features
        If True, restrict the feature set to ``MAIN_FEATURES`` (one
        representative per collinear group) instead of the full
        ``MALDI_INTRINSIC_FEATURES``.
    lcms_ids
        Already parsed LC-MS/MS identifications (``parse_lcms_ids`` output). When
        given, the ``lcms_*_path`` arguments are not read again.
    lcms_peptides_path
        Path to LC-MS/MS peptide-level identification results. When provided (or
        ``lcms_ids`` is), activates Strategy C: candidates are the identified
        peptides, plus the digest of the identified proteins with ``digest=True``.
    lcms_proteins_path
        Path to LC-MS/MS protein-level results (optional; proteins are derived
        from the peptide table when omitted).
    lcms_psms_path
        Path to LC-MS/MS PSM-level file for RT and intensity aggregation (optional).
    lcms_id_format
        Format of the LC-MS/MS ID files: ``"percolator"`` (default),
        ``"mzidentml"``, ``"psm_utils"``, ``"msf"`` or ``"ms2rescore"``.
    psm_utils_reader
        Reader hint for ``lcms_id_format="psm_utils"``. A psm_utils filetype
        key (e.g. ``"maxquant"``) or reader class name. When ``None``,
        auto-detection from the filename is attempted.
    protein_fdr, peptide_fdr
        Strategy C protein- and peptide-level FDR thresholds.
    extra_fasta_path
        Additional FASTA file (e.g. contaminants database). All proteins in
        this file are always included. Peptides already present in the primary
        digest are not duplicated.
    use_protein_level_features
        If True, include ``PROTEIN_LEVEL_FEATURES`` in the ranker feature set.
    use_spatial_ranker_features
        If True, include ``SPATIAL_RANKER_FEATURES`` (feature-level spatial
        quality and protein colocalization) in the ranker feature set.
        Deduplicated against ``PROTEIN_LEVEL_FEATURES``.
    verbose
        If True, write per-step debug files to ``output_dir``.
    output_dir
        Directory for all output files.
    debug_dir
        Directory for debug figures (optional; set by the CLI with ``--verbose``).
    n_debug, debug_seed
        Number of MALDI features in the debug visualizations, and the sampling seed.
    observed_ccs_per_feature
        Dict mapping ``feature_idx → observed CCS`` (optional). Enables the
        IM2Deep CCS features.
    im2deep_calibration
        IM2Deep CCS calibration mode: ``"linear"``, ``"spline"``, or ``"finetune"``.
    im2deep_kwargs
        Keyword arguments forwarded to ``im2deep.core.finetune`` when
        ``im2deep_calibration="finetune"``.
    digest
        If True, also digest ``fasta_path`` (identified proteins under Strategy C,
        the whole FASTA otherwise).
    gt_peptides
        Ground-truth peptide sequences for diagnostic reporting (not used in scoring).
    decoy_method
        ``"substitution"`` (default): interior residues of each target are
        substituted. ``"mz_shuffle"``: derangement of the peptide→feature
        assignment, so each real target peptide is relocated onto another
        peptide's real feature (co-located 1 target + 1 decoy per feature).
    maldi_query_raw
        When ``True``, ion images are queried directly from the raw ``.d`` data
        at candidate-derived m/z values instead of from a pre-picked feature
        list. Superseded; kept to reproduce older results. Requires ``maldi_d_path``.
    maldi_d_path
        Path to the raw Bruker ``.d`` directory. Required when ``maldi_query_raw=True``.
    raw_query_cache, raw_query_cache_dir
        In-process and on-disk caches for raw-query mode (see CLAUDE.md).
    extraction_ppm
        Ion image extraction half-window (ppm) used by raw-query mode.
    calibration_percentile
        Fraction (0–1) of TARGET candidates used to finetune IM2Deep, selected by
        low ``ppm_error_abs`` and high ``theo_isotope_cosine``, blind to ``is_decoy``.
    features_preset
        ``"all"`` (``MALDI_INTRINSIC_FEATURES``) or ``"main"`` (``MAIN_FEATURES``).
    features_exclude
        Feature names to remove from the ranker feature set.
    train_fdr_escalate
        If ``init_fdr``/``train_fdr`` would otherwise yield zero pseudo-positives,
        retry at increasing thresholds (steps of 0.005, capped at 0.5).
    pseudo_label_growth_cap
        H-fdr-5, opt-in. Stops the self-training loop once the pseudo-positive
        count exceeds this multiple of the initial seed size. ``None`` disables it.
    model_repeats
        H-fdr-6, opt-in. Number of independent replicate fits (each with its own
        CV partition) whose standardised scores are averaged (F-030).
    r1_seed_percentile
        Top-ppm fallback fraction for the seed when the ppm threshold yields no
        positives.

    Returns
    -------
    tuple of (result_df, features_df)
        ``result_df`` has one row per candidate with ``peptide``, ``protein``,
        ``feature_mz``, ``feature_idx``, ``is_decoy``, the score of every
        candidate (``<model>_score_r1``), and for winners only ``score``,
        ``q_value``, ``pep``, ``pep_q_value``, ``peptide_q_value``
        (NaN for non-winners), plus ``is_tdc_winner`` and ``is_peptide_winner``.
        ``features_df`` is the feature matrix the ranker was fitted on, same row
        order.
    """
    # --- Step 1: Candidate generation ---
    # Default (digest=False): use only LC-MS/MS identified peptides as candidates.
    # With digest=True: also digest the provided FASTA for additional candidates.
    # The CLI parses the IDs once and passes them in as ``lcms_ids``; other callers
    # may pass the paths instead.
    if lcms_ids is None and lcms_peptides_path is not None:
        from msi_picasso.lcms_ids import parse_lcms_ids

        logger.info("Step 1: Parsing LC-MS/MS identifications...")
        lcms_ids = parse_lcms_ids(
            proteins_path=lcms_proteins_path,
            peptides_path=lcms_peptides_path,
            psms_path=lcms_psms_path,
            protein_fdr=protein_fdr,
            peptide_fdr=peptide_fdr,
            format=lcms_id_format,
            psm_utils_reader=psm_utils_reader,
        )
    if lcms_ids is not None:
        if verbose:
            logger.debug("Writing parsed LC-MS/MS IDs to debug_lcms_ids.tsv")
            lcms_ids.peptides.to_csv(
                f"{output_dir}/5_debug_lcms_ids.tsv", sep="\t", index=False
            )

        # Pass fasta_path only when --digest is active; None = LC-MS/MS peptides only.
        _digest_fasta_arg = fasta_path if digest else None
        if digest:
            logger.info(
                f"  --digest active: digesting identified proteins from {fasta_path}"
            )
        else:
            logger.info("  Using LC-MS/MS identified peptides only (pass --digest to also digest FASTA).")

        peptide_db = digest_identified_proteins(
            _digest_fasta_arg,
            lcms_ids,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )
        if verbose:
            logger.debug(
                f"Writing peptide database to {output_dir}/6_debug_peptide_db.tsv"
            )
            pd.DataFrame(peptide_db).to_csv(
                f"{output_dir}/6_debug_peptide_db.tsv", sep="\t", index=False
            )
        if len(peptide_db) == 0 and digest and fasta_path:
            logger.warning(
                "  No candidates from identified proteins — falling back to full FASTA digest."
            )
            peptide_db = digest_fasta(
                fasta_path,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
                generate_decoys=True,
            )
    elif digest and fasta_path:
        logger.info("Step 1: --digest active, no LC-MS/MS IDs — digesting full FASTA...")
        peptide_db = digest_fasta(
            fasta_path,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=True,
        )
        if verbose:
            pd.DataFrame(peptide_db).to_csv(
                f"{output_dir}/7_debug_peptide_db_full.tsv", sep="\t", index=False
            )
    else:
        raise ValueError(
            "No candidate source available. Provide --lcms-peptides (or --msf) "
            "to use LC-MS/MS identified peptides as candidates, or add --digest "
            "with --fasta to perform an in-silico digest."
        )

    # --- Step 1b: Merge extra FASTA (contaminants / spike-ins) ---
    if extra_fasta_path is not None:
        logger.info(f"Step 1b: Merging extra FASTA: {extra_fasta_path}")
        extra_db = digest_fasta(
            extra_fasta_path,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=True,
        )
        existing_seqs = set(peptide_db["peptide"].values)
        extra_db = extra_db[~extra_db["peptide"].isin(existing_seqs)].copy()
        n_target_extra = int((~extra_db["is_decoy"]).sum())
        n_decoy_extra = int(extra_db["is_decoy"].sum())
        peptide_db = pd.concat([peptide_db, extra_db], ignore_index=True)
        logger.info(
            f"  Added {n_target_extra} target + {n_decoy_extra} decoy peptide entries "
            f"from {extra_fasta_path!r}"
        )

    # --- True full-digest peptide count per protein (for protein_coverage) ---
    # peptide_db is the complete in-silico tryptic digest (length-filtered),
    # BEFORE m/z matching, so its per-protein unique-peptide count is the true
    # number of theoretically observable peptides. This is the correct
    # denominator for protein_coverage. Computed over target peptides only and
    # keyed by the base accession; decoys (DECOY_/ENTRAPMENT_ namespaces) inherit
    # the count of their source protein, keeping coverage symmetric. (The
    # per-candidate count produced downstream is the *observed* peptide pool, not
    # the full digest, and would make coverage degenerate — see Step 6.)
    _pdb_decoy = (
        peptide_db["is_decoy"].astype(bool)
        if "is_decoy" in peptide_db.columns
        else pd.Series(False, index=peptide_db.index)
    )
    protein_full_tryptic_count = (
        peptide_db.loc[~_pdb_decoy].groupby("protein")["peptide"].nunique().to_dict()
    )

    # --- Pre-generate entrapment DB (before maldi_mzs is fixed for raw-query mode) ---
    # In raw-query mode maldi_mzs is derived from peptide_db mzs; entrapment
    # peptides are not in that grid.  Building _entrapment_db here lets us expand
    # the grid before Step 1c so entrapment mzs are included in the extraction.
    _entrapment_db = None
    if entrapment:
        if lcms_ids is None:
            raise ValueError(
                "--entrapment requires LC-MS/MS IDs (--lcms-peptides or --msf); "
                "entrapment candidates are generated from confirmed sequences."
            )
        from msi_picasso.candidates import generate_entrapment_from_lcms_ids
        _entrapment_db = generate_entrapment_from_lcms_ids(
            lcms_ids,
            matching_ppm=matching_ppm,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )

    # --- Raw-query mode: invert ordering (candidates drive MALDI extraction) ---
    # The candidate digest m/z become the matching grid; the actual ion images
    # are queried from the raw .d AFTER candidate generation (see below).
    if maldi_query_raw:
        if maldi_d_path is None:
            raise ValueError(
                "maldi_query_raw=True requires maldi_d_path (the raw Bruker .d directory)"
            )
        _target_mzs = np.sort(np.unique(
            peptide_db["mh_mz"].dropna().to_numpy(dtype=np.float64)
        ))
        if _entrapment_db is not None and len(_entrapment_db) > 0:
            _ent_extra = np.sort(np.unique(
                _entrapment_db["mh_mz"].dropna().to_numpy(dtype=np.float64)
            ))
            maldi_mzs = np.sort(np.unique(np.concatenate([_target_mzs, _ent_extra])))
            logger.info(
                "Raw-query mode + entrapment: expanded matching grid from %d to %d "
                "unique m/z (added %d entrapment mzs).",
                len(_target_mzs), len(maldi_mzs), len(_ent_extra),
            )
        else:
            maldi_mzs = _target_mzs
        logger.info(
            "Raw-query mode: using %d unique candidate m/z as the matching grid; "
            "ion images will be extracted from %s after candidate generation.",
            len(maldi_mzs), maldi_d_path,
        )

    _maldi_intensities_arr = None
    maldi_intensities_p90 = None
    maldi_intensities_sum = None
    if spatial_features is not None:
        if "intensity_p90" in spatial_features.columns:
            maldi_intensities_p90 = spatial_features["intensity_p90"].to_numpy(dtype=np.float32)
        if "intensity_sum" in spatial_features.columns:
            maldi_intensities_sum = spatial_features["intensity_sum"].to_numpy(dtype=np.float32)
        if "mean_intensity" in spatial_features.columns:
            _maldi_intensities_arr = spatial_features["mean_intensity"].to_numpy(dtype=np.float32)
    elif ion_images is not None:
        _maldi_intensities_arr = np.array(
            [img[img > 0].mean() if (img > 0).any() else 0.0 for img in ion_images]
        )

    # --- Step 1c: Decoy generation ---
    target_db = target_rows(peptide_db)
    if decoy_method == "mz_shuffle":
        # Derangement of the peptide->feature assignment: each target peptide is
        # relocated onto another peptide's real feature (co-located 1 target + 1
        # decoy per feature). Feature-quality features are then identical between a
        # feature's target and decoy, so the ranker must discriminate on the
        # peptide-specific predicted-vs-observed match (CCS, isotope).
        logger.info("Step 1c: Generating m/z-assignment-shuffle (derangement) decoys...")
        # In raw-query mode with entrapment the grid is expanded; restrict the
        # shuffle destinations to target mzs so decoys don't land on entrapment
        # features (_target_mzs is set in the raw-query block above, else maldi_mzs).
        _shuffle_grid = (
            _target_mzs
            if maldi_query_raw and _entrapment_db is not None and len(_entrapment_db) > 0
            else maldi_mzs
        )
        candidates = generate_mz_shuffle_candidates(
            target_db,
            _shuffle_grid,
            matching_ppm=matching_ppm,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
    elif decoy_method == "substitution":
        logger.info(
            "Step 1c: Generating substitution decoys "
            "(n_residues=%d, seed=%d, collision_filter=%s)...",
            substitution_n_residues, substitution_seed, substitution_collision_filter,
        )
        candidates = generate_substitution_candidates(
            target_db,
            maldi_mzs,
            matching_ppm=matching_ppm,
            n_residues=substitution_n_residues,
            random_seed=substitution_seed,
            mass_shift_min_da=substitution_mass_shift_min_da,
            mass_shift_max_da=substitution_mass_shift_max_da,
            residue_weighting=substitution_residue_weighting or "uniform",
            preserve_sulfur=bool(substitution_preserve_sulfur),
            collision_filter=substitution_collision_filter,
            collision_ppm=substitution_collision_ppm,
            snap_to_features=not maldi_query_raw,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
    else:
        raise ValueError(
            f"Unknown decoy_method {decoy_method!r}. Choose 'substitution' or 'mz_shuffle'."
        )
    # --- Entrapment: inject shuffled pseudo-target candidates ---
    if _entrapment_db is not None and len(_entrapment_db) > 0:
        _ent_cands = match_to_maldi_features(
            maldi_mzs, _entrapment_db, matching_ppm,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
        if len(_ent_cands) > 0:
            candidates = pd.concat([candidates, _ent_cands], ignore_index=True)
            candidates["is_decoy"] = candidates["is_decoy"].astype(bool)
            _fc = "feature_idx" if "feature_idx" in candidates.columns else "feature_mz"
            candidates["n_candidates"] = (
                candidates.groupby(_fc)[_fc].transform("count")
            )
            logger.info(
                "entrapment: added %d shuffled pseudo-target candidates "
                "(%d ENTRAPMENT proteins, %d features).",
                len(_ent_cands), _ent_cands["protein"].nunique(), _ent_cands[_fc].nunique(),
            )
        else:
            logger.warning(
                "entrapment: no shuffled candidates matched any MALDI feature "
                "(matching_ppm=%.1f). Validation will report 0 entrapment IDs.",
                matching_ppm,
            )

    if verbose:
        logger.debug(f"Writing matched candidates to {output_dir}/9_debug_candidates.tsv")
        candidates.to_csv(f"{output_dir}/9_debug_candidates.tsv", sep="\t", index=False)

    if len(candidates) == 0:
        raise ValueError("No candidates matched any MALDI features")

    logger.info(
        f"  {len(candidates)} candidates ({(~candidates['is_decoy']).sum()} target, "
        f"{candidates['is_decoy'].sum()} decoy) across "
        f"{candidates['feature_mz'].nunique()} features"
    )

    # --- Raw-query mode: extract ion images at the candidate-derived m/z ---
    # candidates["feature_mz"] now holds every queried m/z (for a substitution decoy
    # this is its own [M+H]+).  Extract directly from the raw .d, then attach
    # the freshly computed per-feature intensities back onto the candidate rows.
    if maldi_query_raw:
        from msi_picasso.maldi_query import (
            extract_observed_feature_stats_raw,
            query_raw_maldi,
        )

        query_mzs = np.sort(
            candidates["feature_mz"].dropna().to_numpy(dtype=np.float64)
        )
        query_mzs = np.unique(query_mzs)

        # Bidirectional extraction cache. ``raw_query_cache`` lets a caller (e.g. a
        # grid search) extract the candidate-grid ion images / observed centroids /
        # CCS once and reuse them across many rescore() runs that vary only the
        # scoring parameters: the candidate m/z set is fixed by the digest + decoy
        # method (constant across such runs), so the cached full-grid arrays are a
        # superset of any run's query_mzs and are reused as-is — extra ion images
        # are ignored by the feature_mz → image lookups. Semantics:
        #   None                       → always extract (default).
        #   {} (or no "ion_images")    → extract, then populate the dict for reuse.
        #   {"ion_images": ...}         → reuse without touching the .d.
        if raw_query_cache is not None and raw_query_cache.get("ion_images") is not None:
            maldi_mzs = raw_query_cache["maldi_mzs"]
            ion_images = raw_query_cache["ion_images"]
            extra_ion_images = raw_query_cache["extra_ion_images"]
            spatial_features = raw_query_cache["spatial_features"]
            maldi_envelopes = raw_query_cache["maldi_envelopes"]
            _ccs_arr = raw_query_cache["ccs_arr"]
            _centroid_arr = raw_query_cache["centroid_arr"]
            _peak_quality = raw_query_cache.get("peak_quality")
            ion_image_mzs = maldi_mzs
            logger.info(
                "Raw-query mode: reusing cached extraction (%d grid ion images) "
                "for %d candidate m/z.", len(maldi_mzs), len(query_mzs),
            )
        else:
            (
                maldi_mzs,
                ion_images,
                extra_ion_images,
                spatial_features,
                maldi_envelopes,
            ) = query_raw_maldi(maldi_d_path, query_mzs, extraction_ppm=extraction_ppm)
            ion_image_mzs = maldi_mzs
            # Observed peak centroids + CCS from the raw .d (alphatims). imzy exposes
            # neither, so the .d is opened a second time here.
            _ccs_arr, _centroid_arr, _peak_quality = extract_observed_feature_stats_raw(
                maldi_d_path, maldi_mzs, extraction_ppm=extraction_ppm,
                mob_quality_window_ppm=mob_quality_mz_window_ppm,
                mob_quality_k0_tol=mob_quality_k0_tol,
                cache_dir=raw_query_cache_dir,
            )
            logger.info(
                "Raw-query mode: extracted %d ion images; %d features with M0 envelope signal.",
                len(maldi_mzs), len(maldi_envelopes),
            )
            if raw_query_cache is not None:
                raw_query_cache.update(
                    maldi_mzs=maldi_mzs, ion_images=ion_images,
                    extra_ion_images=extra_ion_images, spatial_features=spatial_features,
                    maldi_envelopes=maldi_envelopes, ccs_arr=_ccs_arr, centroid_arr=_centroid_arr,
                    peak_quality=_peak_quality,
                )

        # Attach per-feature intensities (mapped by m/z) onto the candidate rows.
        _p90 = dict(zip(spatial_features["feature_mz"], spatial_features["intensity_p90"]))
        _sum = dict(zip(spatial_features["feature_mz"], spatial_features["intensity_sum"]))
        _mean = dict(zip(spatial_features["feature_mz"], spatial_features["mean_intensity"]))
        candidates["feature_intensity_p90"] = candidates["feature_mz"].map(_p90)
        candidates["feature_intensity_sum"] = candidates["feature_mz"].map(_sum)
        candidates["feature_intensity"] = candidates["feature_mz"].map(_mean)

        # Attach intrinsic 2D peak-quality columns (feature-level, mapped by m/z) when
        # ion mobility was extracted.  Aligned with maldi_mzs; bridged via feature_mz
        # exactly like the intensity maps above.  Absent for TSF/no-mobility data.
        if _peak_quality is not None:
            for _col, _vals in _peak_quality.items():
                candidates[_col] = values_at_mz(_vals, maldi_mzs, candidates["feature_mz"])

        # Recompute ppm_error symmetrically from the observed peak centroid in each
        # candidate's own window. In raw-query, candidates are matched against the
        # theoretical digest grid, so the default (feature_mz - mh_mz) ppm is 0 for
        # every self-match and decoys inherit 0. Replacing it with
        # (observed_centroid - feature_mz)/feature_mz * 1e6 gives a real, symmetric
        # mass-accuracy feature: targets and decoys are measured identically against
        # their own anchor, with no inheritance and no label leak.  Candidates whose
        # window has no observed peak (e.g. decoys whose m/z lands in empty space)
        # get the worst-case ppm (the extraction window edge) rather than NaN, so
        # they are penalised on ppm instead of median-imputed to an average value.
        if np.isfinite(_centroid_arr).any():
            _fmz = candidates["feature_mz"].to_numpy()
            _ppm = _recompute_ppm_from_centroids(
                _fmz, maldi_mzs, _centroid_arr, worst_case_ppm=extraction_ppm
            )
            candidates["ppm_error"] = _ppm
            candidates["ppm_error_abs"] = np.abs(_ppm)
            # Count rows whose own window had a real observed peak (vs worst-case fill).
            _n_signal = int(np.isfinite(values_at_mz(_centroid_arr, maldi_mzs, _fmz)).sum())
            logger.info(
                "Raw-query mode: recomputed ppm_error from observed peak centroids "
                "for %d/%d candidate rows; %d empty-window rows set to worst-case "
                "%.1f ppm.",
                _n_signal, len(_ppm), len(_ppm) - _n_signal, extraction_ppm,
            )
        else:
            logger.warning(
                "Raw-query mode: no observed peak centroids available (alphatims "
                "missing or no in-window signal); ppm_error left as matched-grid value."
            )

        # observed_ccs_per_feature unlocks the IM2Deep CCS features, the match_ccs
        # filter, and (with --mob-coloc) mobility-filtered colocalization. Keyed by
        # the candidates' own feature_idx (which indexes the digest grid in raw-query
        # mode, not maldi_mzs), bridged via feature_mz — matching how
        # compute_im2deep_features consumes it (df["feature_idx"].map(...)).
        observed_ccs_per_feature = _observed_ccs_by_feature_idx(
            candidates, maldi_mzs, _ccs_arr
        )
        logger.info(
            "Raw-query mode: observed CCS available for %d features.",
            0 if observed_ccs_per_feature is None else len(observed_ccs_per_feature),
        )

    # --- Select calibration peptides (IM2Deep finetuning anchors) ---
    # theo_isotope_cosine is needed for the quality ranking and is computed in
    # full at Step 6; compute it here (cheap, lru_cache'd, idempotent — Step 6
    # overwrites with identical values). is_calibration_peptide is carried on the
    # candidates DataFrame and reused by the IM2Deep CCS, mobility, and CCS
    # filter steps instead of the decoy-sensitive n_candidates == 1 heuristic.
    from msi_picasso.maldi_features import compute_theoretical_isotope_features
    candidates = compute_theoretical_isotope_features(candidates, maldi_envelopes=maldi_envelopes)
    candidates["is_calibration_peptide"] = _select_calibration_peptides(
        candidates, calibration_percentile
    )
    logger.info(
        f"  Calibration set: {int(candidates['is_calibration_peptide'].sum())} target "
        f"candidates (top {calibration_percentile:.0%} by low ppm + high isotope cosine)"
    )

    # --- Override protein_tryptic_count with the true full-digest count ---
    # Replaces the candidate-pool count (= observed peptides, which makes
    # protein_coverage degenerate/leaky) with the true full tryptic digest count
    # per protein. Decoys strip their namespace prefix to inherit the source
    # protein's count, so protein_coverage is symmetric between a protein and its
    # decoy. See compute_protein_consistency_features for the matching numerator.
    if protein_full_tryptic_count:
        _base_prot = (
            candidates["protein"].astype(str)
            .str.replace(r"^DECOY_", "", regex=True)
            .str.replace(r"^ENTRAPMENT_", "", regex=True)
        )
        _mapped = _base_prot.map(protein_full_tryptic_count)
        # Keep any existing (candidate-pool) count only where the protein is not
        # in the digest map (e.g. LC-only novel peptides with no FASTA protein).
        if "protein_tryptic_count" in candidates.columns:
            _mapped = _mapped.fillna(candidates["protein_tryptic_count"])
        candidates["protein_tryptic_count"] = _mapped.fillna(0).astype(int)

    # --- Step 6: Compute all features ---
    logger.info("Step 6: Computing all features...")
    features_df = compute_all_features(
        candidates,
        spatial_features=spatial_features,
        ion_images=ion_images,
        ion_image_mzs=ion_image_mzs,
        extra_ion_images=extra_ion_images,
        maldi_envelopes=maldi_envelopes,
        observed_ccs_per_feature=observed_ccs_per_feature,
        im2deep_calibration=im2deep_calibration,
        im2deep_kwargs=im2deep_kwargs,
        coloc_tic_quantile=coloc_tic_quantile,
        coloc_measured_pixel_mask=coloc_measured_pixel_mask,
        coloc_tic_normalize=coloc_tic_normalize,
        coloc_common_mode=coloc_common_mode,
        cosine_coloc=cosine_coloc,
        tic_image=tic_image,
        tic_n_features=tic_n_features,
    )
    # Worst-case fill of protein-colocalization NaNs for zero-signal candidates, so a
    # feature with no MALDI signal is penalised rather than median-imputed to an average
    # coloc value (see _fill_nosignal_coloc_worst_case for the symmetry argument).
    features_df = _fill_nosignal_coloc_worst_case(features_df)
    # --- Optional zero-signal candidate removal (drop_zero_signal) ---
    # Under raw-query mode every candidate gets a genuine extraction attempt; a zero
    # feature_intensity_sum means no signal was detected at that m/z across all pixels.
    # Such candidates carry no MALDI evidence and, under mz_shuffle, their co-located
    # target/decoy pair diverge only on peptide_length (AUC 0.91 in the zero-signal
    # subpopulation), which leaks the mass-sorted derangement into the FDR.  Dropping
    # them is symmetric: the mask is is_decoy-blind (feature_intensity_sum is shared
    # between co-located target+decoy under mz_shuffle, and drawn from the same ion
    # image for all other decoy methods), so target and decoy counts drop in lock-step.
    if drop_zero_signal and "feature_intensity_sum" in features_df.columns:
        _no_signal = ~(features_df["feature_intensity_sum"] > 0)
        n_drop = int(_no_signal.sum())
        if n_drop:
            _n_t = int(_no_signal[~features_df["is_decoy"]].sum())
            _n_d = int(_no_signal[features_df["is_decoy"]].sum())
            features_df = features_df[~_no_signal].reset_index(drop=True)
            logger.info(
                f"  drop_zero_signal: removed {n_drop} zero-signal candidates "
                f"({_n_t} targets + {_n_d} decoys)."
            )
        else:
            logger.info("  drop_zero_signal: no zero-signal candidates found.")
    # --- CCS-based candidate filtering (optional) ---
    # IM2Deep finetuning (inside compute_all_features) uses the calibration-peptide
    # set as its CCS reference.  After finetuning, im2deep_abs_delta_ccs_pct is
    # available for all candidates.  We derive a data-driven threshold from the p95
    # calibration residual on that same set (analogous to
    # rt_window_min = rt_window_multiplier * p95_mae).
    #
    # ``ccs_window_pct`` overrides that with a fixed window, because the p95 is not a
    # fixed quantity: it is measured on the calibration peptides the run happens to
    # have, and a denser peak list adds calibration peptides sitting on noise peaks.
    # Measured on amyloidosis, the p95 went 3.18% at min_regions=2 to 4.06% at 1, so
    # the *same* multiplier loosens the window exactly where it needed tightening, and
    # two runs being compared do not share a threshold (PROGRESS.md F-050).
    _ccs_tol_pct: float | None = None
    if match_ccs:
        if observed_ccs_per_feature is not None and "im2deep_abs_delta_ccs_pct" in features_df.columns:
            _cal_mask = (
                features_df["is_calibration_peptide"]
                if "is_calibration_peptide" in features_df.columns
                else (features_df["n_candidates"] == 1)
            )
            _single_ccs = features_df.loc[_cal_mask, "im2deep_abs_delta_ccs_pct"].dropna()
            _ccs_tol_pct, _p95_ccs, _why = ccs_threshold_pct(
                _single_ccs, ccs_window_multiplier, ccs_window_pct
            )
            (logger.info if _ccs_tol_pct is not None else logger.warning)(_why)
            if _ccs_tol_pct is not None:
                n_before = len(features_df)
                _ccs_fail = (
                    features_df["im2deep_abs_delta_ccs_pct"].notna()
                    & (features_df["im2deep_abs_delta_ccs_pct"] > _ccs_tol_pct)
                )
                features_df = features_df[~_ccs_fail].reset_index(drop=True)
                n_after = len(features_df)
                logger.info(
                    f"  Removed {n_before - n_after} of {n_before} candidates "
                    f"({100*(n_before-n_after)/max(n_before,1):.1f}%). {n_after} remain."
                )
                # n_candidates / log_n_candidates are now stale; recompute per feature.
                _feat_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"
                features_df["n_candidates"] = (
                    features_df.groupby(_feat_col)[_feat_col].transform("count")
                )
                features_df["log_n_candidates"] = np.log1p(features_df["n_candidates"])
        else:
            logger.warning(
                "match_ccs=True but CCS filter cannot be applied: no observed CCS values "
                "were provided or IM2Deep features were not computed. Skipping CCS filter."
            )

    # --- Step 6c: per-candidate mobility-filtered colocalization (optional) ---
    _has_mob_coloc = False
    if mob_coloc and tdf_path is not None and "im2deep_predicted_ccs" in features_df.columns:
        try:
            from msi_picasso.maldi_features import compute_mobility_colocalization_features
            logger.info("Computing per-candidate mobility colocalization features (step 6c)…")
            features_df = compute_mobility_colocalization_features(
                features_df,
                tdf_path,
                mob_window_multiplier=mob_window_multiplier,
                extraction_ppm=ppm_tolerance,
                protein_coloc=mob_protein_coloc,
                tic_normalize=coloc_tic_normalize,
            )
            _has_mob_coloc = True
        except Exception as exc:
            logger.warning(f"Per-candidate mobility colocalization failed: {exc}. Skipping.")

    # Resolve the set of features explicitly excluded from the ranker: the
    # user-supplied features_exclude plus, for mz_shuffle, the raw CCS + mobility-
    # gated colocalization features that leak the m/z baseline (see the ranker
    # feature-pool assembly below for the rationale). Computed HERE, after
    # compute_mobility_colocalization_features above, not earlier: the mz_shuffle rule
    # matches by column name (_mz_shuffle_leaking_features), and the *_mob
    # colocalization columns (protein_colocalization_mob, adduct_colocalization_*_mob,
    # isotope_colocalization_*_mob, fraction_detected_mob, log_mean_intensity_mob,
    # spatial_morans_i_mob, intensity_cv_mob) do not exist in features_df.columns until
    # step 6c runs. Computing this earlier — as a prior version of this code did —
    # silently missed every one of them (PROGRESS.md F-016/E007: measured target/decoy
    # AUC as low as 0.0009 on the *_mob columns, with protein_colocalization_mob the
    # top-importance ranker feature and 2555/2583 amyloidosis targets passing at 1% FDR).
    # Computed once here so 13_debug_features.tsv reflects exactly the same exclusions
    # the ranker applies, and reused (not recomputed) when assembling the pool below.
    _exclude_set = set(features_exclude or [])

    # F-045 / H-decoy-17: the size-driven protein features are largely a readout of how
    # many tryptic peptides the protein has, not of whether it is present. Add a
    # size-free companion for each and put THAT in the ranker instead of the raw column.
    # Done here, above the debug-table write, so 13_debug_features.tsv records exactly
    # what the ranker saw (F-032).
    _size_resid_added: list[str] = []
    if protein_size_residualize:
        from msi_picasso.feature_generator import (
            PROTEIN_SIZE_RESID_SUFFIX,
            SIZE_DRIVEN_PROTEIN_FEATURES,
            residualize_against_protein_size,
        )
        _size_resid_added = residualize_against_protein_size(features_df)
        # A companion must not smuggle back a feature the config excluded. The raw
        # names are added to _exclude_set below, so the check has to be against what
        # the CONFIG asked for, not against the set this block is building.
        _bypassed = [
            c for c in _size_resid_added
            if c[: -len(PROTEIN_SIZE_RESID_SUFFIX)] in set(features_exclude or [])
        ]
        if _bypassed:
            logger.info(
                "  Not adding %d size-residualized column(s) whose raw form the config "
                "excludes: %s", len(_bypassed), _bypassed,
            )
            _size_resid_added = [c for c in _size_resid_added if c not in _bypassed]
        if _size_resid_added:
            _raw = [c[: -len(PROTEIN_SIZE_RESID_SUFFIX)] for c in _size_resid_added]
            _exclude_set |= set(_raw)
            # The seed search must follow the rename, or a config seeding on these
            # features finds none of them, fails, and returns a constant score --
            # which used to report every target as passing (F-044).
            if seed_features:
                seed_features = [
                    f + PROTEIN_SIZE_RESID_SUFFIX
                    if f + PROTEIN_SIZE_RESID_SUFFIX in _size_resid_added else f
                    for f in seed_features
                ]
                logger.info("  Seed features remapped to size-residualized: %s", seed_features)

    if decoy_method == "mz_shuffle":
        _ccs_mz_leak_feats = _mz_shuffle_leaking_features(features_df.columns)
        # The by-construction rule must never exclude less than the old explicit list did.
        _missed = (_MZ_SHUFFLE_CCS_LEAK_FEATURES & set(features_df.columns)) - _ccs_mz_leak_feats
        assert not _missed, f"mz_shuffle leak guard regressed; missed {sorted(_missed)}"
        # If mobility colocalization ran (step 6c, immediately above), at least one *_mob
        # column must be present and caught. This is the exact failure this block once had
        # silently: the guard ran before the *_mob columns existed and excluded nothing.
        # Re-ordering this block again in the future would reproduce that bug; this assert
        # turns it back into a loud failure instead of a silent one (F-016/E007).
        if _has_mob_coloc:
            _mob_cols_present = {c for c in features_df.columns if c.endswith("_mob")}
            assert _mob_cols_present & _ccs_mz_leak_feats, (
                "mz_shuffle leak guard ran before mobility colocalization columns were "
                "visible; see PROGRESS.md F-016/E007"
            )
        # theo_isotope_kl and siblings (MZ_SHUFFLE_OWN_MASS_ENVELOPE_FEATURES) are always
        # present once compute_theoretical_isotope_features has run, unconditionally on
        # any config, so this assert is not gated the way the mob_coloc one is.
        from msi_picasso.maldi_features import MZ_SHUFFLE_OWN_MASS_ENVELOPE_FEATURES
        _iso_present = set(features_df.columns) & MZ_SHUFFLE_OWN_MASS_ENVELOPE_FEATURES
        if _iso_present:
            assert _iso_present <= _ccs_mz_leak_feats, (
                "mz_shuffle leak guard missed the own-mass isotope-envelope features "
                f"{sorted(_iso_present - _ccs_mz_leak_feats)}; see PROGRESS.md F-020"
            )
        if _ccs_mz_leak_feats - _exclude_set:
            logger.info(
                "  decoy_method='mz_shuffle': excluding %d mobility-gated, predicted-CCS, "
                "and own-mass isotope-envelope features from the ranker (they leak the m/z "
                "baseline, PROGRESS.md F-016/F-020). The *_resid CCS variants are excluded "
                "too: detrending does NOT make them safe (measured AUC 0.76-0.86). "
                "theo_isotope_kl and siblings correlate with the decoy's construction mass "
                "gap at Spearman 0.70-0.80 (measured on all three ground-truth datasets). "
                "Excluded: %s",
                len(_ccs_mz_leak_feats), sorted(_ccs_mz_leak_feats),
            )
        _exclude_set |= _ccs_mz_leak_feats
    if _exclude_set:
        logger.info(f"  Excluding {len(_exclude_set)} features: {sorted(_exclude_set)}")

    # Log-transform heavy-tail features in place.  These features span 4+
    # orders of magnitude on real data; after StandardScaler the few extreme
    # values dominate and suppress discrimination from well-behaved features.
    #
    # This MUST stay above the 13_debug_features.tsv write below.  It used to sit
    # after it, so the debug table held the untransformed columns while the ranker
    # saw the transformed ones -- and that table is what every offline analysis in
    # PROGRESS.md reads.  A refit from it diverged from the run it was supposed to
    # reproduce (kidney q-value floor 0.0222 against the run's 0.0536) until the
    # transform was applied by hand in the analysis script.
    _HEAVY_TAIL_FEATURES = (
        "chca_cluster_distance_ppm",
        "theo_isotope_chi2",
        "ppm_best_ratio",
        "theo_m1_ratio_diff",
        "theo_m2_ratio_diff",
    )
    for _f in _HEAVY_TAIL_FEATURES:
        if _f in features_df.columns:
            features_df[_f] = np.log1p(
                np.clip(features_df[_f].values.astype(float), 0.0, None)
            )

    if verbose:
        logger.debug(f"Writing computed features to {output_dir}/13_debug_features.tsv")
        _debug_cols = [c for c in features_df.columns if c not in _exclude_set]
        features_df[_debug_cols].to_csv(f"{output_dir}/13_debug_features.tsv", sep="\t", index=False)

    # Intrinsic features that are actually present in the DataFrame.
    # Protein-level features are excluded by default (TDC correctness); opt-in
    # only when use_protein_level_features=True (--use-protein-level-feats).
    # Feature list assembly: preset → exclude → optional protein-level
    _use_main = only_main_features or features_preset == "main"
    _base_features = MAIN_FEATURES if _use_main else MALDI_INTRINSIC_FEATURES
    if _use_main:
        logger.info(
            f"  Features preset 'main': using {len(MAIN_FEATURES)} representative features "
            f"(vs {len(MALDI_INTRINSIC_FEATURES)} in the full set)"
        )

    # _exclude_set (features_exclude + the mz_shuffle CCS/mobility leak features) was
    # resolved above, after mob_coloc and after 13_debug_features.tsv was written, so
    # the debug table and the ranker apply identical exclusions. See that block for the
    # mz_shuffle rationale and why the resolution point matters.
    # Assemble the intrinsic feature pool: base + optional protein-level + optional
    # spatial-ranker.  protein_colocalization_* appear in both PROTEIN_LEVEL_FEATURES
    # and SPATIAL_RANKER_FEATURES; the order-preserving dedup below prevents
    # double-inclusion when both flags are active.
    _pool = list(_base_features)
    if use_protein_level_features:
        _pool += PROTEIN_LEVEL_FEATURES
    if use_spatial_ranker_features:
        _pool += SPATIAL_RANKER_FEATURES
        logger.info(
            f"  Spatial ranker features enabled ({len(SPATIAL_RANKER_FEATURES)} features) "
            f"with decoy_method='{decoy_method}'"
        )
    if cosine_coloc:
        _pool += COSINE_COLOCALIZATION_FEATURES
        logger.info(
            f"  Median-thresholded cosine colocalization features enabled "
            f"({len(COSINE_COLOCALIZATION_FEATURES)} features, H-feat-3/H-decoy-9)"
        )
    # F-045: the size-free companions computed above. Their raw counterparts were added
    # to _exclude_set at the same time, so the pool carries one or the other, never both.
    if _size_resid_added:
        _pool += _size_resid_added
        logger.info(
            f"  Protein-size residualized features enabled ({len(_size_resid_added)}): "
            f"{_size_resid_added}. Their raw counterparts are excluded (PROGRESS.md F-045)."
        )
    # Intrinsic 2D peak-quality features: default-on. For substitution the decoy sits at
    # a distinct (often empty or noisy) m/z, so feature-level peak quality discriminates.
    # For mz_shuffle the co-located target and decoy share the feature, so these columns
    # are exactly symmetric (AUC ~= 0.5), which is harmless. They use only the observed
    # peak and no prediction, so unlike the im2deep_* CCS scalars they do not leak the
    # m/z baseline under mz_shuffle. Only present when extracted (raw-query + ion
    # mobility); the intrinsic_present intersection below drops them otherwise. Drop
    # explicitly via features_exclude if unwanted.
    if any(f in features_df.columns for f in MOB_QUALITY_FEATURES):
        _pool += MOB_QUALITY_FEATURES
        logger.info(
            f"  2D peak-quality features enabled ({len(MOB_QUALITY_FEATURES)} features) "
            f"with decoy_method='{decoy_method}'"
        )
    # Mass-normalized isotope-envelope features (H-decoy-7a): the safe replacement for
    # theo_isotope_kl/theo_m1_ratio_diff/theo_m2_ratio_diff, which F-020 found leak the
    # mz_shuffle construction mass gap and are excluded under mz_shuffle by
    # _mz_shuffle_leaking_features. Gated to mz_shuffle only -- other decoy methods already
    # have a working, non-leaking theo_isotope_kl and do not need this experimental variant.
    if decoy_method == "mz_shuffle" and any(
        f in features_df.columns for f in MZ_SHUFFLE_MASSNORM_ISOTOPE_FEATURES
    ):
        _pool += MZ_SHUFFLE_MASSNORM_ISOTOPE_FEATURES
        logger.info(
            f"  Mass-normalized isotope-envelope features enabled "
            f"({len(MZ_SHUFFLE_MASSNORM_ISOTOPE_FEATURES)} features) with "
            f"decoy_method='{decoy_method}' (H-decoy-7a)"
        )
    _seen: set[str] = set()
    _intrinsic_pool = [
        f for f in _pool
        if f not in _exclude_set and not (f in _seen or _seen.add(f))
    ]
    intrinsic_present = [f for f in _intrinsic_pool if f in features_df.columns]

    # Drop constant / near-constant features from the ranker input.  A column
    # with one unique value contributes zero variance and only consumes a slot
    # in the SVM grid search; in pathological CV folds it can also produce
    # warnings.
    _constant = [f for f in intrinsic_present if features_df[f].nunique(dropna=True) <= 1]
    if _constant:
        logger.info(f"  Dropping {len(_constant)} constant features from ranker: {_constant}")
        intrinsic_present = [f for f in intrinsic_present if f not in _constant]

    logger.info(f"  {len(intrinsic_present)} ranker features")

    # --- Step 8: Rescoring ---
    logger.info(f"Step 8: Running rescoring (model='{model}')...")
    if model not in _ESTIMATOR_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose one of {sorted(_ESTIMATOR_MODELS)}.")
    feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

    # Fitted-pipeline capture for SHAP debug explanations (see debug_pfm_explanations).
    fitted: dict = {}
    scores_all, importances, struct_coefs, struct_names, imp_names = _rescore_linear(
        features_df,
        intrinsic_present,
        init_ppm_threshold=init_ppm_threshold,
        init_fdr=init_fdr,
        train_fdr=train_fdr,
        max_iter=max_iter,
        r1_seed_percentile=r1_seed_percentile,
        min_seed_positives=min_seed_positives,
        seed_features=seed_features,
        clf_name=model,
        make_clf=_estimator_factory(
            model, svm_c=svm_c, rbf_svm_c=rbf_svm_c, rbf_svm_gamma=rbf_svm_gamma
        ),
        fitted_out=fitted,
        train_fdr_escalate=train_fdr_escalate,
        pseudo_label_growth_cap=pseudo_label_growth_cap,
        model_repeats=model_repeats,
    )
    if verbose:
        with open(f"{output_dir}/17_debug_{model}_scores_r1.pkl", "wb") as f:
            pickle.dump(scores_all, f)
        _imp_df = pd.DataFrame({"feature": imp_names, "importance": importances})
        if struct_coefs is not None and struct_names:
            _imp_df = _imp_df.merge(
                pd.DataFrame({"feature": struct_names, "structure_coef": struct_coefs}),
                on="feature", how="left",
            )
        _imp_df.sort_values("importance", ascending=False).to_csv(
            f"{output_dir}/17_debug_{model}_importances_r1.tsv", sep="\t", index=False
        )

    # --- Per-feature winner selection, then TDC over the winners ---
    # Winner selection is the target-vs-decoy competition that defines the TDC
    # population; the winners keep the score they were selected on.
    winner_pos, winners_df = _select_feature_winners(features_df, scores_all, feature_col, winner_percentile)
    logger.info(
        f"  Winner selection: {len(winners_df)} candidates retained "
        f"({int(winners_df['is_decoy'].sum())} decoys)"
    )
    scores_w = scores_all[winner_pos]
    is_decoy_w = winners_df["is_decoy"].values.astype(bool)
    q_w = _tdc_qvalues(scores_w, is_decoy_w)
    pep_w = estimate_pep(scores_w, is_decoy_w)
    peptide_q_w, is_pep_rep_w = _peptide_level_qvalues(
        scores_w, is_decoy_w, winners_df["peptide"].to_numpy()
    )

    def _scatter(values, fill=np.nan):
        """Place per-winner ``values`` onto every candidate row; ``fill`` elsewhere."""
        out = np.full(len(features_df), fill)
        out[winner_pos] = values
        return out

    is_decoy = features_df["is_decoy"].values.astype(bool)
    is_winner_full = _scatter(True, fill=False)
    q_full = _scatter(q_w)
    peptide_q_full = _scatter(peptide_q_w)
    is_pep_winner_full = _scatter(is_pep_rep_w, fill=False)
    result_df = pd.DataFrame(
        {
            "peptide": features_df["peptide"].values,
            "protein": features_df["protein"].values if "protein" in features_df.columns else "",
            "feature_mz": features_df["feature_mz"].values if "feature_mz" in features_df.columns else np.nan,
            "feature_idx": features_df.get(
                "feature_idx", pd.Series(range(len(features_df)))
            ).values,
            "is_decoy": is_decoy,
            f"{model}_score_r1": scores_all,
            "score": _scatter(scores_w),
            "q_value": q_full,
            "pep": _scatter(pep_w),
            "pep_q_value": _scatter(_pep_qvalues(pep_w)),
            "is_tdc_winner": is_winner_full,
            # Peptide-level FDR: one hypothesis per peptide, not per
            # (peptide, feature) match. `peptide_q_value` is broadcast to
            # every match of a peptide; `is_peptide_winner` marks the single
            # best-scoring match that represents it. Count IDs as
            # `is_peptide_winner & ~is_decoy & peptide_q_value <= alpha`.
            "peptide_q_value": peptide_q_full,
            "is_peptide_winner": is_pep_winner_full,
        }
    )
    _ccs_map = observed_ccs_per_feature or {}
    result_df["feature_ccs"] = [
        _ccs_map.get(int(i), np.nan) for i in result_df["feature_idx"]
    ]

    for fdr_threshold in [0.01, 0.05, 0.10]:
        n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
        n_pep = (is_pep_winner_full & ~is_decoy & (peptide_q_full <= fdr_threshold)).sum()
        # The first line counts (peptide, feature) matches. In feature-list
        # extraction one peptide matches several features, so that is not a
        # count of identified peptides -- the second line is.
        logger.info(f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (feature-level FDR)")
        logger.info(
            f"  At {fdr_threshold*100:.0f}% FDR: {n_pep} target PEPTIDES "
            f"(peptide-level FDR)"
        )

    if debug_dir is not None:
        from msi_picasso.debug_viz import debug_pfm_explanations, save_debug_figures

        save_debug_figures(
            features_df, result_df,
            ion_images=ion_images, ion_image_mzs=ion_image_mzs,
            maldi_envelopes=maldi_envelopes,
            feature_names=intrinsic_present, model_name=model,
            importances=importances,
            importance_names=imp_names,
            structure_coefs=struct_coefs,
            structure_names=struct_names,
            debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
            gt_peptides=gt_peptides, ccs_tol_pct=_ccs_tol_pct,
        )

        # Per-PFM SHAP explanations on the fitted model, restricted to winner rows.
        if verbose and fitted.get("pipe") is not None:
            try:
                debug_pfm_explanations(
                    result_df.iloc[winner_pos].reset_index(drop=True),
                    fitted["X"][winner_pos], fitted["pipe"], fitted["feature_names"],
                    ion_images=ion_images, feature_mzs=ion_image_mzs,
                    output_dir=debug_dir,
                )
            except Exception as _pfm_exc:
                logger.warning("debug_pfm_explanations failed: %s", _pfm_exc)

    if entrapment:
        _report_entrapment(result_df, features_df, output_dir)
    return result_df, features_df
