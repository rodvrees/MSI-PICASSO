"""
MSI-PICASSO feature generator: orchestrates all feature categories.

Adds every rescoring feature column to a candidate DataFrame (from
match_to_maldi_features).
"""

import logging

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import (
    _pearson_r_matrix,
    _median_thresholded_cosine_matrix,
    compute_tissue_mask,
    compute_adduct_colocalization,
    compute_candidate_ambiguity_features,
    compute_chca_cluster_features,
    compute_colocalization_features,
    compute_cosine_colocalization_features,
    compute_im2deep_features,
    compute_isotopologue_colocalization,
    compute_maldi_ionization_features,
    compute_maldi_signal_features,
    compute_mass_accuracy_features,
    compute_mass_defect_features,
    compute_peptide_properties,
    compute_peptide_property_features,
    compute_protein_consistency_features,
    compute_spatial_autocorrelation_full,
    compute_spatial_features,
    compute_theoretical_isotope_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# MALDI-intrinsic features: computed entirely from MALDI data and in-silico
# properties. These are the sole input to the ranker/SVM so that the model
# scores MALDI match quality, not LC-MS/MS identification quality.
#
# Optional features are listed here too; the pipeline keeps only the columns that
# were actually computed.
MALDI_INTRINSIC_FEATURES = [
    # --- mass accuracy (A-group) ---
    "ppm_error_abs", "ppm_rank", "ppm_best_ratio", "log_ppm_best_ratio",
    "ppm_error_pct", "ppm_error_squared",
    # --- ambiguity ---
    "n_candidates", "log_n_candidates",
    # --- peptide properties ---
    "peptide_length", "n_missed_cleavages",
    "has_oxidized_met", "has_cys", "n_proline",
    "acidic_residue_density",
    # --- MALDI signal ---
    "log_maldi_intensity_p90", "log_maldi_intensity_sum",
    # --- mass defect features (A-group) ---
    "kendrick_mass_defect",          # A10 — computed in match_to_maldi_features
    "mass_defect_residual",          # A11
    # --- CHCA matrix interference ---
    "chca_cluster_distance_ppm",     # A12
    # --- theoretical isotope ---
    "theo_isotope_cosine", "theo_isotope_chi2", "theo_isotope_kl",
    "theo_has_sulfur", "averagine_deviation", "averagine_deviation_sulfur",
    "theo_m1_ratio_diff", "theo_m2_ratio_diff",
    # H-decoy-13: the same envelope comparison referenced to averagine at the matched
    # peak's mass rather than to the candidate's own formula, so it cannot read the
    # composition asymmetry `substitution` creates (PROGRESS.md F-036/F-039). Value is
    # per-peak, identical for every candidate on it, so it ranks peaks rather than
    # separating candidates competing for one.
    "averagine_envelope_kl",
    "averagine_envelope_m1_ratio_diff", "averagine_envelope_m2_ratio_diff",
    "monoisotopic_confidence",       # A8
    # --- ionization priors ---
    "n_arginine", "n_basic_residues", "n_aromatic",
    "gravy_score", "charge_proxy",
    # --- ion mobility (B-group) — optional, requires im2deep + observed CCS ---
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank",
    # m/z-detrended (conformational) CCS residuals — for mz_shuffle these REPLACE
    # the raw CCS features above (pipeline excludes the raw ones); for other decoy
    # methods they supplement them.
    "im2deep_delta_ccs_resid", "im2deep_abs_delta_ccs_pct_resid",
    "im2deep_ccs_zscore_resid", "im2deep_ccs_rank_resid",
    # --- isotopologue co-localization (E1) — optional, requires ion_images ---
    "isotope_image_colocalization_m1", "isotope_image_colocalization_m2",
    "isotope_image_colocalization_mean",
    # --- adduct co-localization (E2) — optional, requires ion_images ---
    "adduct_colocalization_na", "adduct_colocalization_k", "adduct_colocalization_chca",
    # --- per-candidate mobility-filtered colocalization — optional, requires tdf_path + im2deep ---
    "isotope_colocalization_m1_mob", "isotope_colocalization_m2_mob",
    "isotope_colocalization_mean_mob",
    "adduct_colocalization_na_mob", "adduct_colocalization_k_mob",
    "adduct_colocalization_chca_mob",
]

# Protein-level features whose value is largely a readout of how many tryptic peptides
# the protein has, rather than of whether the protein is present (PROGRESS.md F-045).
# In feature-list mode a candidate is kept only if it matches a detected peak, but a
# 10 ppm window against 35-54k peaks matches targets and decoys at the same rate
# (67.7% against 67.9% on amyloidosis, 50.7% against 50.7% on kidney), so "peptides of
# this protein that matched" is mostly "peptides this protein has". Measured Spearman
# against protein_tryptic_count: log_protein_n_features and
# protein_colocalization_n_partners +0.90 to +0.95, protein_coverage -0.46 to -0.51,
# is_single_peptide_protein -0.63 to -0.66, protein_best_ratio +0.34 to +0.41.
# The protein_colocalization_* family proper is deliberately NOT here: removing size
# from those sharpens them, while removing it from these strips their ground-truth
# signal on kidney and her2 (F-045).
SIZE_DRIVEN_PROTEIN_FEATURES = [
    "log_protein_n_features",
    "protein_n_features",
    "protein_colocalization_n_partners",
    "protein_coverage",
    "is_single_peptide_protein",
    "protein_best_ratio",
]

PROTEIN_SIZE_COLUMN = "protein_tryptic_count"
PROTEIN_SIZE_RESID_SUFFIX = "_sizeresid"


def residualize_against_protein_size(
    features_df,
    columns=None,
    size_column=PROTEIN_SIZE_COLUMN,
    n_bins=12,
    suffix=PROTEIN_SIZE_RESID_SUFFIX,
):
    """Add a size-free companion column for each size-driven protein feature.

    Each value is replaced by its rank *within a bin of the protein's tryptic
    count*, scaled to [0, 1]. That removes any monotone dependence on protein size
    without assuming a functional form, and leaves whatever else the feature
    carries.

    Bins are built from targets and decoys pooled and the function never sees
    ``is_decoy`` (invariant 1): the transform is identical for both classes and so
    cannot itself separate them. Measured after the fact on E024, residualized
    columns keep a target/decoy AUC within 0.02 of the raw ones.

    New columns are ADDED rather than substituted, following F-039's precedent with
    the averagine envelope, so both versions stay measurable side by side and
    ``scripts/audit_protein_size.py`` can check each. Returns the list of names added.
    """
    from scipy.stats import rankdata

    if size_column not in features_df.columns:
        logger.warning(
            "protein-size residualization skipped: no %s column", size_column
        )
        return []

    size = features_df[size_column].to_numpy(dtype=float)
    try:
        bins = pd.qcut(pd.Series(size), n_bins, labels=False, duplicates="drop").to_numpy()
    except ValueError:                      # too few distinct sizes to bin
        bins = np.zeros(len(size), dtype=float)

    added = []
    for col in (columns if columns is not None else SIZE_DRIVEN_PROTEIN_FEATURES):
        if col not in features_df.columns:
            continue
        values = features_df[col].to_numpy(dtype=float)
        out = np.full(values.shape, np.nan)
        for b in np.unique(bins[~pd.isna(bins)]):
            m = (bins == b) & np.isfinite(values)
            if m.sum() < 5:
                continue
            out[m] = (rankdata(values[m]) - 0.5) / m.sum()
        # A bin too small to rank leaves NaN; fall back to the global rank there so
        # the column is never mostly-NaN on a dataset with few distinct protein sizes.
        gap = ~np.isfinite(out) & np.isfinite(values)
        if gap.any():
            out[gap] = (rankdata(values[gap]) - 0.5) / gap.sum()
        features_df[col + suffix] = out
        added.append(col + suffix)

    logger.info(
        "protein-size residualization: added %d column(s) (%s)",
        len(added), ", ".join(added) if added else "none",
    )
    return added


# Protein-level features: aggregate signal across all candidates sharing a protein,
# including decoys. This breaks the TDC null model (decoys inherit inflated counts
# from target co-occurring proteins), so these features are excluded from the ranker
# by default and only used when --use-protein-level-feats is explicitly requested.
PROTEIN_LEVEL_FEATURES = [
    # protein consistency (from compute_protein_consistency_features)
    "protein_n_features", "log_protein_n_features", "protein_coverage",
    "protein_rank", "protein_best_ratio",
    # structural indicator: 1.0 when the protein has a single observed peptide, so
    # within-protein colocalization is undefined (median-imputed) and
    # log_protein_n_features is at its floor.  Lets the ranker treat these peptides
    # as a separate group and rely on their intrinsic evidence (see O8).
    "is_single_peptide_protein",
    # protein co-localization (from compute_colocalization_features; requires ion_images)
    "protein_colocalization", "protein_colocalization_max",
    "protein_colocalization_median", "protein_colocalization_n_partners",
    # mobility-gated protein co-localization (from compute_mobility_colocalization_features)
    "protein_colocalization_mob", "protein_colocalization_mob_max",
    "protein_colocalization_mob_n_partners",
    # indicator: 1.0 when within-protein colocalization is defined (>=1 partner),
    # 0.0 otherwise.  Lets the ranker separate "not colocalizable" (small protein,
    # coloc median-imputed) from "colocalizes poorly" (see compute_colocalization_features).
    "has_coloc",
    # intensity-weighted and rank-weighted (top-k) within-protein colocalization
    "protein_colocalization_weighted", "protein_colocalization_weighted_max",
    "protein_colocalization_top2", "protein_colocalization_top3", "protein_colocalization_top5",
]

# Median-thresholded cosine colocalization (opt-in via --cosine-coloc, requires
# ion_images). Ovchinnikova et al. (2020, ColocML): median-thresholded cosine
# similarity of raw ion images, validated at Spearman 0.794 against 42 expert
# raters (matching a trained deep model), as a replacement for the
# Pearson-plus-TIC-mask colocalization above (PROGRESS.md H-feat-3 / H-decoy-9).
# Takes only the observed ion images -- no candidate mass or composition -- so it
# is safe by construction under every decoy method, mz_shuffle included: there is
# no F-020-style leak vector to audit for. Protein-level, so valid only because
# decoys occupy a separate protein namespace (see PROTEIN_LEVEL_FEATURES above).
COSINE_COLOCALIZATION_FEATURES = [
    "protein_colocalization_cosine",
    "protein_colocalization_cosine_max",
    "protein_colocalization_cosine_median",
]

# Spatial ranker features: opt-in (--use-spatial-ranker-features). Both remaining
# decoy methods put each decoy on a real MALDI feature, so these have a symmetric
# null. The protein_colocalization_* members overlap PROTEIN_LEVEL_FEATURES;
# pipeline.py deduplicates when both opt-in flags are active.
SPATIAL_RANKER_FEATURES = [
    # Feature-level spatial quality
    "spatial_autocorrelation",
    "spatial_morans_i",
    "spatial_gearys_c",
    "fraction_detected",
    "intensity_cv",
    # Protein-level colocalization
    "protein_colocalization",
    "protein_colocalization_max",
    "protein_colocalization_median",
    "protein_colocalization_n_partners",
]

# Intrinsic 2D peak-quality features (m/z × intensity × 1/K0), computed per MALDI
# feature from the observed peaks in raw-query mode (maldi_query._peak_quality_in_windows).
# Feature-level and observation-only (no prediction → no m/z-baseline leak).  Kept OUT of
# MALDI_INTRINSIC_FEATURES so they are not a global default for every decoy method; the
# pipeline appends them to the ranker pool only for decoy methods in
# _MOB_QUALITY_DEFAULT_DECOYS (substitution, mz_shuffle).
MOB_QUALITY_FEATURES = [
    "mob_2d_concentration",
    "mob_k0_spread",
    "mob_mz_spread_ppm",
    "mob_peak_snr",
]

# Mass-normalized isotope-envelope features (PROGRESS.md H-decoy-7a): theo_isotope_kl and
# siblings build the theoretical envelope from the candidate's OWN mass, which leaks under
# mz_shuffle because the derangement deliberately places a decoy far away in mass from its
# assigned feature (F-020, Spearman 0.70-0.80 with the construction mass gap). These
# variants rescale the candidate's elemental composition to the mass IMPLIED BY THE ASSIGNED
# FEATURE before building the theoretical envelope, removing the raw mass confound while
# retaining whatever composition-type signal remains. Experimental: may prove
# symmetric-but-uninformative like MOB_QUALITY_FEATURES rather than genuinely discriminative
# — kept OUT of MALDI_INTRINSIC_FEATURES and gated to mz_shuffle only in pipeline.py, mirroring
# how MOB_QUALITY_FEATURES is gated to _MOB_QUALITY_DEFAULT_DECOYS.
MZ_SHUFFLE_MASSNORM_ISOTOPE_FEATURES = [
    "theo_isotope_kl_massnorm",
    "theo_m1_ratio_diff_massnorm",
    "theo_m2_ratio_diff_massnorm",
]

# Alias kept separate so LDA-specific feature selection can diverge later.
LDA_FEATURES = MALDI_INTRINSIC_FEATURES

# Reduced feature set: one representative per collinear group.
# Use with --features-preset main to cut the feature count from ~46 to ~19
# and remove inter-feature redundancy before training.
MAIN_FEATURES = [
    # ppm (from 6 → 1)
    "ppm_error_abs",
    # isotope_theo (from 5 → 1)
    "theo_isotope_cosine",
    # sequence_comp (from 8 → 2)
    "gravy_score",
    "peptide_length",
    # adduct (keep all 3, not collinear across adduct types)
    "adduct_colocalization_na",
    "adduct_colocalization_k",
    "adduct_colocalization_chca",
    # chca
    "chca_cluster_distance_ppm",
    # maldi_intensity (from 2 → 1)
    "log_maldi_intensity_p90",
    # im2deep (from 4 → 1)
    "im2deep_ccs_zscore",
    # averagine (from 2 → 1)
    "averagine_deviation",
    # monoisotopic
    "monoisotopic_confidence",
    # isotope_image (from 3 → 1)
    "isotope_image_colocalization_m2",
    # candidates (from 2 → 1)
    "n_candidates",
    # sulfur (from 2 → 1)
    "theo_has_sulfur",
    # mass_defect (from 2 → 1)
    "kendrick_mass_defect",
    # other (keep both, not collinear)
    "has_oxidized_met",
    "n_missed_cleavages",
]

# Feature-specific NaN fill values applied before the generic median imputer.
# Missing values for these features have a known structural meaning, so filling
# with the column median would be misleading.
#
# Values:
#   float      — fill with that constant
#   "col_max"  — fill with np.nanmax of that column (worst-case penalty)
#   "col_min"  — fill with np.nanmin of that column
FEATURE_NAN_FILL: dict[str, float | str] = {
    # 2D peak quality: a candidate whose m/z window has no observed peak (e.g. a
    # substitution decoy in empty m/z) gets the worst-case, not the median.
    "mob_2d_concentration": 0.0,      # no peak → zero concentration
    "mob_peak_snr": "col_min",        # no peak → lowest observed signal contrast
    "mob_k0_spread": "col_max",       # no peak → widest (worst) mobility spread
    "mob_mz_spread_ppm": "col_max",   # no peak → widest (worst) m/z spread
    # Mass-normalized isotope envelope (H-decoy-7a): no observed envelope at the assigned
    # feature → worst-case, matching theo_m1_ratio_diff/theo_m2_ratio_diff's existing
    # col_max treatment. theo_isotope_kl_massnorm defaults to 0.0 (perfect match) when no
    # envelope exists, same as the unnormalized theo_isotope_kl -- that is a pre-existing
    # asymmetry in the original feature, not introduced here.
    "theo_m1_ratio_diff_massnorm": "col_max",
    "theo_m2_ratio_diff_massnorm": "col_max",
    # H-decoy-13, same treatment and same pre-existing asymmetry: the ratio diffs go to
    # worst-case when no envelope exists, averagine_envelope_kl stays 0.0 like the
    # unnormalized theo_isotope_kl it parallels.
    "averagine_envelope_m1_ratio_diff": "col_max",
    "averagine_envelope_m2_ratio_diff": "col_max",
}

def compute_all_features(
    candidates_df: pd.DataFrame,
    spatial_features: pd.DataFrame | None = None,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    extra_ion_images: dict | None = None,
    maldi_envelopes: dict | None = None,
    observed_ccs_per_feature: dict | None = None,
    im2deep_calibration: str = "linear",
    im2deep_kwargs: dict | None = None,
    coloc_tic_quantile: float = 0.0,
    tic_image: np.ndarray | None = None,
    tic_n_features: int | None = None,
    coloc_measured_pixel_mask: "np.ndarray | None" = None,
    coloc_tic_normalize: bool = False,
    coloc_common_mode: bool = False,
    cosine_coloc: bool = False,
) -> pd.DataFrame:
    """
    Compute all features on the candidate DataFrame.

    Parameters
    ----------
    candidates_df
        Output of match_to_maldi_features().
    spatial_features
        Pre-computed per-feature spatial statistics DataFrame (optional).
    ion_images
        MALDI ion images array, shape (n_features, H, W) (optional).
    ion_image_mzs
        m/z values aligned with ion_images (optional).
    maldi_envelopes
        MALDI isotope envelopes: feature_mz → array (optional).
    observed_ccs_per_feature
        Dict mapping feature_idx → observed CCS value for IM2Deep features (optional).

    Returns
    -------
    DataFrame with all feature columns added.
    """
    df = candidates_df.copy()

    # --- Always-computed features ---
    df = compute_mass_accuracy_features(df)
    df = compute_candidate_ambiguity_features(df)
    df = compute_protein_consistency_features(df)
    df = compute_peptide_properties(df)
    df = compute_peptide_property_features(df)          # C-group
    df = compute_maldi_signal_features(df)
    df = compute_mass_defect_features(df)               # A11
    df = compute_chca_cluster_features(df)              # A12

    # --- Theoretical isotope (adds monoisotopic_confidence, A8) ---
    df = compute_theoretical_isotope_features(
        df,
        maldi_envelopes=maldi_envelopes,
    )

    # --- MALDI ionization ---
    df = compute_maldi_ionization_features(df)

    # --- B: IM2Deep CCS features (optional) ---
    if observed_ccs_per_feature is not None:
        logger.debug("Computing IM2Deep CCS features (B-group) using observed CCS values")
        df = compute_im2deep_features(
            df,
            observed_ccs_per_feature=observed_ccs_per_feature,
            calibration_method=im2deep_calibration,
            im2deep_kwargs=im2deep_kwargs,
        )

    # --- Spatial (optional) ---
    if spatial_features is not None:
        logger.debug("Computing spatial features (A3/A4) using pre-computed spatial_features DataFrame")
        df = compute_spatial_features(df, spatial_features)

    # --- Ion-image-based features (optional) ---
    if ion_images is not None and ion_image_mzs is not None:
        logger.debug("Computing ion-image-based features (co-localization, full spatial autocorrelation, etc.) using ion_images and ion_image_mzs")
        # On-tissue pixel mask (TIC proxy): raw MALDI ion images share a dominant
        # on/off-tissue component that inflates every pairwise r toward the tissue
        # outline. Restricting the correlation to on-tissue pixels removes it so
        # colocalization reflects co-distribution within the tissue (see
        # compute_tissue_mask). TIC == 0 padding is always dropped.
        pixel_mask = compute_tissue_mask(
            ion_images, tic_quantile=coloc_tic_quantile, tic_image=tic_image
        )
        if coloc_measured_pixel_mask is not None:
            pixel_mask = pixel_mask & coloc_measured_pixel_mask
        _mask_suffix = ", measured-coord mask applied" if coloc_measured_pixel_mask is not None else ""
        logger.info(
            f"Colocalization on-tissue mask: {int(pixel_mask.sum())}/{pixel_mask.size} "
            f"pixels kept (tic_quantile={coloc_tic_quantile}{_mask_suffix})"
        )
        # Compute the full Pearson correlation matrix once (single BLAS call) and
        # share it across the isotopologue/adduct colocalization functions to avoid
        # redundant work.  Those correlate an M0 image against its own
        # isotopologue/adduct image, where common-mode removal is not meaningful, so
        # they always use the raw cache.
        corr_cache = _pearson_r_matrix(ion_images, ion_image_mzs, pixel_mask=pixel_mask)
        # The cross-feature protein colocalization optionally uses a preprocessed
        # cache (per-pixel TIC normalization / common-mode removal) to strip the
        # shared tissue envelope; only built when a toggle is set to avoid a second
        # BLAS pass in the default case.
        if coloc_tic_normalize or coloc_common_mode:
            protein_corr_cache = _pearson_r_matrix(
                ion_images, ion_image_mzs, pixel_mask=pixel_mask,
                tic_normalize=coloc_tic_normalize, common_mode_removal=coloc_common_mode,
                full_tic=tic_image, full_n_features=tic_n_features,
            )
        else:
            protein_corr_cache = corr_cache
        df = compute_colocalization_features(df, ion_images, ion_image_mzs, _corr_cache=protein_corr_cache)
        if cosine_coloc:
            cosine_corr_cache = _median_thresholded_cosine_matrix(ion_images, ion_image_mzs, pixel_mask=pixel_mask)
            df = compute_cosine_colocalization_features(df, ion_images, ion_image_mzs, _corr_cache=cosine_corr_cache)
        df = compute_isotopologue_colocalization(df, ion_images, ion_image_mzs, _corr_cache=corr_cache, extra_ion_images=extra_ion_images, pixel_mask=pixel_mask)  # E1
        df = compute_adduct_colocalization(df, ion_images, ion_image_mzs, _corr_cache=corr_cache, extra_ion_images=extra_ion_images, pixel_mask=pixel_mask)        # E2
        df = compute_spatial_autocorrelation_full(df, ion_images, ion_image_mzs)                         # E5/E6

    return df

