"""The mz_shuffle leak guard must be by construction, not by enumeration.

PROGRESS.md F-016: the previous enumerated `_MZ_SHUFFLE_CCS_LEAK_FEATURES` covered only the
isotope_*_mob and adduct_*_mob families. `protein_colocalization_mob_max` (target/decoy AUC
0.0003) reached the ranker as its top feature and passed 5070/5070 kidney targets at 1% FDR.
"""

from msi_picasso.pipeline import (
    _MZ_SHUFFLE_CCS_LEAK_FEATURES,
    _mz_shuffle_leaking_features,
)

# Columns measured as leaking on results/kidney/E005 and results/her2/E005 (F-016).
MEASURED_LEAKS = [
    "protein_colocalization_mob_max",
    "protein_colocalization_mob",
    "protein_colocalization_mob_n_partners",
    "fraction_detected_mob",
    "log_mean_intensity_mob",
    "spatial_morans_i_mob",
    "intensity_cv_mob",
    # *_resid CCS variants: detrending does NOT make them safe, AUC 0.76-0.86.
    "im2deep_ccs_rank_resid",
    "im2deep_abs_delta_ccs_pct_resid",
    "im2deep_delta_ccs_resid",
]

# Symmetric under mz_shuffle because target and decoy share the observation, so they must
# survive the guard or the ranker is left with nothing to work with.
MUST_SURVIVE = [
    "ppm_error_pct",
    "theo_isotope_kl",
    "isotope_image_colocalization_m1",
    "protein_colocalization_weighted",
    "log_protein_n_features",
    "spatial_gearys_c",
    "mob_2d_concentration",  # intrinsic peak quality: observed only, no prediction
    "mob_k0_spread",
    "mob_peak_snr",
    "mob_mz_spread_ppm",
]


def test_catches_every_measured_leak():
    got = _mz_shuffle_leaking_features(MEASURED_LEAKS + MUST_SURVIVE)
    assert set(MEASURED_LEAKS) <= got, f"missed {sorted(set(MEASURED_LEAKS) - got)}"


def test_keeps_the_symmetric_features():
    got = _mz_shuffle_leaking_features(MEASURED_LEAKS + MUST_SURVIVE)
    assert not (set(MUST_SURVIVE) & got), f"over-excluded {sorted(set(MUST_SURVIVE) & got)}"


def test_is_a_superset_of_the_old_enumerated_list():
    """The by-construction rule must never exclude less than the list it replaced."""
    cols = sorted(_MZ_SHUFFLE_CCS_LEAK_FEATURES) + MUST_SURVIVE
    assert _MZ_SHUFFLE_CCS_LEAK_FEATURES <= _mz_shuffle_leaking_features(cols)


def test_a_newly_added_mobility_feature_is_covered_automatically():
    """The point of the rewrite: no list to update when a feature is added."""
    assert "some_future_coloc_mob" in _mz_shuffle_leaking_features(["some_future_coloc_mob"])
