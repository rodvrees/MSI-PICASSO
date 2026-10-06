"""theo_isotope_kl_massnorm and siblings (PROGRESS.md H-decoy-7a).

theo_isotope_kl leaks under mz_shuffle (F-020): it builds the theoretical envelope from the
candidate's OWN mass, then compares against the observed envelope at the assigned feature,
which mz_shuffle's derangement deliberately places far away in mass. The massnorm variant
rescales the candidate's composition to the mass IMPLIED BY THE ASSIGNED FEATURE first, so
this test constructs a case where that rescaling should turn a real mass mismatch into a
perfect match.
"""

import numpy as np
import pandas as pd
import pytest

from msi_picasso.maldi_features import compute_theoretical_isotope_features
from msi_picasso.utils import PROTON, theoretical_isotope_distribution


def _make_candidate(n_c, n_h, n_n, n_o, n_s, feature_mz):
    aa_mass = 12.0 * n_c + 1.007825 * n_h + 14.003074 * n_n + 15.994915 * n_o + 31.972071 * n_s
    return {
        "n_C": n_c, "n_H": n_h, "n_N": n_n, "n_O": n_o, "n_S": n_s,
        "mass": aa_mass, "feature_mz": feature_mz,
    }


def test_massnorm_removes_a_pure_mass_scaling_mismatch():
    """A candidate placed on a feature 2x its own mass: theo_isotope_kl_massnorm should see
    a near-perfect match (the observed envelope is exactly what a 2x-scaled composition
    predicts), while the unnormalized theo_isotope_kl should not.
    """
    n_c, n_h, n_n, n_o, n_s = 50, 80, 15, 15, 0
    own_mass = (
        12.0 * n_c + 1.007825 * n_h + 14.003074 * n_n + 15.994915 * n_o + 31.972071 * n_s
    )
    feature_mz = 2.0 * own_mass + PROTON  # implied mass = 2x the candidate's own mass

    df = pd.DataFrame([_make_candidate(n_c, n_h, n_n, n_o, n_s, feature_mz)])

    # "Observed" envelope = the theoretical envelope for the DOUBLED composition, i.e.
    # exactly what a real peptide of the implied mass with this same elemental ratio would
    # produce. This is what theo_isotope_kl_massnorm's rescaling is supposed to reconstruct.
    scaled_dist = theoretical_isotope_distribution(2 * n_c, 2 * n_h, 2 * n_n, 2 * n_o, 2 * n_s, n_peaks=3)
    maldi_envelopes = {feature_mz: list(scaled_dist)}

    out = compute_theoretical_isotope_features(df.copy(), maldi_envelopes=maldi_envelopes)

    assert out["theo_isotope_kl_massnorm"].iloc[0] < 1e-6, (
        "massnorm should reconstruct a near-perfect match after rescaling to the implied mass"
    )
    assert out["theo_isotope_kl"].iloc[0] > 1e-3, (
        "sanity check: the RAW (unnormalized) feature must show a real mismatch here, or "
        "this test does not exercise the leak massnorm is meant to fix"
    )


def test_massnorm_columns_present_and_finite_without_envelopes():
    """No maldi_envelopes -> columns exist, default values, no crash (mirrors theo_isotope_kl's
    existing zero/NaN defaults)."""
    df = pd.DataFrame([_make_candidate(50, 80, 15, 15, 0, 1000.0)])
    out = compute_theoretical_isotope_features(df.copy(), maldi_envelopes=None)
    assert "theo_isotope_kl_massnorm" in out.columns
    assert "theo_m1_ratio_diff_massnorm" in out.columns
    assert "theo_m2_ratio_diff_massnorm" in out.columns
    assert out["theo_isotope_kl_massnorm"].iloc[0] == 0.0
    assert np.isnan(out["theo_m1_ratio_diff_massnorm"].iloc[0])


def test_massnorm_symmetric_when_gap_is_zero():
    """No mass gap (feature_mz implies the candidate's own mass) -> massnorm and raw should
    give the same answer, since the rescaling factor is ~1."""
    n_c, n_h, n_n, n_o, n_s = 50, 80, 15, 15, 0
    own_mass = (
        12.0 * n_c + 1.007825 * n_h + 14.003074 * n_n + 15.994915 * n_o + 31.972071 * n_s
    )
    feature_mz = own_mass + PROTON
    df = pd.DataFrame([_make_candidate(n_c, n_h, n_n, n_o, n_s, feature_mz)])

    obs_dist = theoretical_isotope_distribution(n_c, n_h + 3, n_n, n_o, n_s, n_peaks=3)
    maldi_envelopes = {feature_mz: list(obs_dist)}

    out = compute_theoretical_isotope_features(df.copy(), maldi_envelopes=maldi_envelopes)
    assert out["theo_isotope_kl_massnorm"].iloc[0] == pytest.approx(
        out["theo_isotope_kl"].iloc[0], rel=1e-6
    )
