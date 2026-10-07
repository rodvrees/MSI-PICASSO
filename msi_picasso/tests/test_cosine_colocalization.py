"""Tests for median-thresholded cosine colocalization features (PROGRESS.md H-feat-3).

Ovchinnikova et al. (2020, ColocML): threshold each ion image at its own median,
then score the pairwise similarity as cosine of the thresholded images. See
``_median_thresholded_cosine_matrix`` / ``compute_cosine_colocalization_features``
in ``maldi_features.py``.
"""

import numpy as np
import pandas as pd
import pytest

from msi_picasso.maldi_features import (
    _median_thresholded_cosine_matrix,
    compute_cosine_colocalization_features,
    compute_tissue_mask,
)

_COSINE_COLOC_COLS = [
    "protein_colocalization_cosine",
    "protein_colocalization_cosine_max",
    "protein_colocalization_cosine_median",
]


def _images_with_shared_region():
    """6 images on an 8x8 grid, same layout as the region-colocalization tests.

    Protein A's three features all concentrate in the SAME top-right block
    (shared bright region). Protein B's three features each occupy a DIFFERENT
    region, so their above-median pixels barely overlap.
    """
    H = W = 8
    imgs = np.zeros((6, H, W), dtype=np.float32)
    for i in range(3):
        imgs[i, 0:3, 4:7] = 1.0 + 0.05 * i      # A0, A1, A2 — same block
    imgs[3, 5:8, 4:6] = 1.0                       # B0 } three
    imgs[4, 0:2, 6:8] = 1.0                       # B1 } disjoint
    imgs[5, 4:6, 6:8] = 1.0                       # B2 } regions
    mzs = np.array([100.0, 110.0, 120.0, 200.0, 210.0, 220.0])
    proteins = ["A", "A", "A", "B", "B", "B"]
    df = pd.DataFrame({"feature_mz": mzs, "protein": proteins})
    return df, imgs, mzs


class TestMedianThresholdedCosineMatrix:
    def test_identical_images_score_one(self):
        H = W = 6
        imgs = np.zeros((2, H, W), dtype=np.float32)
        imgs[:, 1:4, 1:4] = np.random.RandomState(0).rand(3, 3).astype(np.float32)
        imgs[1] = imgs[0]
        mzs = np.array([100.0, 200.0])
        mask = compute_tissue_mask(imgs, 0.0)
        cos_matrix, valid_mz, mz_to_idx = _median_thresholded_cosine_matrix(imgs, mzs, pixel_mask=mask)
        i, j = mz_to_idx[100.0], mz_to_idx[200.0]
        assert cos_matrix[i, j] == pytest.approx(1.0)

    def test_disjoint_bright_regions_score_zero(self):
        H = W = 8
        imgs = np.zeros((2, H, W), dtype=np.float32)
        imgs[0, :, 0:4] = 1.0   # left half bright
        imgs[1, :, 4:8] = 1.0   # right half bright
        mzs = np.array([100.0, 110.0])
        mask = compute_tissue_mask(imgs, 0.0)
        cos_matrix, valid_mz, mz_to_idx = _median_thresholded_cosine_matrix(imgs, mzs, pixel_mask=mask)
        i, j = mz_to_idx[100.0], mz_to_idx[110.0]
        assert cos_matrix[i, j] == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        # Median thresholding zeroes below-median pixels, so both vectors being
        # compared are non-negative and cosine can never go below 0 (unlike the
        # signed Pearson r region-profile metric).
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        cos_matrix, _, _ = _median_thresholded_cosine_matrix(imgs, mzs, pixel_mask=mask)
        assert (cos_matrix >= -1e-6).all()


class TestCosineColocalizationFeatures:
    def test_adds_three_columns(self):
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_cosine_colocalization_features(df, imgs, mzs, pixel_mask=mask)
        for col in _COSINE_COLOC_COLS:
            assert col in out.columns

    def test_shared_region_scores_higher(self):
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_cosine_colocalization_features(df, imgs, mzs, pixel_mask=mask)
        a = out.loc[out["protein"] == "A", "protein_colocalization_cosine"].mean()
        b = out.loc[out["protein"] == "B", "protein_colocalization_cosine"].mean()
        assert a > b

    def test_decoy_namespace_not_pooled_with_target(self):
        df, imgs, mzs = _images_with_shared_region()
        df["protein"] = ["P", "P", "P", "DECOY_P", "DECOY_P", "DECOY_P"]
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_cosine_colocalization_features(df, imgs, mzs, pixel_mask=mask)
        t = out.loc[out["protein"] == "P", "protein_colocalization_cosine"].mean()
        d = out.loc[out["protein"] == "DECOY_P", "protein_colocalization_cosine"].mean()
        assert t > d

    def test_blind_to_is_decoy_and_deterministic(self):
        # No is_decoy parameter exists anywhere in the call chain (invariant 1);
        # same inputs -> identical output, no randomness involved.
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        a = compute_cosine_colocalization_features(df.copy(), imgs, mzs, pixel_mask=mask)[
            "protein_colocalization_cosine"
        ].to_numpy()
        b = compute_cosine_colocalization_features(df.copy(), imgs, mzs, pixel_mask=mask)[
            "protein_colocalization_cosine"
        ].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_corr_cache_reused_matches_fresh_computation(self):
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        cache = _median_thresholded_cosine_matrix(imgs, mzs, pixel_mask=mask)
        cached = compute_cosine_colocalization_features(df.copy(), imgs, mzs, _corr_cache=cache)
        fresh = compute_cosine_colocalization_features(df.copy(), imgs, mzs, pixel_mask=mask)
        np.testing.assert_allclose(
            cached["protein_colocalization_cosine"].to_numpy(),
            fresh["protein_colocalization_cosine"].to_numpy(),
        )

