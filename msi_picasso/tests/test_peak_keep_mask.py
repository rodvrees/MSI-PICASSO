"""keep_mask: shrink ion images without changing the on-tissue mask (F-048).

Only 18-34% of peaks are ever matched by a candidate, so most ion images are
never read per-candidate. But they are read collectively: `compute_tissue_mask`
sums every image per pixel to decide which pixels are tissue, and F-006 records
that this mask is what stops every pairwise correlation measuring the tissue
outline. Dropping the unmatched peaks from the extraction shifted the whole
colocalization family on ~100% of rows. So the sum must run over all peaks and
only the images may be discarded.
"""
import numpy as np
import pytest

from msi_picasso.maldi_features import _pearson_r_matrix, compute_tissue_mask


class TestTissueMaskWithPrecomputedTIC:
    @staticmethod
    def _images(n=20, h=4, w=5, seed=0):
        rng = np.random.default_rng(seed)
        im = rng.random((n, h, w)).astype(np.float32)
        im[:, :, 0] = 0.0          # a column of unmeasured padding
        return im

    def test_precomputed_tic_reproduces_the_full_sum(self):
        im = self._images()
        tic = im.reshape(len(im), -1).sum(axis=0)
        np.testing.assert_array_equal(
            compute_tissue_mask(im), compute_tissue_mask(im[:5], tic_image=tic)
        )

    def test_this_is_the_bug_it_exists_to_prevent(self):
        """A subset alone gives a different mask -- the failure mode in F-048."""
        im = self._images()
        im[5:, :, 1] = 0.0          # a column only the dropped peaks carry signal in
        full = compute_tissue_mask(im)
        subset_only = compute_tissue_mask(im[5:])
        assert not np.array_equal(full, subset_only), (
            "fixture must distinguish the full mask from the subset's")
        tic = im.reshape(len(im), -1).sum(axis=0)
        np.testing.assert_array_equal(full, compute_tissue_mask(im[5:], tic_image=tic))

    def test_quantile_threshold_uses_the_precomputed_tic(self):
        im = self._images()
        tic = im.reshape(len(im), -1).sum(axis=0)
        np.testing.assert_array_equal(
            compute_tissue_mask(im, tic_quantile=0.3),
            compute_tissue_mask(im[:2], tic_quantile=0.3, tic_image=tic),
        )

    def test_pixel_count_mismatch_is_rejected(self):
        im = self._images()
        with pytest.raises(ValueError, match="pixels"):
            compute_tissue_mask(im, tic_image=np.ones(7))

    def test_no_tic_keeps_the_old_behaviour(self):
        im = self._images()
        m = compute_tissue_mask(im)
        assert m.sum() == im.shape[1] * (im.shape[2] - 1)   # the zero column is dropped


class TestKeepMaskAlignment:
    """keep_mask and drop_zero_signal index different axes and must be composed.

    With keep_mask the isotope and adduct images are extracted only at the kept
    peaks, so their rows are indexed by flatnonzero(keep_mask). The zero-signal
    drop is indexed by the full feature axis. Subsetting the extras with the
    full-length mask misaligns every extra image against its own peak -- silently,
    because the shapes still broadcast whenever the counts happen to agree.
    """

    def test_composing_the_two_masks_keeps_rows_paired(self):
        keep = np.array([True, False, True, True, False, True])
        detected = np.array([True, True, False, True, True, True])
        # Extras hold one row per kept peak, in keep order: peaks 0, 2, 3, 5.
        extra = np.arange(6)[keep]
        survives = detected[np.flatnonzero(keep)]
        assert list(extra[survives]) == [0, 3, 5], "extras must follow keep-order"
        # The main axis is filtered by detected, then by the narrowed keep.
        main = np.arange(6)[detected]
        assert list(main[keep[detected]]) == [0, 3, 5], "main and extras must agree"

    def test_naive_subsetting_would_misalign(self):
        """The bug this guards: using the full-length mask on keep-indexed rows."""
        keep = np.array([True, False, True, True, False, True])
        detected = np.array([True, True, False, True, True, True])
        extra = np.arange(6)[keep]
        with pytest.raises(IndexError):
            _ = extra[detected]          # 4 rows indexed by a 6-long mask


class TestPearsonWholeListTransforms:
    """`_pearson_r_matrix`'s pre-correlation transforms are per-pixel sums ACROSS
    features, so they read every image just as the tissue mask does.

    Measured on kidney E028: with the subset alone, 3 of 31 ranker features
    (`protein_colocalization_weighted`, `_weighted_max`, `_top5`) differed from the
    full run because common-mode removal used the 7875 retained images instead of all
    35665. Passing `full_tic` restores identity (PROGRESS.md F-048).
    """

    @staticmethod
    def _images(seed=0, n=12, h=6, w=5):
        rng = np.random.default_rng(seed)
        img = rng.random((n, h, w)).astype(np.float32) + 0.5
        return img, np.linspace(500.0, 900.0, n)

    def _subset_matches_full(self, **flags):
        img, mzs = self._images()
        keep = np.zeros(len(mzs), bool)
        keep[[1, 4, 5, 9]] = True
        full_tic = img.reshape(len(mzs), -1).sum(axis=0)

        full_m, full_mz, full_idx = _pearson_r_matrix(img, mzs, **flags)
        sub_m, sub_mz, sub_idx = _pearson_r_matrix(
            np.ascontiguousarray(img[keep]), mzs[keep],
            full_tic=full_tic, full_n_features=len(mzs), **flags,
        )
        assert list(sub_mz) == [m for m in mzs[keep]]
        rows = [full_idx[float(m)] for m in sub_mz]
        expect = full_m[np.ix_(rows, rows)]
        np.testing.assert_allclose(sub_m, expect, rtol=0, atol=1e-6)

    def test_common_mode_removal(self):
        self._subset_matches_full(common_mode_removal=True)

    def test_tic_normalize(self):
        self._subset_matches_full(tic_normalize=True)

    def test_both_transforms(self):
        self._subset_matches_full(tic_normalize=True, common_mode_removal=True)

    def test_subset_alone_is_wrong(self):
        """Without full_tic the subset gives different correlations — the bug."""
        img, mzs = self._images()
        keep = np.zeros(len(mzs), bool)
        keep[[1, 4, 5, 9]] = True
        full_m, _, full_idx = _pearson_r_matrix(img, mzs, common_mode_removal=True)
        sub_m, sub_mz, _ = _pearson_r_matrix(
            np.ascontiguousarray(img[keep]), mzs[keep], common_mode_removal=True
        )
        rows = [full_idx[float(m)] for m in sub_mz]
        assert not np.allclose(sub_m, full_m[np.ix_(rows, rows)], rtol=0, atol=1e-6)

    def test_pixel_mask_is_applied_to_full_tic(self):
        img, mzs = self._images()
        keep = np.zeros(len(mzs), bool)
        keep[[0, 3, 7]] = True
        pmask = np.zeros(img.shape[1] * img.shape[2], bool)
        pmask[::2] = True
        full_tic = img.reshape(len(mzs), -1).sum(axis=0)
        full_m, _, full_idx = _pearson_r_matrix(
            img, mzs, pixel_mask=pmask, common_mode_removal=True
        )
        sub_m, sub_mz, _ = _pearson_r_matrix(
            np.ascontiguousarray(img[keep]), mzs[keep], pixel_mask=pmask,
            common_mode_removal=True, full_tic=full_tic, full_n_features=len(mzs),
        )
        rows = [full_idx[float(m)] for m in sub_mz]
        np.testing.assert_allclose(sub_m, full_m[np.ix_(rows, rows)], rtol=0, atol=1e-6)

    def test_wrong_pixel_count_is_rejected(self):
        img, mzs = self._images()
        with pytest.raises(ValueError, match="pixels"):
            _pearson_r_matrix(img, mzs, common_mode_removal=True,
                              full_tic=np.zeros(3), full_n_features=len(mzs))
