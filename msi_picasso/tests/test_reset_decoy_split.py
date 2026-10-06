"""Tests for H-fdr-2 (Percolator-RESET decoy splitting + training-FDR
escalation) and H-fdr-5 (self-training growth cap).

Freestone et al. (2025): reusing the same decoys for training and FDR
estimation causes a self-selection bias. The fix split decoys into a training
half and a held-out half; every internal seed/pseudo-label decision uses only
the training half, and the final reported FDR is estimated against the
held-out half. See ``_split_decoys_for_reset``, ``_tdc_qvalues_masked``,
``_find_best_feature_labels``'s ``include_mask``, and ``_rescore_linear``'s
``decoy_split_mask``/``train_fdr_escalate``/``pseudo_label_growth_cap`` in
``pipeline.py``.
"""

import numpy as np
import pandas as pd
import pytest

from msi_picasso.pipeline import (
    _find_best_feature_labels,
    _find_best_feature_labels_escalating,
    _rescore_lda,
    _split_decoys_for_reset,
    _tdc_qvalues,
    _tdc_qvalues_masked,
)


# ---------------------------------------------------------------------------
# _split_decoys_for_reset
# ---------------------------------------------------------------------------


class TestSplitDecoysForReset:
    def test_deterministic(self):
        is_decoy = np.array([False] * 50 + [True] * 50)
        a = _split_decoys_for_reset(is_decoy, seed=0)
        b = _split_decoys_for_reset(is_decoy, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_only_decoy_rows_can_be_true(self):
        is_decoy = np.array([False] * 50 + [True] * 50)
        mask = _split_decoys_for_reset(is_decoy)
        assert not mask[~is_decoy].any()

    def test_roughly_half_the_decoys(self):
        is_decoy = np.array([False] * 100 + [True] * 100)
        mask = _split_decoys_for_reset(is_decoy)
        assert int(mask.sum()) == 50

    def test_too_few_decoys_returns_none(self):
        is_decoy = np.array([False] * 50 + [True] * 3)
        assert _split_decoys_for_reset(is_decoy) is None

    def test_different_seeds_can_differ(self):
        is_decoy = np.array([False] * 50 + [True] * 200)
        a = _split_decoys_for_reset(is_decoy, seed=0)
        b = _split_decoys_for_reset(is_decoy, seed=1)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# _tdc_qvalues_masked
# ---------------------------------------------------------------------------


class TestTdcQvaluesMasked:
    def _synthetic(self, seed=0, n=200):
        rng = np.random.default_rng(seed)
        is_decoy = rng.choice([True, False], n)
        scores = rng.normal(0, 1, n) + np.where(is_decoy, -0.5, 0.5)
        return scores, is_decoy

    def test_none_mask_matches_plain_tdc_qvalues(self):
        scores, is_decoy = self._synthetic()
        np.testing.assert_allclose(
            _tdc_qvalues_masked(scores, is_decoy, estimate_mask=None),
            _tdc_qvalues(scores, is_decoy),
        )

    def test_excluded_rows_are_nan(self):
        scores, is_decoy = self._synthetic()
        mask = np.zeros(len(scores), dtype=bool)
        mask[:100] = True
        out = _tdc_qvalues_masked(scores, is_decoy, estimate_mask=mask)
        assert np.isnan(out[~mask]).all()
        assert np.isfinite(out[mask]).all()

    def test_included_rows_match_manual_subset(self):
        scores, is_decoy = self._synthetic()
        mask = np.zeros(len(scores), dtype=bool)
        mask[:100] = True
        out = _tdc_qvalues_masked(scores, is_decoy, estimate_mask=mask)
        expected = _tdc_qvalues(scores[mask], is_decoy[mask])
        np.testing.assert_allclose(out[mask], expected)


# ---------------------------------------------------------------------------
# _find_best_feature_labels — include_mask (RESET firewall)
# ---------------------------------------------------------------------------


class TestIncludeMask:
    def _held_out_corruption_case(self):
        """100 targets, 100 decoys. Training-half decoys (first 50) score low
        (clean signal); held-out-half decoys (last 50) are deliberately planted
        to score just as high as targets, so INCLUDING them would corrupt the
        seed search (they would look like pseudo-positives / dilute the null).
        include_mask excludes them entirely, so the search must behave exactly
        as if they never existed.
        """
        rng = np.random.default_rng(3)
        n_t = 100
        good = np.concatenate([
            rng.uniform(0.8, 1.0, n_t),
            rng.uniform(0.0, 0.2, 50),   # train-half decoys: clearly low
            rng.uniform(0.85, 1.0, 50),  # held-out decoys: planted high
        ])
        is_decoy = np.array([False] * n_t + [True] * 50 + [True] * 50)
        include_mask = np.array([True] * n_t + [True] * 50 + [False] * 50)
        X = good.reshape(-1, 1)
        return X, is_decoy, include_mask

    def test_matches_manual_subset_call(self):
        X, is_decoy, include_mask = self._held_out_corruption_case()
        result_masked = _find_best_feature_labels(
            X, is_decoy, ["good"], init_fdr=0.1, include_mask=include_mask
        )
        result_sub = _find_best_feature_labels(
            X[include_mask], is_decoy[include_mask], ["good"], init_fdr=0.1
        )
        assert result_masked is not None and result_sub is not None
        labels_masked, name_masked, n_masked = result_masked
        labels_sub, name_sub, n_sub = result_sub
        assert name_masked == name_sub
        assert n_masked == n_sub
        np.testing.assert_array_equal(labels_masked[include_mask], labels_sub)

    def test_excluded_rows_always_label_zero(self):
        X, is_decoy, include_mask = self._held_out_corruption_case()
        result = _find_best_feature_labels(
            X, is_decoy, ["good"], init_fdr=0.1, include_mask=include_mask
        )
        assert result is not None
        labels, _, _ = result
        assert (labels[~include_mask] == 0).all()

    def test_output_length_matches_original_full_length(self):
        X, is_decoy, include_mask = self._held_out_corruption_case()
        result = _find_best_feature_labels(
            X, is_decoy, ["good"], init_fdr=0.1, include_mask=include_mask
        )
        assert result is not None
        labels, _, _ = result
        assert labels.shape == (X.shape[0],)

    def test_planted_held_out_signal_does_not_change_the_answer(self):
        """The whole point: the held-out decoys' planted high scores must not
        change n_passing relative to a world where they were simply absent."""
        X, is_decoy, include_mask = self._held_out_corruption_case()
        result_masked = _find_best_feature_labels(
            X, is_decoy, ["good"], init_fdr=0.1, include_mask=include_mask
        )
        # Reference: as if the held-out decoys were never generated at all.
        X_ref = X[include_mask]
        is_decoy_ref = is_decoy[include_mask]
        result_ref = _find_best_feature_labels(X_ref, is_decoy_ref, ["good"], init_fdr=0.1)
        assert result_masked is not None and result_ref is not None
        assert result_masked[2] == result_ref[2]


# ---------------------------------------------------------------------------
# _find_best_feature_labels_escalating — H-fdr-2 training-FDR escalation
# ---------------------------------------------------------------------------


class TestEscalation:
    def _clean_separation(self, n=100, seed=5):
        rng = np.random.default_rng(seed)
        is_decoy = np.array([False] * n + [True] * n)
        good = np.concatenate([rng.uniform(0.8, 1.0, n), rng.uniform(0.0, 0.2, n)])
        return good.reshape(-1, 1), is_decoy

    def test_noop_when_configured_threshold_already_succeeds(self):
        X, is_decoy = self._clean_separation()
        result, fdr_used = _find_best_feature_labels_escalating(
            X, is_decoy, ["good"], init_fdr=0.1, escalate=True
        )
        assert result is not None
        assert fdr_used == 0.1

    def test_escalates_when_threshold_too_strict(self):
        X, is_decoy = self._clean_separation()
        # 1e-6 is far stricter than the smallest achievable q-value here.
        no_escalate, fdr_no = _find_best_feature_labels_escalating(
            X, is_decoy, ["good"], init_fdr=1e-6, escalate=False
        )
        escalated, fdr_yes = _find_best_feature_labels_escalating(
            X, is_decoy, ["good"], init_fdr=1e-6, escalate=True
        )
        assert no_escalate is None
        assert escalated is not None
        assert fdr_yes > 1e-6

    def test_disabled_by_default_matches_plain_call(self):
        X, is_decoy = self._clean_separation()
        result, fdr_used = _find_best_feature_labels_escalating(
            X, is_decoy, ["good"], init_fdr=1e-6,
        )
        assert result is None
        assert fdr_used == 1e-6


# ---------------------------------------------------------------------------
# End-to-end via _rescore_lda: growth cap (H-fdr-5) and decoy_split (H-fdr-2)
# ---------------------------------------------------------------------------


def _make_growth_df(n_per_class: int = 100, seed: int = 11) -> pd.DataFrame:
    """A single clean feature separating targets/decoys, plus n_candidates=1
    everywhere so _rescore_lda's seed step is trivial. Used with an explicit
    small seed_mask so the FIRST pseudo-label iteration jumps to a much larger
    positive count -- the scenario the growth cap is meant to catch.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    is_decoy = np.array([False] * n_per_class + [True] * n_per_class)
    good = np.concatenate([
        rng.uniform(0.8, 1.0, n_per_class),
        rng.uniform(0.0, 0.2, n_per_class),
    ])
    return pd.DataFrame({
        "peptide": [f"PEP{i}" for i in range(n)],
        "protein": [f"PROT{i}" for i in range(n)],
        "is_decoy": is_decoy,
        "feature_idx": np.arange(n),
        "feature_mz": np.arange(n) * 10.0 + 500.0,
        "good_feature": good,
        "n_candidates": np.ones(n),
    })


class TestGrowthCap:
    def test_cap_triggered_on_iteration_one_matches_max_iter_one(self):
        df = _make_growth_df()
        feat = ["good_feature"]
        # Tiny seed: only the single highest-scoring target.
        seed_mask = np.zeros(len(df), dtype=bool)
        seed_mask[np.argmax(df["good_feature"].where(~df["is_decoy"], -np.inf))] = True
        n_seed = int(seed_mask.sum())
        assert n_seed == 1

        kwargs = dict(
            seed_mask=seed_mask, init_ppm_threshold=5.0, train_fdr=0.5, max_iter=5,
        )
        scores_capped, *_ = _rescore_lda(
            df, feat, pseudo_label_growth_cap=1.0, **kwargs
        )
        scores_max1, *_ = _rescore_lda(
            df, feat, max_iter=1, seed_mask=seed_mask, init_ppm_threshold=5.0, train_fdr=0.5,
        )
        # Iteration 1 sees a seed of 1 and (by construction, a clean separator
        # at train_fdr=0.5) far more than 1x1 pseudo-positives -> the cap fires
        # before any label update is accepted, so the returned model is
        # trained on the seed alone, identical to stopping after one iteration.
        np.testing.assert_allclose(scores_capped, scores_max1)

    def test_uncapped_grows_past_the_capped_run(self):
        df = _make_growth_df()
        feat = ["good_feature"]
        seed_mask = np.zeros(len(df), dtype=bool)
        seed_mask[np.argmax(df["good_feature"].where(~df["is_decoy"], -np.inf))] = True

        scores_capped, *_ = _rescore_lda(
            df, feat, seed_mask=seed_mask, init_ppm_threshold=5.0, train_fdr=0.5,
            max_iter=5, pseudo_label_growth_cap=1.0,
        )
        scores_uncapped, *_ = _rescore_lda(
            df, feat, seed_mask=seed_mask, init_ppm_threshold=5.0, train_fdr=0.5,
            max_iter=5, pseudo_label_growth_cap=None,
        )
        # Different models were trained (capped stops after iteration 1;
        # uncapped keeps iterating), so scores should not be identical.
        assert not np.allclose(scores_capped, scores_uncapped)


class TestDecoySplitEndToEnd:
    def test_decoy_split_runs_without_error_and_scores_all_rows(self):
        df = _make_growth_df(n_per_class=60)
        feat = ["good_feature"]
        scores, *_ = _rescore_lda(
            df, feat, init_ppm_threshold=5.0, init_fdr=0.1, train_fdr=0.1,
            decoy_split_mask=_split_decoys_for_reset(df["is_decoy"].values),
        )
        assert scores.shape == (len(df),)
        assert np.isfinite(scores).all()

    def test_decoy_split_none_matches_unsplit_default(self):
        """decoy_split_mask=None must reproduce plain behaviour exactly --
        the default, backward-compatible path."""
        df = _make_growth_df(n_per_class=60)
        feat = ["good_feature"]
        a, *_ = _rescore_lda(df, feat, init_ppm_threshold=5.0, init_fdr=0.1, train_fdr=0.1)
        b, *_ = _rescore_lda(
            df, feat, init_ppm_threshold=5.0, init_fdr=0.1, train_fdr=0.1,
            decoy_split_mask=None,
        )
        np.testing.assert_allclose(a, b)
