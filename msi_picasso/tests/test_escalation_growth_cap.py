"""Tests for training-FDR escalation (H-fdr-2) and the self-training growth cap
(H-fdr-5). The Percolator-RESET decoy split these were introduced with was
removed; these two options remain. See ``_find_best_feature_labels_escalating``
and ``_rescore_linear``'s ``train_fdr_escalate``/``pseudo_label_growth_cap``.
"""

import numpy as np
import pandas as pd

from msi_picasso.pipeline import (
    _find_best_feature_labels_escalating,
    _rescore_linear,
)

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
# End-to-end via _rescore_linear: growth cap (H-fdr-5)
# ---------------------------------------------------------------------------


def _make_growth_df(n_per_class: int = 100, seed: int = 11) -> pd.DataFrame:
    """A single clean feature separating targets/decoys, plus n_candidates=1
    everywhere so _rescore_linear's seed step is trivial. Used with an explicit
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
        scores_capped, *_ = _rescore_linear(
            df, feat, pseudo_label_growth_cap=1.0, **kwargs
        )
        scores_max1, *_ = _rescore_linear(
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

        scores_capped, *_ = _rescore_linear(
            df, feat, seed_mask=seed_mask, init_ppm_threshold=5.0, train_fdr=0.5,
            max_iter=5, pseudo_label_growth_cap=1.0,
        )
        scores_uncapped, *_ = _rescore_linear(
            df, feat, seed_mask=seed_mask, init_ppm_threshold=5.0, train_fdr=0.5,
            max_iter=5, pseudo_label_growth_cap=None,
        )
        # Different models were trained (capped stops after iteration 1;
        # uncapped keeps iterating), so scores should not be identical.
        assert not np.allclose(scores_capped, scores_uncapped)


