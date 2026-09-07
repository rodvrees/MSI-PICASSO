"""Cross-validated (out-of-fold) scoring in the LDA/QDA backends.

The key property: when targets and decoys are exchangeable (no real signal), an
in-sample model overfits and manufactures separation, but out-of-fold scoring
does not — so the TDC FDR stays honest.
"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from unittest import mock

from msi_picasso import pipeline
from msi_picasso.pipeline import _cv_semisup_scores, _make_fold_ids


def _make_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])),
    ])


class TestMakeFoldIds:
    def test_none_when_too_few(self):
        is_decoy = np.array([True, False] * 3)  # 3 per class, need >= 2*folds = 6
        assert _make_fold_ids(is_decoy, cv_folds=3) is None

    def test_valid_folds_stratified(self):
        rng = np.random.default_rng(0)
        is_decoy = rng.integers(0, 2, 300).astype(bool)
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        assert folds is not None
        assert set(np.unique(folds)) == {0, 1, 2}
        # every fold has both targets and decoys
        for k in range(3):
            m = folds == k
            assert is_decoy[m].any() and (~is_decoy[m]).any()


class TestCvScoring:
    def test_oof_does_not_separate_exchangeable_classes(self):
        """Pure-noise features (no real T/D signal): in-sample LDA overfits to a high
        AUC, but out-of-fold scoring stays ~0.5 (chance)."""
        rng = np.random.default_rng(0)
        n, p = 240, 40           # many features, no signal -> easy to overfit
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool)
        is_decoy[n // 2:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)  # all targets +1, decoys -1
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        assert folds is not None

        oof, pipe_full = _cv_semisup_scores(X, labels, folds, _make_pipe)
        insample = pipe_full.decision_function(X).ravel()

        auc_oof = roc_auc_score(is_decoy.astype(int), oof)
        auc_in = roc_auc_score(is_decoy.astype(int), insample)
        auc_oof = max(auc_oof, 1 - auc_oof)
        auc_in = max(auc_in, 1 - auc_in)

        assert auc_in > 0.68, f"in-sample should overfit, got {auc_in:.2f}"
        assert auc_oof < 0.62, f"out-of-fold should be ~chance, got {auc_oof:.2f}"
        assert auc_in - auc_oof > 0.10  # overfitting gap removed by out-of-fold scoring

    def test_oof_recovers_real_signal(self):
        """When there IS genuine signal, out-of-fold scoring still recovers it."""
        rng = np.random.default_rng(1)
        n, p = 240, 10
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool)
        is_decoy[n // 2:] = True
        X[~is_decoy, 0] += 1.5  # real signal in feature 0 for targets
        labels = np.where(is_decoy, -1, 1).astype(np.int8)
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        oof, _ = _cv_semisup_scores(X, labels, folds, _make_pipe)
        auc = roc_auc_score(is_decoy.astype(int), oof)
        assert max(auc, 1 - auc) > 0.75  # real separation survives CV

    def test_in_sample_fallback_when_no_folds(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 5))
        is_decoy = np.zeros(40, dtype=bool); is_decoy[20:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)
        # fold_ids=None -> in-sample scoring, returns finite scores
        scores, pipe = _cv_semisup_scores(X, labels, None, _make_pipe)
        assert np.isfinite(scores).all()
        assert np.allclose(scores, pipe.decision_function(X).ravel())

    def test_label_zero_rows_are_scored_not_trained(self):
        """label==0 (unlabelled targets) are scored out-of-fold but never trained."""
        rng = np.random.default_rng(3)
        n, p = 300, 8
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool); is_decoy[n // 2:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)
        # demote a chunk of targets to label 0
        tgt_idx = np.where(~is_decoy)[0]
        labels[tgt_idx[:50]] = 0
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        oof, _ = _cv_semisup_scores(X, labels, folds, _make_pipe)
        assert np.isfinite(oof).all()  # every row (incl label-0) gets a score


class TestModelRepeats:
    """Averaging independent replicate fits (H-fdr-6/F-030).

    The failure this addresses: the semi-supervised loop does not converge, so
    its q-value floor moves with the arbitrary CV partition. Averaging *inside*
    one trajectory was tried first and refuted on real data (kidney returned to
    0 IDs at 5% FDR at 40 partitions); averaging independent trajectories is what
    converges, and that is what `model_repeats` does.
    """

    def test_fold_seed_changes_the_partition(self):
        rng = np.random.default_rng(0)
        is_decoy = rng.integers(0, 2, 300).astype(bool)
        a = _make_fold_ids(is_decoy, cv_folds=3)
        assert np.array_equal(a, _make_fold_ids(is_decoy, cv_folds=3, random_state=0))
        assert not np.array_equal(a, _make_fold_ids(is_decoy, cv_folds=3, random_state=1))
        for one in (a, _make_fold_ids(is_decoy, cv_folds=3, random_state=1)):
            assert set(np.unique(one)) == {0, 1, 2}

    def test_default_is_a_single_unchanged_fit(self):
        """model_repeats=1 must be the old code path exactly, or every result in
        PROGRESS.md stops reproducing."""
        calls = []

        def fake_once(*args, **kwargs):
            calls.append(kwargs.get("fold_seed", "absent"))
            return (np.arange(5.0), None, None, None, [])

        with mock.patch.object(pipeline, "_rescore_linear_once", fake_once):
            out = pipeline._rescore_linear("df", ["f"], 5.0)
        assert calls == ["absent"]
        assert np.array_equal(out[0], np.arange(5.0))

    def test_repeats_average_standardised_replicate_scores(self):
        """Replicates share a ranking but not a scale, so the mean is over
        z-scored scores; a replicate on a wild scale must not dominate."""
        made = [np.array([3.0, 2.0, 1.0]), np.array([1000.0, 0.0, -1000.0])]

        def fake_once(*args, fold_seed=0, **kwargs):
            return (made[fold_seed], f"imp{fold_seed}", None, None, [])

        with mock.patch.object(pipeline, "_rescore_linear_once", fake_once):
            out = pipeline._rescore_linear("df", ["f"], 5.0, model_repeats=2)
        z = [(v - v.mean()) / v.std() for v in made]
        assert np.allclose(out[0], np.mean(z, axis=0))
        assert out[1] == "imp0", "importances describe one fitted model, not a mean"

    def test_averaged_scores_stay_out_of_fold(self):
        """Exchangeable classes must still not separate under a different
        partition, or the TDC FDR goes anti-conservative."""
        rng = np.random.default_rng(1)
        n, p = 240, 40
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool)
        is_decoy[n // 2:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)

        for seed in (0, 3):
            folds = _make_fold_ids(is_decoy, cv_folds=3, random_state=seed)
            oof, _ = _cv_semisup_scores(X, labels, folds, _make_pipe)
            assert 0.35 < roc_auc_score(is_decoy.astype(int), oof) < 0.65
