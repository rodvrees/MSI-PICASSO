"""Tests for _find_best_feature_labels and best-feature seeding in LDA/QDA."""
import numpy as np
import pandas as pd
import pytest

from msi_picasso.pipeline import _find_best_feature_labels, _rescore_lda, _tdc_qvalues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_df(n_target: int = 200, n_decoy: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic candidates with one cleanly separating feature.

    ``good_feature`` — targets: high values; decoys: low values.
    ``noise_feature`` — uniform noise, carries no signal.
    ``ppm_error_abs``  — targets < 3 ppm; decoys 5–20 ppm.
    ``n_candidates``   — 1 for all (simplifies seeding logic).
    """
    rng = np.random.default_rng(seed)
    n = n_target + n_decoy
    is_decoy = np.array([False] * n_target + [True] * n_decoy)

    good_feature = np.concatenate([
        rng.uniform(0.8, 1.0, n_target),   # targets: clearly high
        rng.uniform(0.0, 0.2, n_decoy),    # decoys: clearly low
    ])
    noise_feature = rng.uniform(0.0, 1.0, n)

    ppm = np.concatenate([
        rng.uniform(0.0, 3.0, n_target),
        rng.uniform(5.0, 20.0, n_decoy),
    ])
    feature_idx = np.tile(np.arange(n_target), 2)

    return pd.DataFrame({
        "peptide": [f"PEP{i}" for i in range(n)],
        "protein": [f"PROT{i % 10}" for i in range(n)],
        "is_decoy": is_decoy,
        "feature_idx": feature_idx,
        "feature_mz": feature_idx * 10.0 + 500.0,
        "ppm_error_abs": ppm,
        "good_feature": good_feature,
        "noise_feature": noise_feature,
        "n_candidates": np.full(n, 2.0),
    })


def _make_uniform_df(n: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature matrix with all-constant columns — no separation possible."""
    rng = np.random.default_rng(seed)
    is_decoy = rng.choice([True, False], n)
    # Constant columns: every candidate gets the same value per feature
    X = np.tile(np.array([1.0, 0.5, 2.0]), (n, 1))
    return X, is_decoy, ["feat_a", "feat_b", "feat_c"]


# ---------------------------------------------------------------------------
# _find_best_feature_labels
# ---------------------------------------------------------------------------


class TestFindBestFeatureLabels:
    def test_returns_tuple_when_feature_separates(self):
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        labels, best_name, n_passing = result
        assert n_passing > 0
        assert best_name in ("good_feature", "noise_feature")

    def test_best_feature_is_the_separating_one(self):
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        _, best_name, _ = result
        assert best_name == "good_feature"

    def test_labels_are_three_valued(self):
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        labels, _, _ = result
        assert set(labels).issubset({-1, 0, 1})

    def test_decoys_all_negative_one(self):
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        labels, _, _ = result
        assert (labels[is_decoy] == -1).all()

    def test_no_decoy_gets_positive_label(self):
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        labels, _, _ = result
        assert (labels[is_decoy] != 1).all()

    def test_n_passing_matches_label_count(self):
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        labels, _, n_passing = result
        assert int((labels == 1).sum()) == n_passing

    def test_returns_none_when_no_targets_pass(self):
        """Constant features: no ranking direction can separate targets from decoys."""
        X, is_decoy, names = _make_uniform_df()

        # Constant features are skipped before TDC; with nothing left, returns None.
        result = _find_best_feature_labels(X, is_decoy, names, init_fdr=1e-15)

        assert result is None

    def test_constant_feature_with_targets_first_returns_none(self):
        """Regression: all-zero feature with targets before decoys must not pass.

        With a mostly-zero feature (e.g. has_cys when cysteine is rare), the
        ascending=False direction scores every zero-row equally.  A stable sort
        preserves the original DataFrame row order (targets first after
        digest_fasta), assigning artificially low q-values to all targets.
        The fix adds a random tiebreak so row order never determines outcomes.
        Two cases are tested:
          (a) fully constant columns — also caught by the std==0 early-exit,
          (b) mostly-zero with a few 1s scattered equally among targets and
              decoys — no real separation, but stable sort would otherwise
              select the feature via the ascending=False direction.
        """
        n = 200
        # Targets first, then decoys — exactly the order produced by digest_fasta.
        is_decoy = np.array([False] * n + [True] * n)

        # (a) Fully constant — std==0 guard should catch this.
        X_const = np.zeros((2 * n, 3))
        assert _find_best_feature_labels(X_const, is_decoy, ["a", "b", "c"], init_fdr=0.01) is None

        # (b) Mostly-zero but not constant: scatter a few 1s uniformly so both
        # classes have the same proportion — truly non-discriminative.
        rng = np.random.default_rng(99)
        X_sparse = np.zeros((2 * n, 1))
        one_idx = rng.choice(2 * n, size=20, replace=False)
        X_sparse[one_idx, 0] = 1.0
        # With random tiebreak the result should be None or have very few
        # passing targets (well below all-targets count n=200).
        result = _find_best_feature_labels(X_sparse, is_decoy, ["has_cys"], init_fdr=0.01)
        if result is not None:
            _, _, n_pass = result
            assert n_pass < n, (
                f"Non-discriminative mostly-zero feature selected {n_pass}/{n} targets — "
                "stable-sort artifact not fixed"
            )

    def test_returns_none_with_extremely_strict_fdr(self):
        """Even with a separating feature, an astronomically strict FDR yields None."""
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=1e-15)

        assert result is None

    def test_shallow_tree_tier_combines_two_weak_features(self):
        """Neither single-feature nor pairwise-sum/diff can separate an XOR-style
        pattern (targets cluster in two opposite quadrants, decoys at the
        origin): any one linear direction only ever catches one quadrant. A
        depth-3 tree can isolate both quadrants as separate leaves and should
        clear ``min_seed_positives`` where the earlier tiers cannot.
        """
        rng = np.random.default_rng(1)
        n_a, n_b, n_dec = 150, 150, 300
        f1 = np.concatenate([
            rng.normal(5, 0.3, n_a), rng.normal(-5, 0.3, n_b), rng.normal(0, 1, n_dec),
        ])
        f2 = np.concatenate([
            rng.normal(5, 0.3, n_a), rng.normal(-5, 0.3, n_b), rng.normal(0, 1, n_dec),
        ])
        is_decoy = np.array([False] * (n_a + n_b) + [True] * n_dec)
        X = np.column_stack([f1, f2])

        result = _find_best_feature_labels(
            X, is_decoy, ["f1", "f2"], init_fdr=0.05, min_seed_positives=200,
        )

        assert result is not None
        _, best_name, n_passing = result
        assert best_name.startswith("tree(")
        assert n_passing > 150  # beats what any single feature/pair tier reaches

    def test_handles_nan_in_features(self):
        """NaN in a feature column must not crash; column median used for ranking."""
        df = _make_clean_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        X = df[["good_feature", "noise_feature"]].values.copy()
        X[::5, 0] = np.nan  # inject NaN every 5th row into good_feature

        result = _find_best_feature_labels(X, is_decoy, ["good_feature", "noise_feature"], init_fdr=0.05)

        assert result is not None
        labels, _, n_passing = result
        assert n_passing > 0
        assert len(labels) == len(df)


# ---------------------------------------------------------------------------
# H-fdr-11 / F-051: protein-level rollup before TDC counting
# ---------------------------------------------------------------------------


def _make_protein_block_df(n_proteins: int = 5, rows_per_protein: int = 8, n_decoys: int = 20):
    """F-051's mechanism in miniature: ``n_proteins`` proteins each carry
    ``rows_per_protein`` rows of one cleanly-separating feature (values spaced
    far apart, so ranking by it never interleaves two proteins' rows), ranked
    above ``n_decoys`` singleton (one-row-per-protein) decoys with distinctly
    lower values.

    Deterministic (no RNG). With every block row outranking every decoy, the
    block reaches its best possible q at its very last (lowest-ranked) row --
    q = (1 + 0 decoys) / (n_proteins * rows_per_protein) -- so row-level TDC
    counts every block row as passing. Rolled up to one row per protein, the
    best q a block can reach is (1 + 0) / n_proteins -- worse by exactly
    ``rows_per_protein``, because that is the whole confound: a protein's row
    count is buying q-value the same way real independent targets would.

    Row-level TDC sees ``n_proteins * rows_per_protein`` passing targets.
    Protein-level TDC sees ``n_proteins``: each protein is one vote.
    """
    block_protein, block_values = [], []
    for k in range(n_proteins):
        top = 1000.0 - 10.0 * k
        block_protein.extend([f"BIG{k}"] * rows_per_protein)
        block_values.extend([top - r for r in range(rows_per_protein)])
    block_protein = np.array(block_protein)
    block_values = np.array(block_values)
    block_is_decoy = np.zeros(len(block_values), dtype=bool)

    # Distinctly below every block value, and each its own protein, so the
    # rollup leaves this side of the population untouched.
    decoy_protein = np.array([f"DECOY_D{i}" for i in range(n_decoys)])
    decoy_values = np.arange(n_decoys, dtype=float)

    protein = np.concatenate([block_protein, decoy_protein])
    feature = np.concatenate([block_values, decoy_values])
    is_decoy = np.concatenate([block_is_decoy, np.ones(n_decoys, dtype=bool)])
    X = feature.reshape(-1, 1)
    return X, is_decoy, protein, n_proteins, n_proteins * rows_per_protein


class TestProteinRollup:
    def test_row_level_counts_every_row_in_the_block(self):
        """Baseline (no protein arg): unchanged row-level behaviour.

        All ``n_proteins * rows_per_protein`` block rows outrank every decoy
        (the singleton region is designed to never pass on its own), so TDC
        over rows counts all 40 of them, exactly as F-051 measured happening
        for real on kidney E031 (a 255-target seed traced back to 4 proteins).
        """
        X, is_decoy, _protein, _n_proteins, n_block_rows = _make_protein_block_df()
        result = _find_best_feature_labels(
            X, is_decoy, ["block_feature"], init_fdr=0.2, min_seed_positives=1,
        )
        assert result is not None
        _, _, n_passing = result
        assert n_passing == n_block_rows == 40

    def test_protein_rollup_counts_one_vote_per_protein(self):
        """With protein given, each of the 5 proteins is one unit of evidence.

        Rolled up to one (best-scoring) row per protein, the 5 block proteins
        rank 1st-5th with no decoy ahead of the 5th (q = (1+0)/5 = 0.2, exactly
        at ``init_fdr``), so all 5 pass and none of the 35 redundant rows do.
        """
        X, is_decoy, protein, n_proteins, _n_block_rows = _make_protein_block_df()
        result = _find_best_feature_labels(
            X, is_decoy, ["block_feature"], init_fdr=0.2, min_seed_positives=1,
            protein=protein,
        )
        assert result is not None
        labels, _, n_passing = result
        assert n_passing == n_proteins == 5
        assert int((labels == 1).sum()) == n_proteins

        # Exactly one representative row per passing protein is positive; its
        # other rows are unlabeled (0), not silently dropped or all positive --
        # broadcasting to the whole block would hand the classifier back the
        # exact multiplicity this rollup exists to remove.
        for k in range(n_proteins):
            block_labels = labels[protein == f"BIG{k}"]
            assert (block_labels == 1).sum() == 1
            assert (block_labels == 0).sum() == len(block_labels) - 1

    def test_protein_rollup_threads_through_include_mask(self):
        """protein must be sliced/expanded consistently with include_mask (H-fdr-2)."""
        X, is_decoy, protein, n_proteins, _n_block_rows = _make_protein_block_df()
        include_mask = np.ones(len(is_decoy), dtype=bool)
        include_mask[-1] = False  # drop one decoy row

        result = _find_best_feature_labels(
            X, is_decoy, ["block_feature"], init_fdr=0.2, min_seed_positives=1,
            protein=protein, include_mask=include_mask,
        )
        assert result is not None
        labels, _, n_passing = result
        assert n_passing == n_proteins
        assert len(labels) == len(is_decoy)
        assert labels[-1] == 0  # excluded row always label 0


# ---------------------------------------------------------------------------
# Fallback to ppm-based seeding when best-feature init returns None
# ---------------------------------------------------------------------------


class TestPpmFallback:
    def test_fallback_produces_valid_scores(self):
        """With train_fdr=1e-15 the best-feature init returns None; ppm fallback must run."""
        df = _make_clean_df(n_target=200, n_decoy=200)
        feat = ["good_feature", "noise_feature", "ppm_error_abs"]

        scores, importances, feat_names, _, _ = _rescore_lda(
            df,
            feat,
            init_ppm_threshold=4.0,
            train_fdr=1e-15,
            max_iter=1,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(df)
        assert np.isfinite(scores).all()

    def test_fallback_targets_score_higher_on_average(self):
        """After ppm fallback, the LDA should still learn some signal."""
        df = _make_clean_df(n_target=200, n_decoy=200)
        feat = ["good_feature", "noise_feature", "ppm_error_abs"]

        scores, _, _, _, _ = _rescore_lda(
            df,
            feat,
            init_ppm_threshold=4.0,
            train_fdr=1e-15,
            max_iter=2,
        )

        is_decoy = df["is_decoy"].values
        assert scores[~is_decoy].mean() > scores[is_decoy].mean()


# ---------------------------------------------------------------------------
# Best-feature seeding in the LDA when a good feature exists
# ---------------------------------------------------------------------------


class TestBestFeatureSeedingInLDA:
    def test_scores_are_finite(self):
        df = _make_clean_df(n_target=200, n_decoy=200)
        feat = ["good_feature", "noise_feature", "ppm_error_abs"]

        scores, _, _, _, _ = _rescore_lda(df, feat, init_ppm_threshold=4.0, train_fdr=0.05, max_iter=3)

        assert np.isfinite(scores).all()

    def test_targets_score_higher_on_average(self):
        df = _make_clean_df(n_target=200, n_decoy=200)
        feat = ["good_feature", "noise_feature", "ppm_error_abs"]

        scores, _, _, _, _ = _rescore_lda(df, feat, init_ppm_threshold=4.0, train_fdr=0.05, max_iter=3)

        is_decoy = df["is_decoy"].values
        assert scores[~is_decoy].mean() > scores[is_decoy].mean()

    def test_correct_output_shape(self):
        df = _make_clean_df(n_target=200, n_decoy=200)
        feat = ["good_feature", "noise_feature", "ppm_error_abs"]

        scores, importances, feat_names, _, _ = _rescore_lda(
            df, feat, init_ppm_threshold=4.0, train_fdr=0.05, max_iter=2
        )

        assert scores.shape == (len(df),)
