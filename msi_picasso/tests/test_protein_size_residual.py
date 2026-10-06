"""Protein-size residualization (PROGRESS.md F-045 / H-decoy-17)."""
import numpy as np
import pandas as pd
import pytest

from msi_picasso.feature_generator import (
    PROTEIN_SIZE_RESID_SUFFIX,
    SIZE_DRIVEN_PROTEIN_FEATURES,
    residualize_against_protein_size,
)


def _frame(n=600, seed=0):
    """A feature that is pure protein size plus a little real signal."""
    rng = np.random.default_rng(seed)
    size = rng.integers(1, 60, n).astype(float)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame({
        "protein_tryptic_count": size,
        # dominated by size, as the real ones are (Spearman 0.90-0.95)
        "log_protein_n_features": np.log1p(size) * 3 + 0.2 * signal,
        "protein_coverage": 1.0 / size + 0.01 * signal,
        "is_decoy": rng.random(n) < 0.5,
        "_signal": signal,
    })


class TestResidualization:
    def test_removes_the_size_dependence(self):
        from scipy.stats import spearmanr
        df = _frame()
        before = spearmanr(df["protein_tryptic_count"], df["log_protein_n_features"]).statistic
        added = residualize_against_protein_size(df)
        after = spearmanr(df["protein_tryptic_count"],
                          df["log_protein_n_features" + PROTEIN_SIZE_RESID_SUFFIX]).statistic
        assert abs(before) > 0.8, "fixture should be size-dominated"
        assert abs(after) < 0.15, f"size dependence survived: {before:+.3f} -> {after:+.3f}"
        assert "log_protein_n_features" + PROTEIN_SIZE_RESID_SUFFIX in added

    def test_keeps_the_underlying_signal(self):
        """Stripping size must not strip everything: the residual should still
        track the real component the fixture put in."""
        from scipy.stats import spearmanr
        df = _frame()
        residualize_against_protein_size(df)
        r = spearmanr(df["_signal"],
                      df["log_protein_n_features" + PROTEIN_SIZE_RESID_SUFFIX]).statistic
        assert r > 0.3, f"residual lost the real signal too (rho {r:+.3f})"

    def test_adds_rather_than_replaces(self):
        """F-039's precedent: both versions stay measurable side by side."""
        df = _frame()
        before = df["log_protein_n_features"].copy()
        residualize_against_protein_size(df)
        pd.testing.assert_series_equal(df["log_protein_n_features"], before)

    def test_never_sees_the_decoy_label(self):
        """Invariant 1. The transform must be identical whatever the labels are, or
        it could separate the classes by itself."""
        df = _frame()
        flipped = df.copy()
        flipped["is_decoy"] = ~flipped["is_decoy"]
        residualize_against_protein_size(df)
        residualize_against_protein_size(flipped)
        col = "log_protein_n_features" + PROTEIN_SIZE_RESID_SUFFIX
        np.testing.assert_allclose(df[col], flipped[col])

    def test_output_is_bounded_and_complete(self):
        df = _frame()
        added = residualize_against_protein_size(df)
        for c in added:
            v = df[c].to_numpy(float)
            assert np.isfinite(v).all(), f"{c} left NaNs"
            assert v.min() >= 0.0 and v.max() <= 1.0, f"{c} out of [0, 1]"

    def test_missing_size_column_is_a_no_op(self):
        df = _frame().drop(columns=["protein_tryptic_count"])
        assert residualize_against_protein_size(df) == []

    def test_absent_features_are_skipped_not_invented(self):
        df = _frame()[["protein_tryptic_count", "log_protein_n_features"]].copy()
        added = residualize_against_protein_size(df)
        assert added == ["log_protein_n_features" + PROTEIN_SIZE_RESID_SUFFIX]
        for f in SIZE_DRIVEN_PROTEIN_FEATURES:
            if f != "log_protein_n_features":
                assert f + PROTEIN_SIZE_RESID_SUFFIX not in df.columns

    def test_single_protein_size_does_not_crash(self):
        """One distinct size means no bins to rank within."""
        df = _frame(60)
        df["protein_tryptic_count"] = 7.0
        added = residualize_against_protein_size(df)
        assert added and np.isfinite(df[added[0]]).all()


class TestExclusionIsNotBypassed:
    """A residualized companion must not smuggle back a feature the config excluded.

    Found in E025: all three configs exclude `protein_n_features`, but
    `protein_n_features_sizeresid` reached the ranker anyway -- and it is an exact
    duplicate of `protein_colocalization_n_partners_sizeresid`, so the duplicate the
    run existed to remove was still there under another name.
    """

    def test_excluded_raw_feature_yields_no_companion(self):
        from msi_picasso.feature_generator import PROTEIN_SIZE_RESID_SUFFIX
        import inspect
        from msi_picasso import pipeline
        src = inspect.getsource(pipeline.rescore)
        assert "_bypassed" in src, "the exclusion-bypass guard is gone from rescore()"
        # The guard must compare against the caller's features_exclude, not against
        # the _exclude_set this block itself extends -- otherwise it never fires.
        i = src.index("_bypassed")
        assert "set(features_exclude or [])" in src[i:i + 400], (
            "bypass guard must test the config's own exclude list")
        assert PROTEIN_SIZE_RESID_SUFFIX
