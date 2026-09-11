"""Tests for debug_viz helpers (FDR-coloured ion-image frames)."""

import numpy as np

from msi_picasso.debug_viz import _fdr_frame_color


class TestFdrFrameColor:
    def test_dark_green_at_or_below_1pct(self):
        assert _fdr_frame_color(0.0) == "#006400"
        assert _fdr_frame_color(0.01) == "#006400"
        assert _fdr_frame_color(0.005) == "#006400"

    def test_light_green_between_1_and_5pct(self):
        assert _fdr_frame_color(0.0101) == "#90EE90"
        assert _fdr_frame_color(0.05) == "#90EE90"

    def test_white_above_5pct(self):
        assert _fdr_frame_color(0.0501) == "white"
        assert _fdr_frame_color(0.5) == "white"

    def test_white_for_nan_or_missing(self):
        assert _fdr_frame_color(float("nan")) == "white"
        assert _fdr_frame_color(None) == "white"


class TestPfmExplanations:
    """`debug_pfm_explanations` used to need `coef_`, so it silently produced
    nothing for the rbf_svm backend that every current config uses. It now falls
    back to shap.KernelExplainer (H-model-2). These check that both estimator
    kinds produce a figure and a summary row, and that the selection is the
    peptide-level reported population (F-029), not the per-feature one.
    """

    @staticmethod
    def _fixture(n=60, seed=0):
        import pandas as pd

        rng = np.random.default_rng(seed)
        X = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        is_decoy = np.array([False] * (n // 2) + [True] * (n // 2))
        X[~is_decoy, 0] += 2.0
        res = pd.DataFrame({
            "peptide": [f"PEP{i}" for i in range(n)],
            "protein": "P1",
            "feature_mz": 800.0 + np.arange(n),
            "is_decoy": is_decoy,
            "is_tdc_winner": True,
            "q_value": np.where(is_decoy, 0.9, 0.5),        # nothing passes here
            "is_peptide_winner": True,
            "peptide_q_value": np.where(is_decoy, 0.9, 0.001),  # 30 targets pass
        })
        return X, res, is_decoy

    def _run(self, tmp_path, estimator):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from msi_picasso.debug_viz import debug_pfm_explanations

        X, res, is_decoy = self._fixture()
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("clf", estimator)])
        pipe.fit(X, (~is_decoy).astype(float))
        debug_pfm_explanations(
            res, X, pipe, ["good_feature", "noise_feature"],
            ion_images=None, feature_mzs=None, spatial_df=None,
            output_dir=str(tmp_path), n_decoys=2, max_targets=3,
            kernel_background=5, kernel_nsamples=64,
        )
        return tmp_path / "pfm_explanations"

    def test_linear_estimator(self, tmp_path):
        from sklearn.svm import LinearSVC

        out = self._run(tmp_path, LinearSVC(dual="auto"))
        assert (out / "summary.tsv").exists()
        assert len(list(out.glob("*.png"))) == 5   # 3 targets (capped) + 2 decoys

    def test_kernel_estimator_without_coef(self, tmp_path):
        from sklearn.svm import SVC

        out = self._run(tmp_path, SVC(kernel="rbf"))
        assert (out / "summary.tsv").exists()
        assert len(list(out.glob("*.png"))) == 5

    def test_selects_the_peptide_level_population(self, tmp_path):
        """Only `peptide_q_value` passes at 1% in the fixture; the per-feature
        `q_value` does not. A run selecting on the wrong one explains nothing."""
        import pandas as pd
        from sklearn.svm import LinearSVC

        out = self._run(tmp_path, LinearSVC(dual="auto"))
        summary = pd.read_csv(out / "summary.tsv", sep="\t")
        assert (~summary["is_decoy"]).sum() == 3
        assert (summary.loc[~summary["is_decoy"], "q_value"] < 0.01).all()
