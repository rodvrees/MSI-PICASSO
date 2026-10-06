"""Tests for debug_viz helpers (FDR-coloured ion-image frames)."""

import numpy as np
import pandas as pd

from msi_picasso.debug_viz import (
    _FIGURES_WRITTEN,
    _co_feature_panels,
    _fdr_frame_color,
    _one_row_per_peptide,
    _reset_figure_dir,
)


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


class TestOneRowPerPeptide:
    """One ion-image figure per peptide, not one per peptide-feature pair (F-029)."""

    @staticmethod
    def _rows():
        return pd.DataFrame({
            "peptide": ["PEPTIDEK", "PEPTIDEK", "PEPTIDEK", "OTHERR", "OTHERR"],
            "is_decoy": [False, False, False, False, False],
            "feature_mz": [1000.1, 1000.2, 1000.3, 1200.1, 1200.2],
            "reweighted_q_value": [0.30, 0.01, 0.50, 0.40, 0.02],
        })

    def test_keeps_the_best_q_value_match(self):
        out = _one_row_per_peptide(self._rows())
        assert len(out) == 2
        best = dict(zip(out["peptide"], out["feature_mz"]))
        assert best["PEPTIDEK"] == 1000.2
        assert best["OTHERR"] == 1200.2

    def test_two_peptides_of_one_protein_give_one_row(self):
        """Every figure shows the whole protein, so the protein is the unit."""
        rows = self._rows()
        rows["protein"] = ["P1", "P1", "P1", "P1", "P1"]
        out = _one_row_per_peptide(rows)
        assert len(out) == 1
        assert out.loc[0, "feature_mz"] == 1000.2
        rows["protein"] = ["P1", "P1", "P1", "P2", "P2"]
        assert len(_one_row_per_peptide(rows)) == 2

    def test_targets_and_decoys_stay_separate(self):
        rows = self._rows()
        rows.loc[2, "is_decoy"] = True
        out = _one_row_per_peptide(rows)
        assert len(out) == 3
        assert out[out["is_decoy"]]["feature_mz"].tolist() == [1000.3]

    def test_nan_q_values_lose_to_real_ones(self):
        rows = self._rows()
        rows.loc[1, "reweighted_q_value"] = np.nan
        out = _one_row_per_peptide(rows)
        assert out.set_index("peptide").loc["PEPTIDEK", "feature_mz"] == 1000.1

    def test_falls_back_to_q_value_then_to_first_row(self):
        rows = self._rows().drop(columns=["reweighted_q_value"])
        rows["q_value"] = [0.3, 0.4, 0.05, 0.4, 0.02]
        assert _one_row_per_peptide(rows).set_index("peptide").loc[
            "PEPTIDEK", "feature_mz"] == 1000.3
        bare = self._rows().drop(columns=["reweighted_q_value"])
        out = _one_row_per_peptide(bare)
        assert len(out) == 2
        assert out.set_index("peptide").loc["PEPTIDEK", "feature_mz"] == 1000.1

    def test_empty_and_peptideless_input_pass_through(self):
        assert len(_one_row_per_peptide(pd.DataFrame())) == 0
        no_pep = pd.DataFrame({"feature_mz": [1.0, 2.0]})
        assert len(_one_row_per_peptide(no_pep)) == 2


class TestResetFigureDir:
    def test_removes_stale_pngs_and_keeps_other_files(self, tmp_path):
        d = tmp_path / "ion_images"
        d.mkdir()
        (d / "T_001_P11087.png").write_bytes(b"old")
        (d / "T_007_P11087.png").write_bytes(b"older, different rank")
        (d / "notes.txt").write_text("keep me")
        _reset_figure_dir(str(d))
        assert sorted(p.name for p in d.iterdir()) == ["notes.txt"]

    def test_creates_the_directory_when_absent(self, tmp_path):
        d = tmp_path / "made"
        _reset_figure_dir(str(d))
        assert d.is_dir()

    def test_clears_the_written_registry(self, tmp_path):
        d = str(tmp_path / "ion_images")
        _FIGURES_WRITTEN[d] = {("T", "P11087")}
        _reset_figure_dir(d)
        assert d not in _FIGURES_WRITTEN


class TestCoFeaturePanels:
    """One co-feature panel per peptide, represented by the best peak it wins."""

    @staticmethod
    def _prot():
        # AAAK matches three near-duplicate peaks, BBBK two, CCCK one.
        return pd.DataFrame({
            "protein": ["P1"] * 6 + ["P2"],
            "peptide": ["AAAK", "AAAK", "AAAK", "BBBK", "BBBK", "CCCK", "OTHER"],
            "feature_mz": [957.53, 957.54, 957.55, 1232.61, 1232.62, 847.44, 500.0],
        })

    def test_one_panel_per_peptide(self):
        """Six peaks, three peptides, precursor excluded -> two panels."""
        out = _co_feature_panels(self._prot(), "P1", 957.53, "AAAK", None, None)
        peps = [p for _, p in out]
        assert sorted(peps) == ["BBBK", "CCCK"]
        assert len(peps) == len(set(peps))

    def test_orders_by_q_value_when_there_is_one(self):
        qvals = {1232.61: 0.01, 847.44: 0.50}
        out = _co_feature_panels(self._prot(), "P1", 957.53, "AAAK", qvals, None)
        assert [p for _, p in out] == ["BBBK", "CCCK"]

    def test_precursor_peptide_is_excluded(self):
        """AAAK's other two peaks must not reappear as co-features."""
        out = _co_feature_panels(self._prot(), "P1", 957.53, "AAAK", None, None)
        assert all(p != "AAAK" for _, p in out)

    def test_prefers_a_peak_the_peptide_wins_over_a_better_q_it_loses(self):
        qvals = {1232.61: 0.01, 1232.62: 0.90}      # the 0.01 peak is better...
        winners = {1232.61: "SOMEONE_ELSE", 1232.62: "BBBK"}  # ...but BBBK loses it
        out = dict((p, m) for m, p in _co_feature_panels(
            self._prot(), "P1", 957.53, "AAAK", qvals, winners))
        assert out["BBBK"] == 1232.62

    def test_falls_back_to_best_q_when_the_peptide_wins_nothing(self):
        qvals = {1232.61: 0.90, 1232.62: 0.01}
        winners = {1232.61: "X", 1232.62: "Y"}
        out = dict((p, m) for m, p in _co_feature_panels(
            self._prot(), "P1", 957.53, "AAAK", qvals, winners))
        assert out["BBBK"] == 1232.62

    def test_other_proteins_are_not_included(self):
        out = _co_feature_panels(self._prot(), "P1", 957.53, "AAAK", None, None)
        assert all(p != "OTHER" for _, p in out)

    def test_unknown_protein_gives_nothing(self):
        assert _co_feature_panels(self._prot(), "NOPE", 1.0, "AAAK", None, None) == []
