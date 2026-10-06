"""`ms1rescore_peptides.tsv` is the delivered identification list.

It must contain targets only, one row per peptide, and the peptide-level q-value
that the reported count uses. It previously did none of those three despite its
name and its log line: it kept every peptide-feature pair, applied the
feature-level q-value, and had no decoy filter at all. amyloidosis E018 shipped
1294 rows covering 308 peptides, 11 of them decoys, where the reported count was
247 target peptides. See PROGRESS.md F-035.
"""

import numpy as np
import pandas as pd

from msi_picasso.cli import _write_results


def _result():
    return pd.DataFrame({
        "peptide":            ["GOOD", "GOOD", "ALSOGOOD", "DECOY", "WEAK"],
        "protein":            ["P1", "P1", "P2", "DECOY_P3", "P4"],
        "feature_idx":        [1, 2, 3, 4, 5],
        "is_decoy":           [False, False, False, True, False],
        "is_tdc_winner":      [True, True, True, True, True],
        "reweighted_q_value": [0.001, 0.002, 0.004, 0.003, 0.5],
        "is_peptide_winner":  [False, True, True, True, True],
        "peptide_q_value":    [np.nan, 0.002, 0.004, 0.003, 0.5],
    })


def _written(tmp_path, df):
    _write_results(df, str(tmp_path))
    return pd.read_csv(tmp_path / "ms1rescore_peptides.tsv", sep="\t")


def test_excludes_decoys(tmp_path):
    out = _written(tmp_path, _result())
    assert "DECOY" not in set(out["peptide"]), "a decoy reached the delivered peptide list"


def test_one_row_per_peptide(tmp_path):
    out = _written(tmp_path, _result())
    assert out["peptide"].is_unique, "the same peptide appears more than once"
    assert set(out["peptide"]) == {"GOOD", "ALSOGOOD"}


def test_keeps_the_best_match_per_peptide(tmp_path):
    """GOOD's representative row carries q=0.002; its q=0.001 row lost the rollup."""
    out = _written(tmp_path, _result())
    row = out[out["peptide"] == "GOOD"].iloc[0]
    assert row["peptide_q_value"] == 0.002
    assert row["feature_idx"] == 2


def test_applies_the_one_percent_threshold(tmp_path):
    out = _written(tmp_path, _result())
    assert "WEAK" not in set(out["peptide"]), "a peptide above 1% FDR was written out"


def test_count_matches_the_reported_id_count(tmp_path):
    """The file's length and the logged ID count must be the same number."""
    df = _result()
    reported = int(
        (df["is_peptide_winner"] & ~df["is_decoy"] & (df["peptide_q_value"] <= 0.01)).sum()
    )
    assert len(_written(tmp_path, df)) == reported


def test_falls_back_to_feature_level_for_raw_query_results(tmp_path):
    """Raw-query results have no peptide-level columns and must still be written,
    still without decoys."""
    df = _result().drop(columns=["is_peptide_winner", "peptide_q_value"])
    out = _written(tmp_path, df)
    assert "DECOY" not in set(out["peptide"])
    assert out["peptide"].is_unique
    assert set(out["peptide"]) == {"GOOD", "ALSOGOOD"}


def test_matches_file_is_unfiltered(tmp_path):
    """The full table keeps everything, including decoys -- that is its purpose."""
    df = _result()
    _write_results(df, str(tmp_path))
    full = pd.read_csv(tmp_path / "ms1rescore_matches.tsv", sep="\t")
    assert len(full) == len(df)
    assert full["is_decoy"].any()
