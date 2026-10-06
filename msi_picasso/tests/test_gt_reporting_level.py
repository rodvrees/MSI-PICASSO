"""Ground-truth recovery must be reported on the population the ID count uses.

F-029 moved ID counting to peptide level and left the ground-truth block in
`cli.py` at feature level, so from E015 on every log printed the two headline
numbers on different footings. It mattered: amyloidosis E018 logged 7/10 GT at 1%
FDR against E016's 8/10, which reads as GT lost while the count rose -- the
signature PROGRESS.md sec.1 calls a red flag for an over-optimistic FDR. At
peptide level the same two runs give 6/10 and 7/10, so GT actually rose with the
count and there was no red flag. See PROGRESS.md F-034.
"""

import re

import numpy as np
import pandas as pd
import pytest


def _recovered(result_df, gt):
    """GT recovery per alpha, over whichever population cli.main would use.

    The block under test is inline in a long `main()`, so this mirrors its column
    choice rather than calling it. That is enough for the two behavioural tests
    below, which exist to document *why* the two populations differ; the actual
    guard against the bug returning is
    `test_cli_reports_the_peptide_level_population_when_available`, which reads
    the source.
    """
    is_target = ~result_df["is_decoy"].astype(bool)
    col, q = (
        ("is_peptide_winner", "peptide_q_value")
        if "is_peptide_winner" in result_df.columns
        else ("is_tdc_winner", "reweighted_q_value")
    )
    reported = result_df[result_df[col] & is_target]
    return {
        a: int(
            reported[reported[q] <= a]
            .drop_duplicates(subset=["peptide"])["peptide"]
            .isin(set(gt))
            .sum()
        )
        for a in (0.01, 0.05, 0.10)
    }


def _frame():
    """A GT peptide that wins its feature but loses the peptide-level rollup.

    This is the whole difference between the two populations: a peptide can be
    the best candidate on some feature while a better-scoring match of the same
    peptide elsewhere carries a q-value above the threshold.
    """
    return pd.DataFrame({
        "peptide":            ["GT1", "GT1", "OTHER", "DEC"],
        "is_decoy":           [False, False, False, True],
        "is_tdc_winner":      [True, True, True, True],
        "reweighted_q_value": [0.005, 0.05, 0.005, 0.005],
        # peptide-level rollup keeps one row per peptide, and its q-value is the
        # one computed over peptides -- here above 1% for GT1
        "is_peptide_winner":  [False, True, True, True],
        "peptide_q_value":    [np.nan, 0.05, 0.005, 0.005],
    })


def test_feature_and_peptide_level_can_disagree():
    df = _frame()
    got = _recovered(df, ["GT1"])
    assert got[0.01] == 0, "peptide-level: GT1's peptide q-value is 0.05, so not at 1%"
    assert got[0.05] == 1

    feature_only = df.drop(columns=["is_peptide_winner", "peptide_q_value"])
    got_feat = _recovered(feature_only, ["GT1"])
    assert got_feat[0.01] == 1, "feature-level counts GT1 at 1% -- the old, inconsistent view"


def test_cli_reports_the_peptide_level_population_when_available():
    """The log line must name the level, so a reader can tell which it is."""
    import inspect

    from msi_picasso import cli

    src = inspect.getsource(cli.main)
    block = src[src.index("if gt_peptides and"):]
    assert "is_peptide_winner" in block, "GT reporting ignores the peptide-level population"
    assert "peptide_q_value" in block, "GT reporting uses the feature-level q-value"
    # and it must say which population it used
    assert re.search(r"same population as the reported\s*(\"\s*\n\s*\")?\s*ID count", block), \
        "the GT log line does not state which population it counted over"


@pytest.mark.parametrize("drop", [True, False])
def test_falls_back_to_feature_level_on_pre_f029_frames(drop):
    """Raw-query results predate the peptide-level columns and must still report."""
    df = _frame()
    if drop:
        df = df.drop(columns=["is_peptide_winner", "peptide_q_value"])
    got = _recovered(df, ["GT1"])
    assert set(got) == {0.01, 0.05, 0.10}
