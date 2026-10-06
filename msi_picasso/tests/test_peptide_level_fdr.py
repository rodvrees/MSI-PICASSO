"""Peptide-level FDR rollup.

Feature-list extraction matches a peptide to every detected feature within
tolerance, so it contributes one row per (peptide, feature) pair -- 4.8 rows per
peptide on amyloidosis, 2.1 on kidney. Computing TDC over those rows counts one
peptide as many discoveries on both sides of the competition. These tests pin
the rollup that fixes it, and the properties that make it safe.
"""

import numpy as np
import pandas as pd
import pytest

from msi_picasso.pipeline import _peptide_level_qvalues, _tdc_qvalues


def test_one_representative_per_peptide_and_class():
    peptides = np.array(["A", "A", "A", "B", "B", "C"])
    is_decoy = np.zeros(6, dtype=bool)
    scores = np.array([1.0, 5.0, 3.0, 2.0, 9.0, 4.0])

    _, is_rep = _peptide_level_qvalues(scores, is_decoy, peptides)

    assert is_rep.sum() == 3
    # the representative is the best-scoring match of each peptide
    assert is_rep[1] and is_rep[4] and is_rep[5]


def test_a_target_and_decoy_sharing_a_sequence_do_not_collapse_together():
    """Grouping keys on the label only to avoid merging across it."""
    peptides = np.array(["A", "A"])
    is_decoy = np.array([False, True])

    _, is_rep = _peptide_level_qvalues(np.array([1.0, 2.0]), is_decoy, peptides)

    assert is_rep.all(), "a target and a decoy must stay separate hypotheses"


def test_rollup_is_blind_to_which_class_is_which():
    """Swapping the labels swaps the outputs and nothing else.

    The rollup necessarily sees `is_decoy` -- TDC is defined on labels -- so the
    property that matters is that it treats the two classes by the same rule.
    """
    peptides = np.array(["A", "A", "B", "B", "C", "C"])
    scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 0.5])
    labels = np.array([False, False, True, True, False, False])

    _, rep_a = _peptide_level_qvalues(scores, labels, peptides)
    _, rep_b = _peptide_level_qvalues(scores, ~labels, peptides)

    # which row represents each peptide cannot depend on the class it was given
    np.testing.assert_array_equal(rep_a, rep_b)


def test_q_is_broadcast_to_every_match_of_a_peptide():
    peptides = np.array(["A", "A", "B"])
    is_decoy = np.array([False, False, True])

    q, is_rep = _peptide_level_qvalues(np.array([3.0, 1.0, 2.0]), is_decoy, peptides)

    assert q[0] == q[1], "both matches of peptide A carry A's peptide-level q"
    assert np.isfinite(q).all()
    assert is_rep.sum() == 2


def test_counts_peptides_not_matches():
    """The defect this exists to fix, stated as a test.

    One strong target peptide replicated across five features must count once,
    not five times.
    """
    n_pep, k = 30, 5
    peptides = np.repeat([f"T{i}" for i in range(n_pep)], k)
    peptides = np.concatenate([peptides, np.repeat([f"D{i}" for i in range(n_pep)], k)])
    is_decoy = np.array([False] * (n_pep * k) + [True] * (n_pep * k))
    # every target match outscores every decoy match; multiplicity is equal
    scores = np.concatenate([np.linspace(10, 5, n_pep * k), np.linspace(4, 0, n_pep * k)])

    row_q = _tdc_qvalues(scores, is_decoy)
    pep_q, is_rep = _peptide_level_qvalues(scores, is_decoy, peptides)

    n_rows = int(((row_q <= 0.05) & ~is_decoy).sum())
    n_peptides = int(((pep_q <= 0.05) & is_rep & ~is_decoy).sum())

    assert n_rows == n_pep * k, "row-level counts each peptide k times"
    assert n_peptides == n_pep, "peptide-level counts each peptide once"


def test_reduces_to_plain_tdc_when_every_peptide_matches_once():
    """With one match per peptide the rollup must change nothing at all.

    This is the guard that the change is inert for raw-query mode, where a
    candidate has exactly one feature by construction, so every result predating
    feature-list extraction is unaffected.
    """
    rng = np.random.default_rng(0)
    n = 200
    peptides = np.array([f"P{i}" for i in range(n)])
    is_decoy = rng.random(n) < 0.5
    scores = rng.normal(size=n) + (~is_decoy) * 0.8

    plain = _tdc_qvalues(scores, is_decoy)
    rolled, is_rep = _peptide_level_qvalues(scores, is_decoy, peptides)

    assert is_rep.all()
    np.testing.assert_allclose(rolled, plain)


def test_asymmetric_multiplicity_is_warned_about(caplog):
    """The rollup takes a maximum over matches, so it is only unbiased while both
    classes have the same number of matches per peptide. Measured symmetric on
    both datasets (KS p=0.26 and 1.0); if that ever stops holding, say so."""
    # targets matched once, decoys matched four times each
    peptides = np.array(["T1", "T2", "T3"] + ["D1"] * 4 + ["D2"] * 4 + ["D3"] * 4)
    is_decoy = np.array([False] * 3 + [True] * 12)
    scores = np.arange(len(peptides), dtype=float)

    with caplog.at_level("WARNING"):
        _peptide_level_qvalues(scores, is_decoy, peptides)

    assert "multiplicity" in caplog.text.lower()


def test_estimate_mask_is_subset_to_the_representatives():
    """Decoy-split (H-fdr-2b) must keep working through the rollup."""
    peptides = np.array(["A", "A", "B", "C"])
    is_decoy = np.array([False, False, True, True])
    scores = np.array([3.0, 1.0, 2.0, 0.5])
    # hold out the second decoy from estimation
    mask = np.array([True, True, True, False])

    q, is_rep = _peptide_level_qvalues(scores, is_decoy, peptides, estimate_mask=mask)

    assert is_rep.sum() == 3
    # a held-out decoy gets NaN by design (it is excluded from the estimate);
    # every target representative must still carry a finite q
    assert np.isfinite(q[is_rep & ~is_decoy]).all()
