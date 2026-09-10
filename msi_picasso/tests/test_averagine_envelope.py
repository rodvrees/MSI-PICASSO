"""H-decoy-13: envelope comparison referenced to averagine, not to the candidate.

`theo_isotope_kl` and its siblings predict the isotope pattern from the candidate's
own elemental formula. That makes them read composition, and `substitution` decoys
differ from targets in composition by construction, so those features partly read
the target/decoy label with no spectral evidence involved (PROGRESS.md F-036).

The property that fixes it, and the one asserted here: the averagine-referenced
value depends only on the matched peak, so two candidates on the same peak get the
same value no matter how different their formulas are. Whatever asymmetry
substitution creates in composition therefore cannot reach the feature.
"""

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import compute_theoretical_isotope_features

AVG = ["averagine_envelope_kl", "averagine_envelope_m1_ratio_diff",
       "averagine_envelope_m2_ratio_diff"]
OWN = ["theo_isotope_kl", "theo_m1_ratio_diff", "theo_m2_ratio_diff"]


def _df(compositions, feature_mz=1500.0):
    """Candidates on ONE peak, with deliberately different elemental formulas."""
    rows = []
    for n_c, n_h, n_n, n_o, n_s in compositions:
        rows.append({
            "peptide": f"P{len(rows)}", "n_C": n_c, "n_H": n_h, "n_N": n_n,
            "n_O": n_o, "n_S": n_s,
            # mass is the candidate's own; feature_mz is the peak it matched
            "mass": feature_mz - 1.00728, "mh_mz": feature_mz,
            "feature_mz": feature_mz,
        })
    return pd.DataFrame(rows)


def _envelope(feature_mz=1500.0):
    """An observed 3-peak envelope, roughly peptide-like at this mass."""
    return {feature_mz: np.array([1.0, 0.62, 0.24])}


def test_value_is_identical_for_every_candidate_on_a_peak():
    """The whole point: the candidate's formula must not enter the calculation."""
    df = compute_theoretical_isotope_features(
        # same mass, wildly different formulas -- including 0 vs 3 sulfurs
        _df([(66, 103, 17, 20, 0), (66, 103, 17, 20, 3), (60, 95, 22, 24, 1)]),
        maldi_envelopes=_envelope(),
    )
    for col in AVG:
        assert df[col].nunique() == 1, \
            f"{col} varies with the candidate's formula: {df[col].tolist()}"


def test_the_own_formula_features_do_vary(_=None):
    """Contrast, so the test above is not passing for a trivial reason."""
    df = compute_theoretical_isotope_features(
        _df([(66, 103, 17, 20, 0), (66, 103, 17, 20, 3), (60, 95, 22, 24, 1)]),
        maldi_envelopes=_envelope(),
    )
    assert df["theo_isotope_kl"].nunique() > 1, \
        "theo_isotope_kl should depend on the formula -- that is why it leaks"


def test_sulfur_does_not_move_the_averagine_value():
    """Sulfur is the strongest composition channel (34S is ~4.2% abundant), and
    substitution decoys carry about twice the sulfur of targets."""
    a = compute_theoretical_isotope_features(_df([(66, 103, 17, 20, 0)]),
                                             maldi_envelopes=_envelope())
    b = compute_theoretical_isotope_features(_df([(66, 103, 17, 20, 4)]),
                                             maldi_envelopes=_envelope())
    for col in AVG:
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), equal_nan=True), \
            f"{col} responds to sulfur count"
    assert not np.allclose(a["theo_isotope_kl"], b["theo_isotope_kl"]), \
        "theo_isotope_kl should respond to sulfur -- that is the leak channel"


def test_value_does_depend_on_the_peak():
    """It must still carry information, or it is just a constant."""
    envs = {1200.0: np.array([1.0, 0.50, 0.16]), 1800.0: np.array([1.0, 0.75, 0.33])}
    vals = []
    for mz in (1200.0, 1800.0):
        df = compute_theoretical_isotope_features(
            _df([(66, 103, 17, 20, 0)], feature_mz=mz), maldi_envelopes=envs)
        vals.append(df["averagine_envelope_kl"].iloc[0])
    assert vals[0] != vals[1], "averagine_envelope_kl is constant across peaks"


def test_a_non_peptide_envelope_scores_worse_than_a_peptide_one():
    """What the feature is for: flagging a peak whose isotope pattern is not
    peptide-like at its mass."""
    mz = 1500.0
    peptide_like = {mz: np.array([1.0, 0.62, 0.24])}
    flat = {mz: np.array([1.0, 1.0, 1.0])}          # nothing like a peptide envelope
    comp = [(66, 103, 17, 20, 0)]
    good = compute_theoretical_isotope_features(_df(comp, mz), maldi_envelopes=peptide_like)
    bad = compute_theoretical_isotope_features(_df(comp, mz), maldi_envelopes=flat)
    assert bad["averagine_envelope_kl"].iloc[0] > good["averagine_envelope_kl"].iloc[0]


def test_missing_envelope_is_handled():
    df = compute_theoretical_isotope_features(
        _df([(66, 103, 17, 20, 0)]), maldi_envelopes={9999.0: np.array([1.0, 0.6, 0.2])})
    assert df["averagine_envelope_kl"].iloc[0] == 0.0
    assert np.isnan(df["averagine_envelope_m1_ratio_diff"].iloc[0])


def test_no_envelopes_at_all_is_handled():
    df = compute_theoretical_isotope_features(_df([(66, 103, 17, 20, 0)]),
                                              maldi_envelopes=None)
    for col in AVG:
        assert col in df.columns
