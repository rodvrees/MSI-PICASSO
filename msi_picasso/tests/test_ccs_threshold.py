"""The CCS filter's threshold decision (PROGRESS.md F-050)."""
import numpy as np

from msi_picasso.pipeline import ccs_threshold_pct


def test_multiplier_scales_the_p95():
    cal = np.linspace(0.0, 2.0, 101)  # p95 = 1.9
    tol, p95, msg = ccs_threshold_pct(cal, 2.0, None)
    assert np.isclose(p95, 1.9)
    assert np.isclose(tol, 3.8)
    assert "2.0×" in msg


def test_fixed_pct_overrides_the_multiplier():
    """The reason the knob exists: a threshold that does not move with the peak list.

    Fails if the multiplier is ever allowed to win, which would silently loosen the
    window on a denser list (amyloidosis p95 3.18% at min_regions=2, 4.06% at 1).
    """
    dense = np.linspace(0.0, 4.0, 101)
    sparse = np.linspace(0.0, 2.0, 101)
    assert ccs_threshold_pct(dense, 2.0, None)[0] != ccs_threshold_pct(sparse, 2.0, None)[0]
    assert ccs_threshold_pct(dense, 2.0, 2.0)[0] == ccs_threshold_pct(sparse, 2.0, 2.0)[0] == 2.0


def test_fixed_pct_works_without_a_calibration_set():
    tol, p95, msg = ccs_threshold_pct(np.array([0.5, 0.6]), 2.0, 2.0)
    assert tol == 2.0 and p95 is None
    assert "not measurable" in msg


def test_too_few_calibration_peptides_and_no_fixed_pct_skips_the_filter():
    tol, p95, msg = ccs_threshold_pct(np.array([0.5, 0.6]), 2.0, None)
    assert tol is None and p95 is None
    assert "Skipping CCS filter" in msg
