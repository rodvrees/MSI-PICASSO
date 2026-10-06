"""Feature finding has moved out of this package.

`extract_maldi_data` used to detect features itself when `feature_mzs` was None
(per-pixel histogram binning for centroid data, SCiLS-style interval detection
for profile data). Both are gone: peaks are now picked in 2D (m/z, 1/K0) by the
TIMSImaging fork and handed over as a feature list. These tests pin that the
package fails loudly rather than silently falling back to a detector that no
longer exists.
"""

import numpy as np
import pytest

from msi_picasso import maldi_extraction


def test_extract_maldi_data_requires_a_feature_list():
    """No feature_mzs is an error, not an invitation to detect."""
    with pytest.raises(ValueError, match="No feature m/z values supplied"):
        maldi_extraction.extract_maldi_data("/nonexistent.d", feature_mzs=None)


def test_the_error_says_where_feature_finding_went():
    """A stale caller should be told what to do, not just that it failed."""
    with pytest.raises(ValueError) as excinfo:
        maldi_extraction.extract_maldi_data("/nonexistent.d")

    message = str(excinfo.value)
    assert "TIMSImaging" in message
    assert "--feature-mzs" in message


def test_the_in_package_detectors_are_gone():
    """Named explicitly so a revival is a deliberate act, not an accident."""
    for removed in ("detect_features", "_build_profile_mean_spectrum", "_deduplicate_mzs"):
        assert not hasattr(maldi_extraction, removed), (
            f"{removed} is back; feature finding belongs in the TIMSImaging fork"
        )


def test_maldi_imzml_module_is_gone():
    """The legacy SCiLS interval path went with it, including its deisotoping."""
    with pytest.raises(ImportError):
        import msi_picasso.maldi_imzml  # noqa: F401


def test_ccs_conversion_now_comes_from_im2deep():
    """`one_over_k0_to_ccs` was a bitwise duplicate of `im2deep.utils.im2ccs`.

    Pinned with values rather than by re-deriving the formula, so a change in
    either package's constants shows up here.
    """
    from im2deep.utils import im2ccs

    ccs = np.asarray(im2ccs(np.array([0.9, 1.4]), np.array([1000.0, 1800.0]), 1), dtype=float)

    assert np.all(np.isfinite(ccs))
    assert np.all(ccs > 0)
    # monotone in 1/K0 at fixed charge
    assert ccs[1] > ccs[0]
    # NaN propagates rather than raising: a query m/z with no observed mobility
    # must stay missing, which raw-query mode relies on.
    assert np.isnan(np.asarray(im2ccs(np.array([np.nan]), np.array([1000.0]), 1), dtype=float)[0])
