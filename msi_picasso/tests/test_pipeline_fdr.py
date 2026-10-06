"""TDC q-value edge cases (PROGRESS.md F-044)."""
import logging

import numpy as np




class TestTdcTieBreaking:
    """Ties must be broken against the targets, not by row order.

    The candidates frame is every target followed by every decoy, so a stable
    sort on score alone inherits that order. If scoring degenerates and returns
    one constant value, every target then ranks above every decoy and the run
    reports nearly all of them at q = 1/n_targets instead of failing. Measured on
    kidney while ablating the protein-level features: a seed failure returned
    all-zero scores and 2284 of 2798 target peptides "passed" at q <= 0.001.
    """

    @staticmethod
    def _frame(n=100):
        """Targets first, then decoys, as the pipeline builds it."""
        return np.array([False] * n + [True] * n)

    def test_constant_scores_report_nothing(self):
        from msi_picasso.pipeline import _tdc_qvalues
        is_decoy = self._frame()
        q = _tdc_qvalues(np.zeros(len(is_decoy)), is_decoy)
        assert (q[~is_decoy] <= 0.05).sum() == 0, (
            "constant scores must not report identifications")
        assert q.min() > 0.5, f"q floor {q.min():.4f} on a degenerate score vector"

    def test_constant_scores_warn(self, caplog):
        from msi_picasso.pipeline import _tdc_qvalues
        with caplog.at_level(logging.WARNING):
            _tdc_qvalues(np.zeros(20), self._frame(10))
        assert any("failed to train" in r.message or "identical" in r.message
                   for r in caplog.records), "degenerate scoring must warn"

    def test_partial_ties_do_not_favour_targets(self):
        """A tie block mid-ranking is resolved decoys-first, so the block cannot
        manufacture a run of targets ahead of the decoys it ties with."""
        from msi_picasso.pipeline import _tdc_qvalues
        is_decoy = np.array([False, False, True, False, True, True])
        scores = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 0.0])   # four-way tie at 1.0
        q = _tdc_qvalues(scores, is_decoy)
        # Within the tie the two decoys are counted before the two targets, so by
        # the end of the block at least 3 decoys are in hand against 3 targets.
        assert q[1] >= 1.0, f"tied target got q={q[1]:.3f}; ties favoured the target"

    def test_distinct_scores_are_unaffected(self):
        """The ordinary case must not change: no real run has ties (E024 has
        zero on all three datasets)."""
        from msi_picasso.pipeline import _tdc_qvalues
        rng = np.random.default_rng(0)
        is_decoy = self._frame(200)
        scores = np.concatenate([rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 200)])
        assert len(np.unique(scores)) == len(scores)
        q = _tdc_qvalues(scores, is_decoy)
        order = np.argsort(-scores, kind="stable")
        t = np.cumsum(~is_decoy[order]).astype(float)
        d = np.cumsum(is_decoy[order]).astype(float)
        expected = np.minimum.accumulate((((d + 1.0) / np.maximum(t, 1.0))[::-1]))[::-1]
        out = np.empty_like(expected); out[order] = expected
        np.testing.assert_allclose(q, np.clip(out, 0.0, 1.0))
