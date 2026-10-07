"""Shared utilities: isotope distributions, compositions, m/z lookup."""

from functools import lru_cache

import numpy as np
from brainpy import isotopic_variants

NEUTRON = 1.003355
PROTON = 1.007276

@lru_cache(maxsize=None)
def theoretical_isotope_distribution(
    n_C: int,
    n_H: int,
    n_N: int,
    n_O: int,
    n_S: int,
    n_peaks: int = 4,
) -> np.ndarray:
    """
    Compute theoretical isotope distribution using brainpy (Mercury algorithm).

    Returns distribution [M0, M1, M2, ...] truncated to n_peaks, then
    normalized sum-to-1 over the truncated peaks so downstream comparisons
    against sum-to-1 observed envelopes (used by chi² and KL) are unbiased.

    Results are cached by composition tuple — O(unique compositions) calls
    rather than O(n_candidates) when used in a vectorized loop.
    """
    composition = {"C": n_C, "H": n_H, "N": n_N, "O": n_O, "S": n_S}
    # charge only shifts the .mz axis; .intensity is charge-independent, so charge=0 is fine
    peaks = isotopic_variants(composition, npeaks=n_peaks, charge=0)
    intensities = np.array([p.intensity for p in peaks], dtype=float)
    if len(intensities) < n_peaks:
        intensities = np.pad(intensities, (0, n_peaks - len(intensities)))
    intensities = intensities[:n_peaks]
    total = intensities.sum()
    if total < 1e-12:
        return np.zeros(n_peaks, dtype=float)
    intensities /= total
    return intensities


def composition_from_sequence(peptide: str) -> dict[str, int]:
    """Get elemental composition {C, H, N, O, S} for a peptide sequence."""
    from pyteomics.mass import Composition

    comp = Composition(sequence=peptide)
    return {
        "C": comp.get("C", 0),
        "H": comp.get("H", 0),
        "N": comp.get("N", 0),
        "O": comp.get("O", 0),
        "S": comp.get("S", 0),
    }


AVERAGINE_C = 0.04443
AVERAGINE_H = 0.06981
AVERAGINE_N = 0.01221
AVERAGINE_O = 0.01329
AVERAGINE_S = 0.00037


def averagine_composition(mass):
    """Averagine elemental composition {C, H, N, O, S} for a mass or an array of masses.

    Atom counts are rounded half to even (``np.round``, the same rule as ``round``).
    A scalar mass gives ``int`` counts, an array gives int arrays.
    """
    m = np.asarray(mass, dtype=np.float64)
    comp = {
        el: np.round(m * per_da).astype(int)
        for el, per_da in (("C", AVERAGINE_C), ("H", AVERAGINE_H), ("N", AVERAGINE_N),
                           ("O", AVERAGINE_O), ("S", AVERAGINE_S))
    }
    if m.ndim == 0:
        return {el: int(v) for el, v in comp.items()}
    return comp


def values_at_mz(values, mzs, query_mzs) -> np.ndarray:
    """Look up ``values`` (aligned with ``mzs``) at each of ``query_mzs`` by exact m/z.

    Non-finite values count as absent, and a repeated m/z keeps its last finite
    value. Returns a float64 array aligned with ``query_mzs``, NaN where absent.
    """
    import pandas as pd

    values = np.asarray(values, dtype=np.float64)
    mzs = np.asarray(mzs, dtype=np.float64)
    keep = np.isfinite(values) & ~np.isnan(mzs)
    lookup = pd.Series(values[keep], index=mzs[keep])
    lookup = lookup[~lookup.index.duplicated(keep="last")]
    return lookup.reindex(np.asarray(query_mzs, dtype=np.float64)).to_numpy(dtype=np.float64)
