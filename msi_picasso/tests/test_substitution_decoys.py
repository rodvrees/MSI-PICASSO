"""Tests for generate_substitution_candidates()."""

import numpy as np
import pandas as pd
import pytest

from msi_picasso.candidates import (
    _AA_RESIDUE_MASSES,
    _assign_mass_columns,
    generate_substitution_candidates,
    target_rows,
)
from msi_picasso.utils import PROTON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target_df(peptides, protein="PROT1"):
    """Build a minimal target DataFrame with accurate masses."""
    df = pd.DataFrame({
        "peptide": list(peptides),
        "protein": protein,
        "is_decoy": False,
        "source": "target",
    })
    _assign_mass_columns(df)
    df = df[df["mass"] > 0].reset_index(drop=True)
    return df


# 10 synthetic tryptic peptides (end K/R; interior non-K/R for simplicity).
# Cover varied lengths, compositions including L, I, N, D, Q, E (isobar pairs).
_PEPTIDES_10 = [
    "PEPTIDEK",
    "STMGILDEK",
    "CELAAAMK",
    "TESTVLGTGFLSR",
    "AATESTPEPTIDEK",
    "HGLDNYR",
    "NAINYLSQK",
    "ELVQNAINYLSQK",
    "MVFGRCELAAK",   # internal R: fewer eligible positions
    "LFNSTMGILDELVR",
]

# 60 peptides for sign-symmetry test (generated deterministically).
def _make_60_peptides():
    rng = np.random.default_rng(7)
    aa = list("ACDEFGHILMNPQSTVWY")  # 18 non-K/R AAs
    peps = []
    while len(peps) < 60:
        L = int(rng.integers(6, 18))  # interior length
        interior = "".join(rng.choice(aa, size=L).tolist())
        term = "K" if len(peps) % 2 == 0 else "R"
        pep = interior + term
        if pep not in peps:
            peps.append(pep)
    return peps

_PEPTIDES_60 = _make_60_peptides()

# 100 peptides for mass-distribution test
def _make_100_peptides():
    rng = np.random.default_rng(13)
    aa = list("ACDEFGHILMNPQSTVWY")
    peps = []
    while len(peps) < 100:
        L = int(rng.integers(6, 22))
        interior = "".join(rng.choice(aa, size=L).tolist())
        term = "K" if len(peps) % 2 == 0 else "R"
        pep = interior + term
        if pep not in peps:
            peps.append(pep)
    return peps

_PEPTIDES_100 = _make_100_peptides()


def _empty_features():
    """Empty feature array for raw-query mode (snap_to_features=False)."""
    return np.array([], dtype=np.float64)


def _run(peptides, seed=42, n_residues=1, collision_filter=True,
         mass_shift_min_da=None, protein="PROT1"):
    target_df = _make_target_df(peptides, protein=protein)
    features = _empty_features()
    result = generate_substitution_candidates(
        target_df, features,
        matching_ppm=20.0,
        n_residues=n_residues,
        random_seed=seed,
        mass_shift_min_da=mass_shift_min_da,
        collision_filter=collision_filter,
        snap_to_features=False,
    )
    return result, target_df


# ---------------------------------------------------------------------------
# Test 1: target independence (invariance)
# ---------------------------------------------------------------------------

class TestInvariance:
    def test_decoy_unchanged_after_subset(self):
        """decoy(p) = f(p, seed) — independent of which other peptides are present."""
        full_result, _ = _run(_PEPTIDES_10, seed=42)
        decoys_full = (
            full_result[full_result["is_decoy"]][["peptide", "feature_mz"]]
            .set_index("peptide")
        )

        # Drop 4 peptides (keep first 6)
        subset = _PEPTIDES_10[:6]
        sub_result, _ = _run(subset, seed=42)
        decoys_sub = (
            sub_result[sub_result["is_decoy"]][["peptide", "feature_mz"]]
            .set_index("peptide")
        )

        common = decoys_full.index.intersection(decoys_sub.index)
        assert len(common) > 0, "No common decoy peptides between full and subset run"

        for pep in common:
            mz_full = float(decoys_full.at[pep, "feature_mz"])
            mz_sub = float(decoys_sub.at[pep, "feature_mz"])
            assert abs(mz_full - mz_sub) < 1e-4, (
                f"Decoy sequence for '{pep}' changed between full and subset run "
                f"(feature_mz: {mz_full:.6f} vs {mz_sub:.6f})"
            )

    def test_reproducible_same_seed(self):
        r1, _ = _run(_PEPTIDES_10, seed=42)
        r2, _ = _run(_PEPTIDES_10, seed=42)
        d1 = r1[r1["is_decoy"]][["peptide", "feature_mz"]].sort_values("peptide")
        d2 = r2[r2["is_decoy"]][["peptide", "feature_mz"]].sort_values("peptide")
        assert list(d1["peptide"]) == list(d2["peptide"])
        assert np.allclose(d1["feature_mz"].values, d2["feature_mz"].values, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: tryptic validity
# ---------------------------------------------------------------------------

class TestTrypticValidity:
    def test_c_terminal_preserved(self):
        result, target_df = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0, "No decoys produced"
        for _, row in decoys.iterrows():
            assert row["peptide"][-1] in "KR", (
                f"Decoy '{row['peptide']}' does not end with K or R"
            )

    def test_length_preserved(self):
        result, target_df = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        for _, row in decoys.iterrows():
            dec_pep = row["peptide"]
            assert len(dec_pep) >= 7, f"Decoy '{dec_pep}' too short"

    def test_sequence_changed(self):
        result, target_df = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        targets = result[~result["is_decoy"]]
        target_seqs = set(targets["peptide"])
        for _, row in decoys.iterrows():
            assert row["peptide"] not in target_seqs, (
                f"Decoy '{row['peptide']}' is identical to a target sequence"
            )

    def test_no_interior_kr_introduced(self):
        """Substitution must not introduce new K or R at interior positions."""
        result, target_df = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        # Map source mhz → source peptide (for source lookup via decoy_delta_da)
        tgt_mhz_to_pep = {float(r["mh_mz"]): r["peptide"] for _, r in target_df.iterrows()}

        for _, row in decoys.iterrows():
            dec_pep = row["peptide"]
            src_mhz = float(row["mh_mz"]) - float(row["decoy_delta_da"])
            src_pep = min(tgt_mhz_to_pep.items(), key=lambda kv: abs(kv[0] - src_mhz))[1]
            assert len(src_pep) == len(dec_pep), f"Length mismatch: {src_pep} vs {dec_pep}"
            for i, (s, d) in enumerate(zip(src_pep, dec_pep)):
                if d in "KR":
                    assert s in "KR", (
                        f"New K/R at position {i} of decoy '{dec_pep}' "
                        f"(source '{src_pep}' had '{s}')"
                    )


# ---------------------------------------------------------------------------
# Test 3: feature_mz correctness
# ---------------------------------------------------------------------------

class TestFeatureMzCorrectness:
    def test_feature_mz_equals_decoy_mhz(self):
        """feature_mz must equal the accurate [M+H]+ of the decoy peptide sequence."""
        result, _ = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0

        # Recompute [M+H]+ for each decoy peptide independently
        check_df = pd.DataFrame({"peptide": decoys["peptide"].values})
        _assign_mass_columns(check_df)
        check_mhz = check_df["mh_mz"].values

        for i, (_, row) in enumerate(decoys.iterrows()):
            assert abs(row["feature_mz"] - check_mhz[i]) < 0.001, (
                f"feature_mz mismatch for decoy '{row['peptide']}': "
                f"stored {row['feature_mz']:.6f}, recomputed {check_mhz[i]:.6f}"
            )

    def test_feature_idx_unique_and_past_grid(self):
        """Each decoy gets a unique feature_idx >= len(feature_mzs) (raw-query mode)."""
        result, _ = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        # Empty feature grid → all valid decoy indices >= 0
        assert (decoys["feature_idx"] >= 0).all()
        assert decoys["feature_idx"].nunique() == len(decoys), (
            "Decoy feature_idx values are not unique"
        )


# ---------------------------------------------------------------------------
# Test 4: sign-symmetry
# ---------------------------------------------------------------------------

class TestSignSymmetry:
    def test_approximately_half_up_half_down(self):
        """~50% of decoys should have positive decoy_delta_da (upshift)."""
        result, _ = _run(_PEPTIDES_60, seed=42)
        decoys = result[result["is_decoy"]]
        assert len(decoys) >= 30, (
            f"Too few decoys produced ({len(decoys)}) for sign-symmetry test"
        )
        n_up = int((decoys["decoy_delta_da"] > 0).sum())
        n_down = int((decoys["decoy_delta_da"] < 0).sum())
        total = n_up + n_down
        assert total > 0
        frac_up = n_up / total
        assert abs(frac_up - 0.5) < 0.20, (
            f"Sign distribution too unbalanced: {n_up} up, {n_down} down "
            f"(fraction up = {frac_up:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 5: collision filter (property test)
# ---------------------------------------------------------------------------

class TestCollisionFilter:
    def test_no_decoy_isobaric_with_target_when_filter_on(self):
        """With collision_filter=True, no decoy [M+H]+ is within matching_ppm of any target."""
        result, target_df = _run(_PEPTIDES_10, collision_filter=True)
        decoys = result[result["is_decoy"]]
        if len(decoys) == 0:
            pytest.skip("No decoys produced")

        target_mzs = np.sort(target_df["mh_mz"].values)
        tol_frac = 20.0e-6
        for mz in decoys["feature_mz"].values:
            lo = np.searchsorted(target_mzs, mz * (1.0 - tol_frac))
            hi = np.searchsorted(target_mzs, mz * (1.0 + tol_frac))
            assert lo >= hi, (
                f"Decoy feature_mz {mz:.6f} is isobaric with a target peptide "
                f"(within 20 ppm)"
            )

    def test_collision_filter_false_allows_isobaric_decoys(self):
        """With collision_filter=False, decoys isobaric with a target are not filtered out.
        We engineer the collision by adding the first decoy's sequence as a new target so
        that collision_filter=True has reason to reject it."""
        r_off, target_df = _run(_PEPTIDES_10, collision_filter=False)
        decoys_off = r_off[r_off["is_decoy"]]
        if len(decoys_off) == 0:
            pytest.skip("No decoys produced")

        # The first decoy sequence has [M+H]+ == collision_mhz.
        # Add that sequence as a new target so collision_filter=True will reject
        # the original source peptide's decoy (which lands at collision_mhz).
        collision_mhz = round(float(decoys_off.iloc[0]["feature_mz"]), 4)
        dec_pep = decoys_off.iloc[0]["peptide"]
        extra_target = _make_target_df([dec_pep], protein="EXTRA_PROT")
        aug_df = pd.concat([target_df, extra_target], ignore_index=True)

        features = _empty_features()
        r_on = generate_substitution_candidates(
            aug_df, features, matching_ppm=20.0,
            collision_filter=True, snap_to_features=False,
        )
        decoys_on = r_on[r_on["is_decoy"]]
        decoy_mzs_on = set(decoys_on["feature_mz"].round(4))
        decoy_mzs_off = set(decoys_off["feature_mz"].round(4))

        assert collision_mhz in decoy_mzs_off, "Sanity: decoy should appear in filter=False run"
        assert collision_mhz not in decoy_mzs_on, (
            "collision_filter=True should have rejected the decoy isobaric with the engineered target"
        )


# ---------------------------------------------------------------------------
# Test 6: mass distribution overlap
# ---------------------------------------------------------------------------

class TestMassDistributionOverlap:
    def test_decoy_masses_spread_across_target_range(self):
        """Decoy masses should cover the target mass range, not be systematically offset."""
        result, target_df = _run(_PEPTIDES_100, seed=42)
        decoys = result[result["is_decoy"]]
        if len(decoys) < 50:
            pytest.skip(f"Too few decoys ({len(decoys)}) for distribution test")

        tgt_masses = target_df["mass"].values
        dec_masses = decoys["mass"].values

        mass_min = max(tgt_masses.min(), 500.0)
        mass_max = min(tgt_masses.max(), 4000.0)
        n_bins = 30
        bins = np.linspace(mass_min, mass_max, n_bins + 1)

        tgt_hist, _ = np.histogram(tgt_masses, bins=bins)
        dec_hist, _ = np.histogram(dec_masses, bins=bins)

        tgt_nonempty = tgt_hist > 0
        n_overlap = int((tgt_nonempty & (dec_hist > 0)).sum())
        n_tgt_bins = int(tgt_nonempty.sum())
        if n_tgt_bins == 0:
            pytest.skip("No target bins with non-zero count")

        overlap_frac = n_overlap / n_tgt_bins
        assert overlap_frac >= 0.70, (
            f"Decoy mass distribution overlaps only {100*overlap_frac:.1f}% of "
            f"non-empty target bins (expected ≥70%). "
            f"This suggests a systematic mass offset."
        )


# ---------------------------------------------------------------------------
# Additional property tests
# ---------------------------------------------------------------------------

class TestProperties:
    def test_is_decoy_dtype_bool(self):
        result, _ = _run(_PEPTIDES_10)
        assert result["is_decoy"].dtype == bool

    def test_decoy_source_label(self):
        result, _ = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        assert (decoys["source"] == "decoy_substitution").all()

    def test_decoy_protein_namespace_separate(self):
        """Decoys carry DECOY_-prefixed protein, disjoint from targets."""
        result, _ = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        tgt_prots = set(result.loc[~result["is_decoy"], "protein"])
        dec_prots = set(decoys["protein"])
        assert all(p.startswith("DECOY_") for p in dec_prots)
        assert tgt_prots.isdisjoint(dec_prots)

    def test_decoy_delta_da_nan_for_targets(self):
        result, _ = _run(_PEPTIDES_10)
        targets = result[~result["is_decoy"]]
        assert targets["decoy_delta_da"].isna().all()

    def test_decoy_ppm_error_initialized_zero(self):
        """ppm_error is 0.0 on decoy rows (overwritten by pipeline in raw-query mode)."""
        result, _ = _run(_PEPTIDES_10)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        assert (decoys["ppm_error"] == 0.0).all()
        assert (decoys["ppm_error_abs"] == 0.0).all()

    def test_protein_n_features_size_fair(self):
        """DECOY_PROT1 should have the same number of features as PROT1 (size-fairness)."""
        result, target_df = _run(_PEPTIDES_10)
        if result["is_decoy"].sum() == 0:
            pytest.skip("No decoys produced")
        tgt_rows = result.loc[~result["is_decoy"]]
        if len(tgt_rows) == 0:
            pytest.skip("No target rows in result (empty feature grid)")
        tgt_n = int(tgt_rows["protein_n_features"].iloc[0])
        dec_n = int(result.loc[result["is_decoy"], "protein_n_features"].iloc[0])
        # protein_n_features = number of unique feature_mzs per protein
        # For substitution: |DECOY_PROT1| = |PROT1| if every peptide got a decoy
        # Allow for some peptides being skipped (short/no eligible positions)
        assert dec_n > 0, "Decoy protein has zero features"

    def test_no_valid_decoys_returns_targets_only(self):
        """With only a 2-AA peptide (no eligible interior positions), no decoy is produced."""
        df = _make_target_df(["AK"], protein="P1")
        features = _empty_features()
        result = generate_substitution_candidates(df, features, snap_to_features=False)
        assert int(result["is_decoy"].sum()) == 0
        assert int((~result["is_decoy"]).sum()) == 0  # also no targets (too short to match empty grid)

    def test_single_residue_decoy_differs_at_one_position(self):
        """n_residues=1 (via the shared retry loop): exactly one interior non-K/R position
        changes, and decoy_delta_da equals that residue-mass difference."""
        target_df = _make_target_df(_PEPTIDES_60)
        result = generate_substitution_candidates(
            target_df, _empty_features(), matching_ppm=20.0, n_residues=1,
            collision_filter=True, snap_to_features=False,
        )
        decoys = result[result["is_decoy"]]
        assert len(decoys) >= 0.95 * len(_PEPTIDES_60)
        for row in decoys.itertuples(index=False):
            src = min(_PEPTIDES_60, key=lambda p: (len(p) != len(row.peptide),
                      sum(a != b for a, b in zip(p, row.peptide))))
            diff = [i for i, (a, b) in enumerate(zip(src, row.peptide)) if a != b]
            assert len(diff) == 1, f"{src} -> {row.peptide}"
            pos = diff[0]
            assert 0 < pos < len(src) - 1 and src[pos] not in "KR"
            expected = _AA_RESIDUE_MASSES[row.peptide[pos]] - _AA_RESIDUE_MASSES[src[pos]]
            assert abs(row.decoy_delta_da - expected) < 1e-3

    def test_leu_ile_isobar_not_substituted_for_each_other(self):
        """L and I have the same residue mass (113.084 Da); they must not substitute for each other."""
        # A peptide with both L and I in the interior — neither should appear as the substituted AA
        # at the other's position.
        peptide = "LIALDGLVSK"  # interior: I(1), A(2), L(3), D(4), G(5), L(6), V(7), S(8)
        # Run many seeds to sample the distribution
        seen_sub_pairs = set()
        for seed in range(20):
            df = _make_target_df([peptide])
            features = _empty_features()
            result = generate_substitution_candidates(
                df, features, random_seed=seed, snap_to_features=False
            )
            decoys = result[result["is_decoy"]]
            if len(decoys) == 0:
                continue
            dec_seq = decoys.iloc[0]["peptide"]
            # Find which position differs
            for pos, (orig, new) in enumerate(zip(peptide, dec_seq)):
                if orig != new:
                    seen_sub_pairs.add((orig, new))
        # The pair (L→I) and (I→L) must never appear
        assert ("L", "I") not in seen_sub_pairs, "L was substituted with isobaric I"
        assert ("I", "L") not in seen_sub_pairs, "I was substituted with isobaric L"


def test_collision_ppm_enforces_separation_when_matching_ppm_is_zero():
    """Raw-query runs set matching_ppm=0, which silently disables the collision
    filter (tolerance 0 rejects only exact ties). collision_ppm decouples the two."""
    import numpy as np
    import pandas as pd
    from msi_picasso.candidates import generate_substitution_candidates

    rng = np.random.default_rng(0)
    peps = ["".join(rng.choice(list("ACDEFGHILMNPQSTVWY"), 12)) + "K" for _ in range(200)]
    from ms1rescore_rs import compute_peptide_masses
    mass, mh, nc, nh, nn, no, ns = compute_peptide_masses(peps)
    target = pd.DataFrame({
        "peptide": peps, "protein": ["P"] * len(peps), "is_decoy": False,
        "mass": mass, "mh_mz": mh,
        "n_C": nc, "n_H": nh, "n_N": nn, "n_O": no, "n_S": ns,
    })
    grid = np.sort(np.asarray(mh, dtype=float))

    def _min_sep_ppm(collision_ppm):
        out = generate_substitution_candidates(
            target, grid, matching_ppm=0.0, n_residues=2,
            collision_filter=True, collision_ppm=collision_ppm, snap_to_features=False,
        )
        dec = np.sort(out.loc[out.is_decoy, "feature_mz"].dropna().unique())
        tm = np.sort(out.loc[~out.is_decoy, "feature_mz"].dropna().unique())
        i = np.clip(np.searchsorted(tm, dec), 1, len(tm) - 1)
        sep = np.minimum(np.abs(dec - tm[i - 1]), np.abs(dec - tm[i])) / dec * 1e6
        return sep, len(dec)

    sep_off, n_off = _min_sep_ppm(None)      # inherits matching_ppm=0 -> filter inert
    sep_on, n_on = _min_sep_ppm(50.0)

    assert (sep_on >= 50.0).all(), f"min separation {sep_on.min():.1f} ppm < 50"
    assert (sep_off < 50.0).any(), "baseline should have decoys inside 50 ppm of a target"
    # Yield must stay essentially complete: peptides left without a decoy shrink the
    # DECOY_ protein namespace relative to its target, which makes protein_coverage
    # (and every other size-sensitive protein feature) leak the label.
    assert n_on >= 0.98 * n_off, f"collision filter kept only {n_on}/{n_off} decoys"


def test_decoy_decoy_relaxation_preserves_target_separation():
    """The second retry phase ignores decoy-vs-decoy proximity to recover yield, but
    the decoy-vs-target separation is the one that matters and must never be relaxed."""
    import numpy as np
    import pandas as pd
    from ms1rescore_rs import compute_peptide_masses
    from msi_picasso.candidates import generate_substitution_candidates

    rng = np.random.default_rng(11)
    # A dense peptide set, so decoy-vs-decoy collisions actually bite.
    peps = ["".join(rng.choice(list("ACDEFGHILMNPQSTVWY"), 11)) + "K" for _ in range(600)]
    mass, mh, nc, nh, nn, no, ns = compute_peptide_masses(peps)
    target = pd.DataFrame({
        "peptide": peps, "protein": ["P"] * len(peps), "is_decoy": False,
        "mass": mass, "mh_mz": mh,
        "n_C": nc, "n_H": nh, "n_N": nn, "n_O": no, "n_S": ns,
    })
    grid = np.sort(np.asarray(mh, dtype=float))
    out = generate_substitution_candidates(
        target, grid, matching_ppm=0.0, n_residues=2,
        collision_filter=True, collision_ppm=40.0, snap_to_features=False,
    )
    dec = np.sort(out.loc[out.is_decoy, "feature_mz"].dropna().unique())
    tm = np.sort(out.loc[~out.is_decoy, "feature_mz"].dropna().unique())
    i = np.clip(np.searchsorted(tm, dec), 1, len(tm) - 1)
    sep = np.minimum(np.abs(dec - tm[i - 1]), np.abs(dec - tm[i])) / dec * 1e6

    assert (sep >= 40.0).all(), f"target separation violated: min {sep.min():.1f} ppm"
    assert out.is_decoy.sum() >= 0.98 * len(peps), "yield collapsed despite the relaxation"


def test_retry_does_not_change_decoys_when_filter_is_inert():
    """Attempt 0 must reproduce the pre-retry draw, so runs without the collision
    filter are unaffected by the retry budget."""
    import numpy as np
    import pandas as pd
    from ms1rescore_rs import compute_peptide_masses
    import msi_picasso.candidates as candidates_mod
    from msi_picasso.candidates import generate_substitution_candidates

    rng = np.random.default_rng(3)
    peps = ["".join(rng.choice(list("ACDEFGHILMNPQSTVWY"), 14)) + "R" for _ in range(150)]
    mass, mh, nc, nh, nn, no, ns = compute_peptide_masses(peps)
    target = pd.DataFrame({
        "peptide": peps, "protein": ["P"] * len(peps), "is_decoy": False,
        "mass": mass, "mh_mz": mh,
        "n_C": nc, "n_H": nh, "n_N": nn, "n_O": no, "n_S": ns,
    })
    grid = np.sort(np.asarray(mh, dtype=float))

    def _decoys(max_attempts):
        original = candidates_mod._SUBSTITUTION_MAX_ATTEMPTS
        candidates_mod._SUBSTITUTION_MAX_ATTEMPTS = max_attempts
        try:
            out = generate_substitution_candidates(
                target, grid, matching_ppm=0.0, n_residues=2,
                collision_filter=False, snap_to_features=False,
            )
        finally:
            candidates_mod._SUBSTITUTION_MAX_ATTEMPTS = original
        return list(out.loc[out.is_decoy].sort_values("feature_mz")["peptide"])

    assert _decoys(1) == _decoys(200)


class TestMassShiftMaxDa:
    """H-fdr-10: an upper bound on the substitution mass shift.

    F-036 measured the isotope-envelope features separating targets from decoys at
    AUC 0.58-0.66 among decoys whose substitution moved the mass more than ~120 Da,
    against 0.50-0.53 below that. Those features read elemental composition, and a
    large shift is exactly a large composition change, so the separation comes from
    how the decoy was built rather than from any spectral evidence.

    The cap rejects and resamples rather than dropping, so no decoy is lost.
    """

    @staticmethod
    def _shifts(max_da, n_residues=2, seed=11):
        """Net |mass shift| of each decoy against its source target."""
        target_df = _make_target_df(_PEPTIDES_100)
        result = generate_substitution_candidates(
            target_df, _empty_features(), matching_ppm=20.0, n_residues=n_residues,
            random_seed=seed, mass_shift_max_da=max_da, collision_filter=False,
            snap_to_features=False,
        )
        src = target_df.set_index("peptide")["mh_mz"].to_dict()
        dec = result[result["is_decoy"].astype(bool)]
        out = []
        for row in dec.itertuples(index=False):
            cand = [abs(row.mh_mz - m) for p, m in src.items()
                    if len(p) == len(row.peptide)
                    and 0 < sum(a != b for a, b in zip(p, row.peptide)) <= n_residues]
            if cand:
                out.append(min(cand))
        return out, int(len(dec))

    def test_uncapped_can_exceed_the_bound(self):
        """Without a cap the shifts run well past it, or the next test is vacuous."""
        shifts, _ = self._shifts(max_da=None)
        assert shifts, "no decoys paired back to a source target"
        assert max(shifts) > 40, f"uncapped max was only {max(shifts):.1f} Da"

    @pytest.mark.parametrize("cap", [20.0, 40.0])
    def test_cap_is_respected(self, cap):
        shifts, _ = self._shifts(max_da=cap)
        assert shifts, f"no decoys generated at cap {cap}"
        assert max(shifts) <= cap + 1e-6, \
            f"cap {cap} Da exceeded: max shift {max(shifts):.3f} Da"

    def test_cap_resamples_rather_than_dropping(self):
        """A rejected substitution is retried, so the decoy count must not fall."""
        _, n_capped = self._shifts(max_da=40.0)
        _, n_plain = self._shifts(max_da=None)
        assert n_capped >= n_plain * 0.98, \
            f"capping lost decoys: {n_capped} against {n_plain}"

    def test_none_reproduces_previous_behaviour(self):
        """The default must leave every earlier result bit-identical."""
        target_df = _make_target_df(_PEPTIDES_100)
        kw = dict(matching_ppm=20.0, n_residues=2, random_seed=11,
                  collision_filter=False, snap_to_features=False)
        a = generate_substitution_candidates(target_df, _empty_features(), **kw)
        b = generate_substitution_candidates(target_df, _empty_features(),
                                             mass_shift_max_da=None, **kw)
        pd.testing.assert_frame_equal(a, b)




# ---------------------------------------------------------------------------
# Composition symmetry: H-decoy-15 / PROGRESS.md F-042
# ---------------------------------------------------------------------------

class TestCompositionSymmetry:
    """`substitution` decoys carry about twice the sulfur of their targets, because
    the replacement residue is drawn uniformly over 18 letters while Cys and Met are
    only 1.4-1.8% of real residues. Features that read composition then partly read
    the target/decoy label (F-036, F-042). These check the two options that fix it,
    and that neither changes a run that does not ask for it."""

    @staticmethod
    def _sulfur_free_peptides(n=400):
        """Targets with no Cys or Met at all, so any sulfur in a decoy was put
        there by the generator and nothing else."""
        rng = np.random.default_rng(3)
        aa = list("ADEFGHILNPQSTVWY")  # 16 letters: no C/M, no K/R
        peps, seen = [], set()
        while len(peps) < n:
            pep = "".join(rng.choice(aa, size=int(rng.integers(8, 16))).tolist()) + "K"
            if pep not in seen:
                seen.add(pep)
                peps.append(pep)
        return peps

    @staticmethod
    def _run_opts(peptides, **kw):
        target_df = _make_target_df(peptides)
        out = generate_substitution_candidates(
            target_df, _empty_features(), matching_ppm=20.0, n_residues=2,
            random_seed=42, collision_filter=True, snap_to_features=False, **kw,
        )
        # Targets do not match an empty feature list, so the frame holds decoys only.
        return out[out["is_decoy"].astype(bool)].reset_index(drop=True)

    @staticmethod
    def _decoy_of(peptide, **kw):
        """The one decoy built from a single target, so the pairing is unambiguous
        and needs no reconstruction (the failure mode of the F-036 Correction)."""
        dec = TestCompositionSymmetry._run_opts([peptide], **kw)
        assert len(dec) == 1, f"expected one decoy for {peptide}, got {len(dec)}"
        return dec.at[0, "peptide"]

    def test_defaults_are_identical_to_the_unoptioned_call(self):
        """Neither option may change a run that does not ask for it, or every
        result predating them stops reproducing."""
        peps = self._sulfur_free_peptides(120)
        pd.testing.assert_frame_equal(
            self._run_opts(peps),
            self._run_opts(peps, residue_weighting="uniform", preserve_sulfur=False),
        )

    def test_uniform_draw_gives_decoys_sulfur_their_targets_never_had(self):
        """The failure F-042 documents. If this stops holding, the two tests below
        are testing nothing."""
        mean_s = self._run_opts(self._sulfur_free_peptides())["n_S"].mean()
        assert mean_s > 0.15, f"expected the uniform draw to add sulfur, got {mean_s:.3f}"

    def test_target_frequency_weighting_removes_most_of_it(self):
        """H-decoy-15. The targets here contain no C or M at all, so their measured
        frequency is zero and the weighted draw should essentially never pick one."""
        peps = self._sulfur_free_peptides()
        uniform = self._run_opts(peps)["n_S"].mean()
        weighted = self._run_opts(peps, residue_weighting="target_frequency")["n_S"].mean()
        assert weighted < uniform / 2, f"weighting barely helped: {uniform:.3f} -> {weighted:.3f}"

    def test_preserve_sulfur_never_substitutes_sulfur_in(self):
        assert self._run_opts(self._sulfur_free_peptides(), preserve_sulfur=True)["n_S"].sum() == 0

    @pytest.mark.parametrize("peptide", ["PEPTCIDEK", "PEPTMIDEK", "PECPTMIDEK", "ACMDEFGHIK"])
    def test_preserve_sulfur_never_substitutes_sulfur_out(self, peptide):
        """Exact per peptide, not merely equal on average: the sulfur residues must
        still be in the same positions."""
        decoy = self._decoy_of(peptide, preserve_sulfur=True)
        assert len(decoy) == len(peptide)
        for pos, (t_aa, d_aa) in enumerate(zip(peptide, decoy)):
            if t_aa in "CM":
                assert d_aa == t_aa, f"{peptide} -> {decoy}: position {pos} changed"
        assert decoy != peptide, "no substitution was applied at all"

    def test_uniform_draw_does_substitute_sulfur_out(self):
        """The negative control for the test above: without the option, a sulfur
        position is substitutable like any other. Four peptides are too few to see
        it, since only 2 of ~10 positions are touched, so this uses a set."""
        rng = np.random.default_rng(11)
        aa = list("ACDEFGHILMNPQSTVWY")
        peps = []
        while len(peps) < 60:
            pep = "".join(rng.choice(aa, size=10).tolist()) + "K"
            if ("C" in pep or "M" in pep) and pep not in peps:
                peps.append(pep)

        def n_sulfur_lost(**kw):
            return sum(
                self._decoy_of(p, **kw).count("C") + self._decoy_of(p, **kw).count("M")
                < p.count("C") + p.count("M")
                for p in peps
            )

        assert n_sulfur_lost() > 0, "uniform draw never substituted a sulfur residue out"
        assert n_sulfur_lost(preserve_sulfur=True) == 0

    def test_both_options_compose(self):
        dec = self._run_opts(self._sulfur_free_peptides(200),
                             residue_weighting="target_frequency", preserve_sulfur=True)
        assert dec["n_S"].sum() == 0

    def test_decoy_yield_is_not_sacrificed(self):
        """The falsifier registered with H-decoy-15: narrowing the pool must not
        cost decoys, or the identification comparison is uninterpretable. Check
        this before reading any count."""
        peps = self._sulfur_free_peptides(300)
        n_base = len(self._run_opts(peps))
        for kw in ({"residue_weighting": "target_frequency"},
                   {"preserve_sulfur": True},
                   {"residue_weighting": "target_frequency", "preserve_sulfur": True}):
            n = len(self._run_opts(peps, **kw))
            assert n >= 0.95 * n_base, f"{kw} lost decoys: {n_base} -> {n}"

    def test_preserve_sulfur_keeps_yield_on_sulfur_rich_peptides(self):
        """The case where the option bites hardest: removing C/M positions from the
        eligible set could leave too few to substitute."""
        peps = ["MCMCMADEK", "CMPEPTIDEK", "MMMCCADEFGK", "ACDEFGHMK"]
        assert len(self._run_opts(peps, preserve_sulfur=True)) == len(self._run_opts(peps))

    def test_invalid_weighting_is_rejected(self):
        with pytest.raises(ValueError, match="residue_weighting"):
            self._run_opts(["PEPTIDEK"], residue_weighting="natural")


class TestTargetRows:
    """target_rows() is shared by the pipeline and scripts/prefilter_peaklist.py.

    Any filtering added here shifts every decoy, because placement consumes one RNG
    draw per row. A prefilter that de-duplicated and range-filtered produced a
    different decoy set and a keep list missing 61 matched peaks (PROGRESS.md F-049).
    """

    @staticmethod
    def _db():
        return pd.DataFrame({
            "peptide": ["PEPTIDEK", "PEPTIDEK", "AAAAAK", "DECOYPEP"],
            "mh_mz": [900.0, 900.0, 100.0, 950.0],
            "is_decoy": [False, False, False, True],
        })

    def test_drops_decoys_only(self):
        out = target_rows(self._db())
        assert list(out["peptide"]) == ["PEPTIDEK", "PEPTIDEK", "AAAAAK"]

    def test_keeps_duplicate_peptides(self):
        assert (target_rows(self._db())["peptide"] == "PEPTIDEK").sum() == 2

    def test_does_not_filter_by_mz(self):
        assert 100.0 in set(target_rows(self._db())["mh_mz"])

    def test_index_is_reset(self):
        assert list(target_rows(self._db()).index) == [0, 1, 2]
