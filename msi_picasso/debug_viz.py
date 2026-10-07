"""
Debug visualization for the MALDI-MSI rescoring pipeline.

Fourteen subsystems:
  1. Ion image colocalization  — per-candidate precursor + ALL same-protein co-feature images; each panel framed by ID FDR (dark green ≤1%, light green ≤5%, white otherwise)
  2. Feature diagnostics       — per-candidate 4×3 panel figure (incl. m/z-detrended CCS, ion-image colocalization, theoretical isotope/mass defect)
  3. Isotope envelopes         — per-candidate spectrum-style envelope comparison
  4. Feature importance        — global sorted bar plots (rounds 1 and 2)
  5. Feature distributions     — per-feature target/decoy histograms (all + winners)
  6. CCS scatter               — observed vs predicted CCS for all candidates
  7. IDs vs FDR curve          — target identifications as a function of FDR threshold
  8. Protein colocalization    — colocalization values split by scoring group
  9. T/D m/z distribution      — target vs decoy m/z coverage and competition status
 10. Candidate competition     — target/decoy candidate counts per feature (with CCS-filter note)
 11. Score PP plot             — empirical CDF of decoy scores vs target scores
 12. Score distributions       — target/decoy score histograms, all candidates and winners
 13. Pearson r distribution    — same-protein vs different-protein ion image Pearson r at 5% FDR
 14. Protein spatial coherence — per-protein peptide count vs mean ion image Pearson r at 5% FDR

Entry point: save_debug_figures()
"""

import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout",
    category=UserWarning,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_and_close(fig, path, dpi=120):
    """Save a figure with the standard tight bounding box and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _flag(df: pd.DataFrame, col: str) -> np.ndarray:
    """Boolean column as an array; all False when the column is absent."""
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[col].fillna(False).astype(bool).values


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    """Numeric column as a float array; all NaN when the column is absent."""
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").values.astype(float)


def _pep_top(is_winner: np.ndarray, pep: np.ndarray, k: int = 5) -> np.ndarray:
    """Row indices of the ``k`` winners with the lowest finite PEP."""
    w = np.where(is_winner)[0]
    p = pep[w]
    finite = np.isfinite(p)
    return w[finite][np.argsort(p[finite])[:k]]


def _annotate_subset(feat: pd.DataFrame, res: pd.DataFrame, idx, group) -> pd.DataFrame:
    """Rows ``idx`` of ``feat`` with the result columns, T/D label and round-1 rank.

    ``feat`` and ``res`` must be row-aligned with a fresh RangeIndex.
    """
    idx = np.asarray(idx, dtype=int)
    sub = feat.iloc[idx].copy().reset_index(drop=True)
    sub["_group"] = group
    sub["_td"] = np.where(_flag(sub, "is_decoy"), "D", "T")
    r1_cols = [c for c in res.columns if c.endswith("_score_r1")]
    cols = r1_cols + [c for c in ("q_value", "is_tdc_winner", "score") if c in res.columns]
    for col in cols:
        sub[col] = res[col].values[idx]
    sub["_score_r1"] = sub[r1_cols[0]] if r1_cols else np.nan
    sub["_rank"] = (
        sub["_score_r1"].rank(ascending=False, method="min", na_option="bottom").astype(int)
    )
    sub["_total"] = len(feat)
    return sub


def _sample_subset(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    n: int = 50,
    seed: int = 42,
    fdr_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Stratified sample of n rows, guaranteeing at least one from each non-empty group:
      ID  — TDC winner with q_value <= fdr_threshold
      R1  — TDC winner but does not pass FDR
      L   — not a winner

    Groups are computed exclusively from result_df to avoid conflicts with
    features_df, which may contain its own is_tdc_winner column from the
    generative pre-scoring step.
    """
    rng = np.random.default_rng(seed)
    N = len(features_df)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    is_winner = _flag(res, "is_tdc_winner")
    passes = is_winner & (_num(res, "q_value") <= fdr_threshold)
    groups = np.where(passes, "ID", np.where(is_winner, "R1", "L"))
    td_labels = np.where(_flag(feat, "is_decoy"), "D", "T")

    # If no candidates pass 1% FDR, seed the ID stratum from the 5 winners
    # with the lowest PEP so at least one high-confidence example appears in
    # every per-candidate debug figure.
    if not passes.any() and "pep" in res.columns:
        groups[_pep_top(is_winner, _num(res, "pep"))] = "ID"

    # Stratify across 6 strata: {ID, R1, L} × {T, D}
    strata = [(grp, td) for grp in ("ID", "R1", "L") for td in ("T", "D")]
    per_stratum = max(1, n // len(strata))
    sampled_idx: list[int] = []
    for grp, td in strata:
        stratum_idx = np.where((groups == grp) & (td_labels == td))[0].tolist()
        if stratum_idx:
            k = min(per_stratum, len(stratum_idx))
            sampled_idx.extend(rng.choice(stratum_idx, size=k, replace=False).tolist())

    remaining = n - len(sampled_idx)
    if remaining > 0:
        sampled_set = set(sampled_idx)
        unsampled = [i for i in range(N) if i not in sampled_set]
        if unsampled:
            sampled_idx.extend(
                rng.choice(unsampled, size=min(remaining, len(unsampled)), replace=False).tolist()
            )

    sampled_idx = rng.permutation(sampled_idx).astype(int)
    return _annotate_subset(feat, res, sampled_idx, groups[sampled_idx])


def _find_image_idx(feature_mz: float, ion_image_mzs: np.ndarray, ppm: float = 25.0) -> int | None:
    if ion_image_mzs is None or len(ion_image_mzs) == 0:
        return None
    diffs = np.abs(np.asarray(ion_image_mzs) - feature_mz) / feature_mz * 1e6
    best = int(np.argmin(diffs))
    return best if diffs[best] < ppm else None


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pairwise_r(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson r for every row pair ``i < j`` of ``flat`` (n, n_pixels).

    Pairs involving a zero-variance row are left out, as ``_pearson_r`` returns NaN
    for them. Returns ``(r, i, j)``.
    """
    ok = flat.std(axis=1) >= 1e-9
    iu, ju = np.triu_indices(len(flat), k=1)
    keep = ok[iu] & ok[ju]
    iu, ju = iu[keep], ju[keep]
    if not len(iu):
        return np.empty(0), iu, ju
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(flat)[iu, ju], iu, ju


def _candidate_title(row: pd.Series) -> str:
    peptide = row.get("peptide", "?")
    protein = row.get("protein", "?")
    fmz = row.get("feature_mz", float("nan"))
    r1 = row.get("_score_r1", float("nan"))
    rank = row.get("_rank", "?")
    total = row.get("_total", "?")
    q = _get(row, "q_value")
    winner = bool(row.get("is_tdc_winner", False))
    passes = winner and np.isfinite(q) and q <= 0.01
    label = "PASS" if passes else "FAIL"
    score_str = f"score={r1:.3f} | " if np.isfinite(r1) else ""
    q_str = f"q={q:.3f} | " if np.isfinite(q) else ""
    return f"{peptide} | {protein} | m/z {fmz:.4f} | {score_str}rank {rank}/{total} | {q_str}{label}"


def _safe_fname(s: str, maxlen: int = 45) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)[:maxlen]


def _get(row: pd.Series, col: str) -> float:
    v = row.get(col, float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = float("nan")
    return v


def _fdr_frame_color(
    qval: float, fdr_strict: float = 0.01, fdr_loose: float = 0.05
) -> str:
    """Frame colour for an ion-image panel by the identified peptide's FDR:
    dark green at q ≤ 1%, light green at q ≤ 5%, white otherwise (incl. NaN)."""
    try:
        q = float(qval)
    except (TypeError, ValueError):
        return "white"
    if not np.isfinite(q):
        return "white"
    if q <= fdr_strict:
        return "#006400"  # dark green
    if q <= fdr_loose:
        return "#90EE90"  # light green
    return "white"


def _na(ax, title: str | None = None) -> None:
    """Grey "N/A" placeholder for a panel with no data."""
    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
            transform=ax.transAxes, fontsize=14, color="gray")
    if title:
        ax.set_title(title, fontsize=8)


def _barh_panel(ax, row: pd.Series, cols, title: str, color, alpha: float = 0.75) -> bool:
    """Horizontal bars for the finite values of ``cols`` (``[(column, label)]``) in ``row``.

    ``color`` is one colour or a function ``(column, value) -> colour``. Draws "N/A"
    when no value is finite. Returns whether any bar was drawn.
    """
    ax.set_title(title, fontsize=8)
    vals = [(c, lab, _get(row, c)) for c, lab in cols]
    vals = [t for t in vals if np.isfinite(t[2])]
    if not vals:
        _na(ax)
        return False
    colors = [color(c, v) for c, _, v in vals] if callable(color) else color
    ax.barh(range(len(vals)), [v for *_, v in vals], color=colors, alpha=alpha)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([lab for _, lab, _ in vals], fontsize=7)
    return True


def _envelopes(row: pd.Series, maldi_envelopes: dict | None):
    """``(observed, theoretical)`` M/M+1/M+2 envelopes of a candidate, each or None."""
    from msi_picasso.utils import theoretical_isotope_distribution

    obs = None
    fmz = row.get("feature_mz")
    if maldi_envelopes is not None and fmz is not None:
        raw = maldi_envelopes.get(float(fmz))
        if raw is not None:
            obs = np.asarray(raw[:3], dtype=float)
    theo = None
    comp_cols = ["n_C", "n_H", "n_N", "n_O", "n_S"]
    if all(c in row.index for c in comp_cols):
        try:
            comp = tuple(int(_get(row, c)) for c in comp_cols)
            if all(v >= 0 for v in comp):
                theo = np.asarray(theoretical_isotope_distribution(*comp, n_peaks=3), dtype=float)[:3]
        except Exception:
            pass
    return obs, theo


def _cand_fname(row: pd.Series) -> str:
    """``{group}_{T|D}_{rank}_{peptide}[_{feature_mz}].png`` for a per-candidate figure."""
    stem = (
        f"{row.get('_group', 'L')}_{row.get('_td', 'T')}_{int(row.get('_rank', 0)):03d}_"
        f"{_safe_fname(str(row.get('peptide', 'unknown')))}"
    )
    fmz = row.get("feature_mz")
    return f"{stem}_{float(fmz):.4f}.png" if fmz is not None else f"{stem}.png"


# ---------------------------------------------------------------------------
# Subsystem 1: Ion image colocalization
# ---------------------------------------------------------------------------

#: q-value columns to choose a representative match by, best first.
_QVAL_PREFERENCE = ("q_value", "peptide_q_value")


def _one_row_per_peptide(subset: pd.DataFrame) -> pd.DataFrame:
    """Collapse peptide-feature rows to one row per protein, keeping the best q-value.

    Two levels of duplication, and this removes both.

    *Per peptide-feature pair.* A peptide matches 2.1 to 4.8 detected peaks at
    ``min_regions=2`` and up to 12.8 at ``min_regions=1`` (PROGRESS.md F-029, F-050), so
    one figure per row draws the same peptide many times over, differing only in which
    feature is the precursor panel.

    *Per peptide of one protein.* Every figure already shows the whole protein — precursor
    plus all same-protein co-features plus the protein mean — so two peptides of one
    protein give the same panel set in a different order. The protein is therefore the
    unit, and the peptide is the fallback only when no protein column exists.

    Targets and decoys are kept apart. Decoy proteins carry a ``DECOY_`` prefix so they
    already separate, but keying on the name alone would silently merge the two classes
    if that ever stopped being true.
    """
    if subset is None or not len(subset):
        return subset
    unit = "protein" if "protein" in subset.columns else "peptide"
    if unit not in subset.columns:
        return subset
    key = [unit] + (["is_decoy"] if "is_decoy" in subset.columns else [])
    qcol = next((c for c in _QVAL_PREFERENCE if c in subset.columns), None)
    if qcol is None:
        # No q-value to rank by: keep the first row rather than dropping the figures
        # entirely, so a run without a scored result still produces diagnostics.
        return subset.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    return (
        subset
        .sort_values(qcol, ascending=True, na_position="last", kind="stable")
        .drop_duplicates(subset=key, keep="first")
        .reset_index(drop=True)
    )


def _co_feature_panels(
    features_df: pd.DataFrame,
    protein,
    precursor_mz: float,
    precursor_peptide: str,
    feature_qvals: dict | None,
    feature_peptides: dict | None,
) -> list:
    """The co-feature panels for one protein: ``[(feature_mz, peptide), ...]``.

    **One panel per peptide, not per peptide-feature pair.** A peptide matches several
    detected peaks — 2.1 to 4.8 at ``min_regions=2`` and up to 12.8 at 1 (F-029, F-050) —
    and at ``min_regions=1`` those are near-duplicate peaks of the same ion, so the panels
    are visually identical and crowd out the protein's other peptides. Measured on kidney
    E030's P47963: 43 peaks for **9 peptides**, with TIGISVDPR taking 6 panels between
    957.5339 and 957.5452.

    Each peptide is represented by **the best peak it actually wins**. Ranking on the
    feature's own q-value alone picks the peak carrying the best identification, which is
    often one a stronger peptide owns, so nearly every panel came out annotated
    "(not winner: ...)" — the least representative peak of the set.

    The precursor's peptide is excluded: it already has its own panel.
    """
    prot_rows = features_df.loc[features_df["protein"] == protein]
    if not len(prot_rows) or "peptide" not in prot_rows.columns:
        return []
    pep_at_mz: dict = {}
    for mz, pep in zip(prot_rows["feature_mz"], prot_rows["peptide"]):
        if pd.notna(mz):
            pep_at_mz.setdefault(float(mz), str(pep))

    def sort_key(m):
        q = feature_qvals.get(m, float("nan")) if feature_qvals else float("nan")
        winner = feature_peptides.get(m, "") if feature_peptides else ""
        mine = pep_at_mz.get(m, "")
        not_winner = 1 if (winner and mine and winner != mine) else 0
        finite = np.isfinite(q) if q is not None else False
        return (not_winner, 0 if finite else 1, q if finite else float("inf"), m)

    candidates = sorted(
        (m for m in pep_at_mz if abs(m - precursor_mz) > 1e-6), key=sort_key
    )
    seen = {precursor_peptide} if precursor_peptide and precursor_peptide != "unknown" else set()
    out = []
    for m in candidates:
        pep = pep_at_mz.get(m, "")
        if pep:
            if pep in seen:
                continue
            seen.add(pep)
        out.append((m, pep))
    return out


def _reset_figure_dir(out_dir: str) -> None:
    """Empty a figure directory so a rerun replaces its figures instead of joining them.

    Figures are named with a rank, and the rank of a protein changes between runs, so
    re-running into an existing output directory leaves both copies: kidney's E025
    directory held 103 files for 59 figures, the surplus being the superseded first run
    of E025 sitting beside the corrected one under different ranks. Nothing distinguishes
    them in a listing, which is exactly the "same protein, different rank" confusion.
    """
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name.endswith(".png"):
                try:
                    os.remove(os.path.join(out_dir, name))
                except OSError:
                    pass
    os.makedirs(out_dir, exist_ok=True)
    _FIGURES_WRITTEN.pop(out_dir, None)


#: Figures already written into each output directory this run, as ``{out_dir: {tag}}``.
#: ``plot_ion_image_colocalization`` is called twice into ``ion_images/`` — once for the
#: protein-level set and once for the ground-truth peptides — and each call can only
#: deduplicate against itself, so a ground-truth peptide's protein came out twice. Cleared
#: by ``_reset_figure_dir`` at the start of a run.
_FIGURES_WRITTEN: dict[str, set] = {}


def plot_ion_image_colocalization(
    subset: pd.DataFrame,
    features_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    feature_qvals: dict | None = None,
    feature_peptides: dict | None = None,
) -> None:
    """
    **At most one figure per peptide**, never one per peptide-feature pair. A peptide
    matches several detected peaks (F-029), and a figure per match redraws the same
    peptide with a different panel promoted to precursor. ``_one_row_per_peptide``
    collapses ``subset`` here, keeping the match with the best q-value, so every caller
    gets this and none has to remember to.

    The main caller collapses further, to one representative row — the lowest-q peptide —
    **per protein**, because every figure already shows the whole protein and per-peptide
    figures of one protein differ only in panel order. The ground-truth caller does not:
    there the point is the named peptides, so each gets its own figure.

    Each figure is the representative feature's ion image + one co-feature panel **per
    same-protein peptide** (its best-q match, ranked by q-value ascending) +
    the mean over those panels. The panels are per peptide for the same reason the figures
    are: a peptide's several matched peaks are near-duplicates of one ion at
    ``min_regions=1`` and their images are visually identical.

    Co-feature panels show the same-protein candidate peptide as the label.  When
    a different-protein peptide is the TDC winner at that feature, it is annotated
    as "(not winner: <winner>)" so mass-coincidence competitors are visible.

    Files are saved as ``{out_dir}/{T|D}_{rank:03d}_{protein}.png`` (rank = protein
    rank by best q-value).
    Panels are arranged in a grid of up to 8 columns.  Each ion-image panel is
    framed by the identified peptide's FDR at that feature (``feature_qvals``):
    dark green at q ≤ 1%, light green at q ≤ 5%, white (no frame) otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)
    subset = _one_row_per_peptide(subset)
    already = _FIGURES_WRITTEN.setdefault(out_dir, set())

    n_saved = 0
    n_skipped = 0
    for _, row in subset.iterrows():
        try:
            feature_mz = row.get("feature_mz")
            if feature_mz is None or not np.isfinite(float(feature_mz)):
                continue
            feature_mz = float(feature_mz)
            rank = int(row.get("_rank", 0))
            peptide = str(row.get("peptide", "unknown"))
            protein = row.get("protein")

            prefix = str(row.get("_group", "L"))
            td = str(row.get("_td", "T"))

            # One figure per protein across the whole run, not just within this call.
            # This function is called twice into the same directory — the protein-level
            # set, then the ground-truth peptides — and a ground-truth peptide's protein
            # is usually in both. The first call wins, and it already chose that
            # protein's best-q peptide, so nothing better is being discarded.
            _prot_tag = _safe_fname(str(protein)) if protein else _safe_fname(peptide)
            if (td, _prot_tag) in already:
                n_skipped += 1
                continue
            already.add((td, _prot_tag))

            prec_idx = _find_image_idx(feature_mz, ion_image_mzs)
            if prec_idx is None:
                continue
            prec_img = ion_images[prec_idx]

            # Collect co-feature images for the same protein, ranked by q-value.
            co_imgs: list[np.ndarray] = []
            co_mzs: list[float] = []
            co_pep_labels: list[str] = []
            if protein and "protein" in features_df.columns and "feature_mz" in features_df.columns:
                for mz, co_pep in _co_feature_panels(
                    features_df, protein, feature_mz, peptide,
                    feature_qvals, feature_peptides,
                ):
                    co_img_idx = _find_image_idx(mz, ion_image_mzs)
                    if co_img_idx is None:
                        continue
                    winner_pep = feature_peptides.get(mz, "") if feature_peptides else ""
                    label = co_pep
                    if co_pep and winner_pep and co_pep != winner_pep:
                        label += f"\n(not winner: {winner_pep})"
                    co_imgs.append(ion_images[co_img_idx])
                    co_mzs.append(mz)
                    co_pep_labels.append(label)

            all_imgs = [prec_img] + co_imgs
            prot_mean = np.mean(all_imgs, axis=0)

            n_panels = 1 + len(co_imgs) + 1
            _ncols = min(n_panels, 8)
            _nrows = (n_panels + _ncols - 1) // _ncols
            fig, _axes_grid = plt.subplots(
                _nrows, _ncols,
                figsize=(3.2 * _ncols, 3.8 * _nrows),
                squeeze=False,
            )
            axes = _axes_grid.ravel()

            def _panel(
                ax: plt.Axes, img: np.ndarray, title: str,
                r: float | None = None, qval: float = float("nan"),
            ) -> None:
                im = ax.imshow(img, cmap="hot", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cap = title if r is None else f"{title}\nr={r:.2f}"
                ax.set_title(cap, fontsize=7)
                # FDR-coded frame: dark green at ≤1% FDR, light green at ≤5%, white otherwise.
                ax.set_xticks([])
                ax.set_yticks([])
                color = _fdr_frame_color(qval)
                lw = 0.0 if color == "white" else 3.5
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(lw)

            _prec_q = feature_qvals.get(feature_mz, float("nan")) if feature_qvals else float("nan")
            _prec_q_s = f"\nq={_prec_q:.3f}" if np.isfinite(_prec_q) else ""
            _panel(axes[0], prec_img, f"Precursor\n{feature_mz:.4f}{_prec_q_s}", qval=_prec_q)
            for i, (cimg, cmz, clabel) in enumerate(zip(co_imgs, co_mzs, co_pep_labels)):
                _q = feature_qvals.get(cmz, float("nan")) if feature_qvals else float("nan")
                _q_s = f"\nq={_q:.3f}" if np.isfinite(_q) else ""
                _co_pep_s = f"\n{clabel}" if clabel else ""
                _panel(axes[1 + i], cimg, f"{cmz:.4f}{_co_pep_s}{_q_s}",
                       r=_pearson_r(prec_img, cimg), qval=_q)
            _panel(axes[n_panels - 1], prot_mean, f"Protein mean\n({len(all_imgs)} imgs)",
                   r=_pearson_r(prec_img, prot_mean))
            for ax in axes[n_panels:]:
                ax.axis("off")

            fig.suptitle(_candidate_title(row), fontsize=8, y=1.01)
            plt.tight_layout()
            fname = f"{td}_{rank:03d}_{_prot_tag}.png"
            _save_and_close(fig, os.path.join(out_dir, fname), dpi=100)
            n_saved += 1
        except Exception as _row_exc:
            logger.debug(
                "Ion image colocalization: skipped row (feature_mz=%s): %s",
                row.get("feature_mz"),
                _row_exc,
            )
            try:
                plt.close("all")
            except Exception:
                pass
    if n_skipped:
        logger.info(
            "Ion image colocalization: %d figures saved, %d skipped as a protein already "
            "drawn this run", n_saved, n_skipped,
        )
    if n_saved == 0 and n_skipped == 0:
        # Only a warning when nothing was drawn AND nothing was deliberately skipped.
        # The ground-truth call legitimately saves nothing when every one of its proteins
        # was already drawn, and that must not read as a feature_mz alignment fault.
        logger.warning(
            "Ion image colocalization: 0 figures saved from %d candidates "
            "(ion_image_mzs has %d entries; check feature_mz alignment)",
            len(subset),
            len(ion_image_mzs) if ion_image_mzs is not None else 0,
        )


# ---------------------------------------------------------------------------
# Subsystem 2: Feature diagnostics
# ---------------------------------------------------------------------------

def plot_feature_diagnostics(
    subset: pd.DataFrame,
    features_df: pd.DataFrame,
    ion_images: np.ndarray | None,
    ion_image_mzs: np.ndarray | None,
    maldi_envelopes: dict | None,
    out_dir: str,
) -> None:
    """
    Per-candidate 4×3 diagnostic figure.

    Panels:
      [0,0] Ion image heatmap + fraction_detected / CV / Moran's I
      [0,1] Mass accuracy horizontal bar with ±2 / ±5 ppm shading
      [0,2] Observed vs theoretical isotope envelope
      [1,0] Peptide properties (bar chart)
      [1,1] Spatial statistics (bar chart)
      [1,2] LC-MS/MS features (bar chart, or "N/A")
      [2,0] CHCA cluster proximity gauge (chca_cluster_distance_ppm)
      [2,1] CHCA adduct colocalization (ion image thumbnails or Pearson r gauge)
      [2,2] Monoisotopic confidence gauge (monoisotopic_confidence)
      [3,0] CCS: raw vs m/z-detrended (``im2deep_*`` vs ``im2deep_*_resid``)
      [3,1] Ion-image colocalization (isotopologue + adduct Pearson r, incl. ``_mob``)
      [3,2] Theoretical isotope + mass-defect detail (bar chart)

    The bottom row surfaces features introduced after the original 3×3 layout:
    the m/z-detrended CCS variants (the ``mz_shuffle`` decoy-leak fix), the
    isotopologue/adduct ion-image colocalizations, and the theoretical-isotope
    and mass-defect quantities.
    """
    os.makedirs(out_dir, exist_ok=True)

    for _, row in subset.iterrows():
        feature_mz = row.get("feature_mz")
        if feature_mz is not None:
            feature_mz = float(feature_mz)

        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.38)
        ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)]

        # ------------------------------------------------------------------
        # [0,0] Ion image
        # ------------------------------------------------------------------
        if ion_images is not None and feature_mz is not None:
            prec_idx = _find_image_idx(feature_mz, ion_image_mzs)
        else:
            prec_idx = None

        if prec_idx is not None:
            im = ax[0][0].imshow(ion_images[prec_idx], cmap="hot", aspect="auto")
            plt.colorbar(im, ax=ax[0][0], fraction=0.046, pad=0.04)
            frac = _get(row, "fraction_detected")
            cv = _get(row, "intensity_cv")
            mi = _get(row, "spatial_autocorrelation")
            frac_s = f"f={frac:.2f}" if np.isfinite(frac) else ""
            cv_s = f"CV={cv:.2f}" if np.isfinite(cv) else ""
            mi_s = f"Moran={mi:.2f}" if np.isfinite(mi) else ""
            subtitle = "  ".join(s for s in [frac_s, cv_s, mi_s] if s)
            ax[0][0].set_title(f"Ion image\n{subtitle}", fontsize=8)
        else:
            ax[0][0].text(0.5, 0.5, "No ion image", ha="center", va="center", transform=ax[0][0].transAxes)
            ax[0][0].set_title("Ion image", fontsize=8)
        ax[0][0].axis("off")

        # ------------------------------------------------------------------
        # [0,1] Mass accuracy
        # ------------------------------------------------------------------
        ppm_abs = _get(row, "ppm_error_abs")
        ppm_signed = _get(row, "ppm_error")
        if not np.isfinite(ppm_signed):
            ppm_signed = ppm_abs  # fall back to unsigned if signed not available
        ax[0][1].axvspan(-2, 2, alpha=0.22, color="steelblue", label="±2 ppm")
        ax[0][1].axvspan(-5, 5, alpha=0.10, color="steelblue", label="±5 ppm")
        ax[0][1].axvline(0, color="gray", lw=0.8, ls="--")
        if np.isfinite(ppm_signed):
            color = "tomato" if abs(ppm_signed) > 5 else "steelblue"
            ax[0][1].barh([0], [ppm_signed], height=0.5, color=color, alpha=0.85)
            ax[0][1].set_xlim(-max(10, abs(ppm_signed) * 1.4), max(10, abs(ppm_signed) * 1.4))
        ax[0][1].set_yticks([])
        ax[0][1].set_xlabel("ppm error", fontsize=8)
        title_ppm = f"Mass accuracy\n{ppm_abs:.2f} ppm" if np.isfinite(ppm_abs) else "Mass accuracy"
        ax[0][1].set_title(title_ppm, fontsize=8)
        ax[0][1].legend(fontsize=6, loc="upper right")

        # ------------------------------------------------------------------
        # [0,2] Isotope envelope comparison
        # ------------------------------------------------------------------
        obs_env, theo_env = _envelopes(row, maldi_envelopes)

        if obs_env is not None or theo_env is not None:
            x = np.arange(3)
            w = 0.35
            if theo_env is not None:
                te = theo_env / theo_env.max() if theo_env.max() > 0 else theo_env
                ax[0][2].bar(x - w / 2, te, width=w, label="Theoretical", color="steelblue", alpha=0.8)
            if obs_env is not None:
                oe = obs_env / obs_env.max() if obs_env.max() > 0 else obs_env
                ax[0][2].bar(x + w / 2, oe, width=w, label="Observed", color="tomato", alpha=0.8)
            ax[0][2].set_xticks(x)
            ax[0][2].set_xticklabels(["M", "M+1", "M+2"], fontsize=8)
            ax[0][2].set_ylabel("Norm. intensity", fontsize=8)
            cosine = _get(row, "theo_isotope_cosine")
            cosine_s = f"cosine={cosine:.3f}" if np.isfinite(cosine) else ""
            ax[0][2].set_title(f"Isotope envelope\n{cosine_s}", fontsize=8)
            ax[0][2].legend(fontsize=6)
        else:
            ax[0][2].text(0.5, 0.5, "No envelope data", ha="center", va="center",
                          transform=ax[0][2].transAxes)
            ax[0][2].set_title("Isotope envelope", fontsize=8)

        # ------------------------------------------------------------------
        # [1,0] Peptide properties, [1,1] spatial statistics, [1,2] LC-MS/MS + CCS
        # ------------------------------------------------------------------
        _barh_panel(ax[1][0], row, [
            ("n_arginine", "Arg count"),
            ("n_basic_residues", "Basic residues"),
            ("gravy_score", "GRAVY"),
            ("peptide_length", "Length"),
            ("n_missed_cleavages", "Missed cleavages"),
        ], "Peptide properties", "steelblue")
        _barh_panel(ax[1][1], row, [
            ("spatial_autocorrelation", "Moran's I"),
            ("spatial_morans_i", "Moran's I (full)"),
            ("spatial_gearys_c", "Geary's C"),
            ("fraction_detected", "Fraction detected"),
            ("intensity_cv", "Intensity CV"),
            ("spatial_entropy", "Entropy"),
        ], "Spatial statistics", "seagreen")
        if _barh_panel(ax[1][2], row, [
            ("lcms_ms2_spectral_angle", "MS2 spectral angle"),
            ("lcms_ms1_intensity", "MS1 intensity"),
            ("lcms_ms1_snr", "MS1 SNR"),
            ("lcms_ms1_isotope_cosine", "MS1 iso cosine"),
            ("theo_m1_ratio_diff_lcms", "M+1 ratio diff (LC-MS)"),
            ("isotope_envelope_cosine", "Envelope cosine"),
            ("lcms_q_value", "LC-MS q-value"),
            ("im2deep_delta_ccs", "Δ CCS (Å²)"),
            ("im2deep_abs_delta_ccs_pct", "|Δ CCS| (%)"),
            ("im2deep_ccs_zscore", "CCS z-score"),
            ("im2deep_ccs_rank", "CCS rank"),
        ], "LC-MS/MS + CCS features",
            lambda c, v: "darkorange" if c.startswith("im2deep") else "mediumpurple"):
            ax[1][2].axvline(0, color="gray", lw=0.6, ls="--")

        # ------------------------------------------------------------------
        # [2,0] CHCA cluster proximity
        # ------------------------------------------------------------------
        chca_dist = _get(row, "chca_cluster_distance_ppm")
        if feature_mz is not None and feature_mz > 0:
            _ppm_dists = np.abs(_CHCA_CLUSTER_MZS - feature_mz) / feature_mz * 1e6
            _nearest_c_idx = int(np.argmin(_ppm_dists))
            nearest_cluster_mz = float(_CHCA_CLUSTER_MZS[_nearest_c_idx])
        else:
            nearest_cluster_mz = float("nan")

        if np.isfinite(chca_dist):
            _disp = min(chca_dist, 100.0)
            _bc = "tomato" if chca_dist < 20.0 else ("orange" if chca_dist < 50.0 else "seagreen")
            ax[2][0].barh([0], [_disp], height=0.5, color=_bc, alpha=0.85)
            ax[2][0].set_xlim(0, 100)
            ax[2][0].axvline(20.0, color="tomato", lw=1.0, ls="--", alpha=0.6, label="20 ppm")
            ax[2][0].axvline(50.0, color="orange", lw=1.0, ls="--", alpha=0.6, label="50 ppm")
            if chca_dist > 100.0:
                ax[2][0].text(96, 0, f">{chca_dist:.0f}", ha="right", va="center", fontsize=7)
            ax[2][0].set_yticks([])
            ax[2][0].set_xlabel("ppm to nearest CHCA cluster", fontsize=8)
            ax[2][0].legend(fontsize=6, loc="upper left")
            _ncmz_s = f" (CHCA@{nearest_cluster_mz:.4f})" if np.isfinite(nearest_cluster_mz) else ""
            ax[2][0].set_title(f"CHCA proximity\n{chca_dist:.1f} ppm{_ncmz_s}", fontsize=8)
        else:
            _na(ax[2][0], "CHCA proximity")

        # ------------------------------------------------------------------
        # [2,1] CHCA adduct colocalization
        # ------------------------------------------------------------------
        chca_corr = _get(row, "adduct_colocalization_chca")
        _adduct_mz = (feature_mz + _CHCA_ADDUCT_DELTA) if feature_mz is not None else None
        _adduct_idx = (
            _find_image_idx(_adduct_mz, ion_image_mzs)
            if (_adduct_mz is not None and ion_image_mzs is not None)
            else None
        )

        if _adduct_idx is not None and prec_idx is not None:
            ax[2][1].axis("off")
            _axl = ax[2][1].inset_axes([0.02, 0.10, 0.44, 0.78])
            _axl.imshow(ion_images[prec_idx], cmap="hot", aspect="auto")
            _axl.set_title(f"Precursor\n{feature_mz:.4f}", fontsize=6)
            _axl.axis("off")
            _axr = ax[2][1].inset_axes([0.54, 0.10, 0.44, 0.78])
            _axr.imshow(ion_images[_adduct_idx], cmap="hot", aspect="auto")
            _axr.set_title(f"CHCA adduct\n{_adduct_mz:.4f}", fontsize=6)
            _axr.axis("off")
            _corr_s = f"r={chca_corr:.3f}" if np.isfinite(chca_corr) else "r=N/A"
            ax[2][1].set_title(f"CHCA adduct colocalization\n{_corr_s}", fontsize=8)
        else:
            if np.isfinite(chca_corr):
                _bar_color_chca = plt.cm.RdYlGn_r((chca_corr + 1.0) / 2.0)
                ax[2][1].barh([0], [chca_corr], height=0.5, color=_bar_color_chca, alpha=0.85)
                ax[2][1].set_xlim(-1.0, 1.0)
                ax[2][1].axvline(0.0, color="gray", lw=0.8, ls="--")
                ax[2][1].axvline(0.5, color="orange", lw=0.8, ls=":", alpha=0.7, label="r=0.5")
                ax[2][1].set_yticks([])
                ax[2][1].set_xlabel("Pearson r (precursor vs CHCA adduct)", fontsize=8)
                ax[2][1].legend(fontsize=6, loc="upper left")
                _corr_s = f"r={chca_corr:.3f}"
            else:
                _na(ax[2][1])
                _corr_s = "N/A"
            ax[2][1].set_title(f"CHCA adduct colocalization\n{_corr_s}", fontsize=8)

        # ------------------------------------------------------------------
        # [2,2] Monoisotopic confidence
        # ------------------------------------------------------------------
        mono_conf = _get(row, "monoisotopic_confidence")
        if np.isfinite(mono_conf):
            _bar_color_mono = plt.cm.RdYlGn(float(mono_conf))
            ax[2][2].barh([0], [mono_conf], height=0.5, color=_bar_color_mono, alpha=0.85)
            ax[2][2].set_xlim(0.0, 1.0)
            ax[2][2].axvline(0.5, color="orange", lw=1.0, ls="--", alpha=0.8, label="0.5")
            ax[2][2].set_yticks([])
            ax[2][2].set_xlabel("Monoisotopic confidence", fontsize=8)
            ax[2][2].legend(fontsize=6, loc="upper left")
            ax[2][2].set_title(f"Monoisotopic confidence\n{mono_conf:.3f}", fontsize=8)
        else:
            _na(ax[2][2], "Monoisotopic confidence")

        # ------------------------------------------------------------------
        # [3,0] CCS: raw vs m/z-detrended (im2deep_* vs im2deep_*_resid)
        # ------------------------------------------------------------------
        # The *_resid variants subtract the expected m/z-gap CCS difference, so
        # for relocated decoys (mz_shift / mz_shuffle / entrapment) they remove
        # the trivial m/z-baseline separation that the raw deltas would leak.
        _ccs_pairs = [
            ("im2deep_delta_ccs", "im2deep_delta_ccs_resid", "Δ CCS (Å²)"),
            ("im2deep_abs_delta_ccs_pct", "im2deep_abs_delta_ccs_pct_resid", "|Δ CCS| (%)"),
            ("im2deep_ccs_zscore", "im2deep_ccs_zscore_resid", "CCS z-score"),
            ("im2deep_ccs_rank", "im2deep_ccs_rank_resid", "CCS rank"),
        ]
        _ccs_names, _ccs_raw, _ccs_res = [], [], []
        for raw_col, res_col, lab in _ccs_pairs:
            rv, sv = _get(row, raw_col), _get(row, res_col)
            if np.isfinite(rv) or np.isfinite(sv):
                _ccs_names.append(lab)
                _ccs_raw.append(rv if np.isfinite(rv) else 0.0)
                _ccs_res.append(sv if np.isfinite(sv) else 0.0)
        if _ccs_names:
            _y = np.arange(len(_ccs_names))
            _h = 0.38
            ax[3][0].barh(_y + _h / 2, _ccs_raw, height=_h, color="darkorange",
                          alpha=0.85, label="raw")
            ax[3][0].barh(_y - _h / 2, _ccs_res, height=_h, color="teal",
                          alpha=0.85, label="m/z-detrended")
            ax[3][0].set_yticks(_y)
            ax[3][0].set_yticklabels(_ccs_names, fontsize=7)
            ax[3][0].axvline(0, color="gray", lw=0.6, ls="--")
            ax[3][0].legend(fontsize=6, loc="best")
        else:
            _na(ax[3][0])
        ax[3][0].set_title("CCS: raw vs m/z-detrended", fontsize=8)

        # ------------------------------------------------------------------
        # [3,1] Ion-image colocalization (isotopologue + adduct Pearson r)
        # ------------------------------------------------------------------
        if _barh_panel(ax[3][1], row, [
            ("isotope_image_colocalization_m1", "iso M+1"),
            ("isotope_image_colocalization_m2", "iso M+2"),
            ("isotope_image_colocalization_mean", "iso mean"),
            ("adduct_colocalization_na", "adduct Na"),
            ("adduct_colocalization_k", "adduct K"),
            ("adduct_colocalization_chca", "adduct CHCA"),
            ("protein_colocalization", "protein (mean)"),
            # Mobility-gated variants (raw-query / --mob-coloc only).
            ("isotope_colocalization_mean_mob", "iso mean (mob)"),
            ("adduct_colocalization_chca_mob", "adduct CHCA (mob)"),
        ], "Ion-image colocalization",
            lambda c, v: "seagreen" if v >= 0 else "tomato", alpha=0.78):
            ax[3][1].set_xlim(-1.0, 1.0)
            ax[3][1].axvline(0.0, color="gray", lw=0.8, ls="--")
            ax[3][1].axvline(0.5, color="orange", lw=0.7, ls=":", alpha=0.7)
            ax[3][1].set_xlabel("Pearson r", fontsize=8)

        # ------------------------------------------------------------------
        # [3,2] Theoretical isotope + mass-defect detail
        # ------------------------------------------------------------------
        if _barh_panel(ax[3][2], row, [
            ("theo_isotope_cosine", "iso cosine"),
            ("theo_isotope_chi2", "iso χ²"),
            ("theo_isotope_kl", "iso KL"),
            ("averagine_deviation", "averagine dev"),
            ("averagine_deviation_sulfur", "averagine dev (S)"),
            ("theo_m1_ratio_diff", "ΔM+1 ratio"),
            ("theo_m2_ratio_diff", "ΔM+2 ratio"),
            ("kendrick_mass_defect", "Kendrick defect"),
            ("mass_defect_residual", "mass-defect resid"),
        ], "Theoretical isotope + mass defect", "slateblue", alpha=0.78):
            ax[3][2].axvline(0, color="gray", lw=0.6, ls="--")

        # ------------------------------------------------------------------
        fig.suptitle(_candidate_title(row), fontsize=9, y=1.01)
        plt.tight_layout()
        _save_and_close(fig, os.path.join(out_dir, _cand_fname(row)), dpi=100)


# ---------------------------------------------------------------------------
# Subsystem 3: Isotope envelope
# ---------------------------------------------------------------------------

_NEUTRON = 1.003355
_ISOTOPE_DISPLAY_FWHM = 0.05  # Da — chosen for visual clarity, not instrument-specific

_CHCA_CLUSTER_MZS = np.array([
    172.0393, 190.0499, 212.0318, 228.0058,
    379.0925, 401.0744, 568.1351, 757.1777,
])
_CHCA_ADDUCT_DELTA = 172.0392  # [M+H]+ → [M+CHCA+H-H2O]+ m/z offset


def _gauss(mz_arr: np.ndarray, center: float, intensity: float) -> np.ndarray:
    sigma = _ISOTOPE_DISPLAY_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return intensity * np.exp(-0.5 * ((mz_arr - center) / sigma) ** 2)


def plot_isotope_envelope_figures(
    subset: pd.DataFrame,
    maldi_envelopes: dict | None,
    out_dir: str,
) -> None:
    """
    Per-candidate isotope envelope figure in spectrum style.

    Simulates peak shapes as Gaussians at the actual M, M+1, M+2 m/z positions
    so the x-axis is m/z (not categorical).  Observed MALDI envelope is shown as
    a filled blue trace; theoretical (from elemental composition, normalized to
    M0 = 1) is overlaid as a dashed red line.  Peak windows are shaded in orange
    (style matches optimize_maldi_params.py interval shading), and the detected
    apex positions are marked with triangles.
    """
    os.makedirs(out_dir, exist_ok=True)

    for _, row in subset.iterrows():
        feature_mz = row.get("feature_mz")
        if feature_mz is not None:
            feature_mz = float(feature_mz)
        obs_env, theo_env = _envelopes(row, maldi_envelopes)

        if obs_env is None and theo_env is None:
            continue
        if feature_mz is None:
            continue

        n_peaks = 3
        peak_mzs = np.array([feature_mz + i * _NEUTRON for i in range(n_peaks)])

        # Normalize: M0 = 1 for whichever series is available
        obs_norm = None
        if obs_env is not None and obs_env[0] > 0:
            obs_norm = obs_env / obs_env[0]

        theo_norm = None
        if theo_env is not None and theo_env[0] > 0:
            theo_norm = theo_env / theo_env[0]

        # Fine m/z grid spanning all isotope peaks
        mz_lo = feature_mz - 3 * _ISOTOPE_DISPLAY_FWHM
        mz_hi = peak_mzs[-1] + 3 * _ISOTOPE_DISPLAY_FWHM
        mz_fine = np.linspace(mz_lo, mz_hi, 2000)

        fig, ax = plt.subplots(figsize=(8, 4))

        # Orange shading for peak windows (±FWHM around each peak)
        for mz_peak in peak_mzs:
            ax.axvspan(
                mz_peak - _ISOTOPE_DISPLAY_FWHM,
                mz_peak + _ISOTOPE_DISPLAY_FWHM,
                alpha=0.12, color="orange",
            )

        # Observed trace
        if obs_norm is not None:
            obs_spectrum = sum(_gauss(mz_fine, mz_peak, h)
                               for mz_peak, h in zip(peak_mzs, obs_norm))
            ax.fill_between(mz_fine, 0, obs_spectrum, color="steelblue", alpha=0.55,
                            label="Observed (MALDI)")
            ax.plot(mz_fine, obs_spectrum, color="steelblue", lw=1.0)
            # Apex markers
            for mz_peak, h in zip(peak_mzs, obs_norm):
                ax.plot(mz_peak, h, "^", color="steelblue", ms=8, zorder=5)

        # Theoretical overlay
        if theo_norm is not None:
            theo_spectrum = sum(_gauss(mz_fine, mz_peak, h)
                                for mz_peak, h in zip(peak_mzs, theo_norm))
            ax.plot(mz_fine, theo_spectrum, color="tomato", lw=1.8, ls="--",
                    label="Theoretical", zorder=4)
            for mz_peak, h in zip(peak_mzs, theo_norm):
                ax.axvline(mz_peak, color="tomato", lw=0.7, ls=":", alpha=0.6)

        # Peak m/z labels on x-axis
        ax.set_xticks(peak_mzs)
        ax.set_xticklabels(
            [f"M+{i}\n{mz:.4f}" for i, mz in enumerate(peak_mzs)], fontsize=8
        )
        ax.set_xlim(mz_lo, mz_hi)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("m/z", fontsize=9)
        ax.set_ylabel("Relative intensity (M0 = 1)", fontsize=9)

        cosine = _get(row, "theo_isotope_cosine")
        m1_diff = _get(row, "theo_m1_ratio_diff")
        m2_diff = _get(row, "theo_m2_ratio_diff")
        parts = []
        if np.isfinite(cosine):
            parts.append(f"cosine={cosine:.3f}")
        if np.isfinite(m1_diff):
            parts.append(f"ΔM+1={m1_diff:+.3f}")
        if np.isfinite(m2_diff):
            parts.append(f"ΔM+2={m2_diff:+.3f}")
        ax.set_title("  ".join(parts), fontsize=8)
        ax.legend(fontsize=8)

        fig.suptitle(_candidate_title(row), fontsize=9, y=1.01)
        plt.tight_layout()
        _save_and_close(fig, os.path.join(out_dir, _cand_fname(row)), dpi=100)


# ---------------------------------------------------------------------------
# Subsystem 4: Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    names: list[str],
    importances: np.ndarray | None,
    out_dir: str,
    model_name: str = "model",
    top_n: int = 30,
    structure_coefs: np.ndarray | None = None,
    structure_names: list[str] | None = None,
) -> None:
    """
    Save the feature importance figure.

    When structure coefficients are provided, the figure has two panels
    (paired horizontal bar chart):
      Left  — whatever the backend reports as an importance, normalised to [-1, 1]
               by the maximum absolute value: ``coef_`` for the linear models
               (signed, and inflatable by collinearity), permutation importance
               for the kernel model that has no ``coef_`` (H-model-2).
      Right — structure coefficient: Pearson r between each (scaled) feature and
               the discriminant score.  Bounded in [-1, 1] and unaffected by
               collinearity.  Features are sorted top-to-bottom by |structure coef|.

    When structure coefficients are absent the original single-panel plot is
    produced.  Blue = positive (target-like), red = negative (decoy-like).

    File: ``{out_dir}/{model_name}_feature_importance.png``.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _one(
        importances: np.ndarray | None,
        names: list[str],
        struct_coefs: np.ndarray | None = None,
        struct_names: list[str] | None = None,
    ) -> None:
        if importances is None or len(importances) == 0:
            return
        importances = np.asarray(importances, dtype=float)
        if len(importances) != len(names):
            logger.warning(
                "Feature importance length mismatch: %d importances vs %d names — skipping",
                len(importances), len(names),
            )
            return

        has_struct = (
            struct_coefs is not None
            and struct_names is not None
            and len(struct_coefs) == len(struct_names)
            and len(struct_coefs) > 0
        )

        if has_struct:
            struct_coefs_arr = np.asarray(struct_coefs, dtype=float)
            # Map raw coef by feature name.
            name_to_raw = dict(zip(names, importances))
            raw_for_struct = np.array([name_to_raw.get(n, np.nan) for n in struct_names])

            # Sort by |structure coef|, largest at top (barh: index 0 = bottom).
            order = np.argsort(np.abs(struct_coefs_arr))[-top_n:]
            plot_names = [struct_names[i] for i in order]
            s_vals = struct_coefs_arr[order]
            r_vals = raw_for_struct[order]

            # Normalise raw coefs to [-1, 1] so both axes share the same scale.
            r_max = np.nanmax(np.abs(r_vals)) if np.any(np.isfinite(r_vals)) else 1.0
            r_norm = r_vals / (r_max + 1e-12)

            n_feats = len(plot_names)
            fig, (ax_raw, ax_struct) = plt.subplots(
                1, 2,
                figsize=(14, max(5, n_feats * 0.38)),
                sharey=True,
            )

            raw_colors = ["steelblue" if v >= 0 else "tomato" for v in r_norm]
            ax_raw.barh(range(n_feats), r_norm, color=raw_colors, alpha=0.80, height=0.65)
            ax_raw.axvline(0, color="black", lw=0.8)
            ax_raw.set_xlim(-1.12, 1.12)
            ax_raw.set_yticks(range(n_feats))
            ax_raw.set_yticklabels(plot_names, fontsize=7)
            ax_raw.set_xlabel("Reported importance  (normalised to max abs)", fontsize=9)
            # The left panel is whatever the backend reports: coef_ for the linear
            # models, permutation importance for the kernel model (H-model-2). Only
            # coef_ is signed, and only it can be inflated by collinearity.
            ax_raw.set_title("Reported importance\ncoef_ / permutation importance", fontsize=9)

            struct_colors = ["steelblue" if v >= 0 else "tomato" for v in s_vals]
            ax_struct.barh(range(n_feats), s_vals, color=struct_colors, alpha=0.80, height=0.65)
            ax_struct.axvline(0, color="black", lw=0.8)
            ax_struct.set_xlim(-1.12, 1.12)
            ax_struct.set_xlabel("Structure coef  r(feature, discriminant score)", fontsize=9)
            ax_struct.set_title("Structure coefficient\n(collinearity-robust, bounded [−1, 1])", fontsize=9)
            ax_struct.tick_params(labelleft=False)

            fig.suptitle(
                f"{model_name}: raw vs structure importance "
                f"(top {n_feats} by |structure coef|  ·  blue = target-like / red = decoy-like)",
                fontsize=9, y=1.01,
            )
        else:
            order = np.argsort(np.abs(importances))[-top_n:]
            plot_names = [names[i] for i in order]
            vals = importances[order]
            colors = ["steelblue" if v >= 0 else "tomato" for v in vals]

            fig, ax_raw = plt.subplots(figsize=(9, max(4, len(plot_names) * 0.32)))
            ax_raw.barh(range(len(plot_names)), vals, color=colors, alpha=0.82)
            ax_raw.set_yticks(range(len(plot_names)))
            ax_raw.set_yticklabels(plot_names, fontsize=7)
            ax_raw.axvline(0, color="black", lw=0.8)
            ax_raw.set_xlabel("Importance", fontsize=9)
            ax_raw.set_title(
                f"{model_name} feature importance (top {len(plot_names)})",
                fontsize=10,
            )

        plt.tight_layout()
        _save_and_close(fig, os.path.join(out_dir, f"{model_name}_feature_importance.png"), dpi=100)

    _one(importances, names, struct_coefs=structure_coefs, struct_names=structure_names)


# ---------------------------------------------------------------------------
# Subsystem 5: Feature distributions
# ---------------------------------------------------------------------------

_DIST_SKIP = frozenset({
    "is_decoy", "peptide", "protein", "feature_mz", "feature_idx", "source",
    "_group", "_td", "_rank", "_total", "_score_r1",
    "is_tdc_winner", "score", "q_value",
})


def plot_feature_distributions(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    feature_names: list[str] | None = None,
    gt_peptides: list[str] | None = None,
) -> None:
    """
    Per-feature target/decoy distribution figures.

    Two subplots per figure:
      Top    — all candidates
      Bottom — winners (is_tdc_winner == True)

    Overlapping histograms are drawn in steelblue (target) and tomato (decoy)
    with dashed median lines.  When ``gt_peptides`` is provided, a solid green
    vertical line is drawn at each GT candidate's value on both subplots.

    Files: ``{out_dir}/{feature_name}.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    is_decoy = _flag(feat, "is_decoy")
    is_winner = _flag(res, "is_tdc_winner")
    # Entrapment pseudo-targets: is_decoy=False but source=="entrapment_shuffled"
    _src = feat.get("source", pd.Series("", index=feat.index)).fillna("").values
    entrapment_mask = (~is_decoy) & (_src == "entrapment_shuffled")
    has_entrapment = entrapment_mask.any()

    target_mask = (~is_decoy) & (~entrapment_mask)
    decoy_mask = is_decoy
    winner_target_mask = is_winner & target_mask
    winner_decoy_mask = is_winner & decoy_mask
    winner_ent_mask = is_winner & entrapment_mask

    gt_mask = np.zeros(len(feat), dtype=bool)
    if gt_peptides and "peptide" in feat.columns:
        gt_set = set(gt_peptides)
        all_gt_mask = target_mask & feat["peptide"].isin(gt_set).values
        if all_gt_mask.any():
            # Each GT peptide may appear at multiple MALDI features; keep only the
            # row with the highest round-1 score so each peptide contributes exactly
            # one vertical line per feature distribution plot.
            _r1_col = next(
                (c for c in res.columns if c.endswith("_r1") and pd.api.types.is_numeric_dtype(res[c])),
                None,
            )
            if _r1_col is None:
                gt_mask = all_gt_mask
            else:
                gt_idx = np.where(all_gt_mask)[0]
                tmp = pd.DataFrame({
                    "row": gt_idx,
                    "peptide": feat["peptide"].values[gt_idx],
                    "score": pd.to_numeric(res[_r1_col].iloc[gt_idx].values, errors="coerce"),
                })
                best = tmp.sort_values("score", ascending=False).drop_duplicates("peptide")
                gt_mask[best["row"].values] = True

    # Always plot every numeric column in features_df that is not in _DIST_SKIP.
    # Explicitly listed features (the ranker inputs) come first; the remaining
    # numeric columns — LC-MS/MS prior features, spatial prior features, and
    # optional intrinsic features dropped before training — follow.
    _explicit = list(feature_names) if feature_names is not None else []
    _explicit_set = set(_explicit)
    _extra = [
        c for c in feat.columns
        if c not in _DIST_SKIP
        and c not in _explicit_set
        and pd.api.types.is_numeric_dtype(feat[c])
    ]
    feature_names = _explicit + _extra

    def _draw(ax: plt.Axes, t_vals: np.ndarray, d_vals: np.ndarray,
               bins: np.ndarray, subtitle: str,
               e_vals: np.ndarray | None = None) -> None:
        ax.set_title(subtitle, fontsize=8)
        if len(t_vals) > 0:
            ax.hist(t_vals, bins=bins, density=True, alpha=0.55,
                    color="steelblue", label=f"Target (n={len(t_vals)})")
            ax.axvline(float(np.nanmedian(t_vals)), color="steelblue",
                       lw=1.3, ls="--", alpha=0.85)
        if len(d_vals) > 0:
            ax.hist(d_vals, bins=bins, density=True, alpha=0.55,
                    color="tomato", label=f"Decoy (n={len(d_vals)})")
            ax.axvline(float(np.nanmedian(d_vals)), color="tomato",
                       lw=1.3, ls="--", alpha=0.85)
        if e_vals is not None and len(e_vals) > 0:
            ax.hist(e_vals, bins=bins, density=True, alpha=0.55,
                    color="goldenrod", label=f"Entrapment (n={len(e_vals)})")
            ax.axvline(float(np.nanmedian(e_vals)), color="goldenrod",
                       lw=1.3, ls="--", alpha=0.85)
        if len(t_vals) == 0 and len(d_vals) == 0 and (e_vals is None or len(e_vals) == 0):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            return
        ax.legend(fontsize=7)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    def _draw_gt(ax: plt.Axes, gt_vals: np.ndarray) -> None:
        for i, v in enumerate(gt_vals):
            ax.axvline(
                v, color="limegreen", lw=1.5, ls="-.", alpha=0.85,
                label=f"GT (n={len(gt_vals)})" if i == 0 else "_nolegend_",
            )
        if len(gt_vals) > 0:
            ax.legend(fontsize=7)

    for feat_col in feature_names:
        col = feat.get(feat_col)
        if col is None:
            continue
        vals = pd.to_numeric(col, errors="coerce").values.astype(float)

        finite_mask = np.isfinite(vals)
        t_all = vals[target_mask & finite_mask]
        d_all = vals[decoy_mask & finite_mask]
        t_r2 = vals[winner_target_mask & finite_mask]
        d_r2 = vals[winner_decoy_mask & finite_mask]
        e_all = vals[entrapment_mask & finite_mask] if has_entrapment else None
        e_r2  = vals[winner_ent_mask & finite_mask]  if has_entrapment else None
        gt_vals = vals[gt_mask & finite_mask]

        all_finite = vals[finite_mask]
        if len(all_finite) == 0:
            continue

        lo = float(np.percentile(all_finite, 1))
        hi = float(np.percentile(all_finite, 99))
        if lo >= hi:
            lo, hi = float(all_finite.min()), float(all_finite.max())
        if lo >= hi:
            lo, hi = lo - 1.0, hi + 1.0
        bins = np.linspace(lo, hi, 51)

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(feat_col, fontsize=10)

        _bot_label = "Winners"
        _e_all_n = len(e_all) if e_all is not None else 0
        _e_r2_n  = len(e_r2)  if e_r2  is not None else 0
        _top_title = (
            f"All candidates  (T={len(t_all)}, D={len(d_all)}, E={_e_all_n})"
            if has_entrapment
            else f"All candidates  (T={len(t_all)}, D={len(d_all)})"
        )
        _bot_title = (
            f"{_bot_label}  (T={len(t_r2)}, D={len(d_r2)}, E={_e_r2_n})"
            if has_entrapment
            else f"{_bot_label}  (T={len(t_r2)}, D={len(d_r2)})"
        )
        _draw(ax_top, t_all, d_all, bins, _top_title, e_vals=e_all)
        _draw(ax_bot, t_r2, d_r2, bins, _bot_title, e_vals=e_r2)

        _draw_gt(ax_top, gt_vals)
        _draw_gt(ax_bot, gt_vals)

        ax_bot.set_xlabel(feat_col, fontsize=8)
        plt.tight_layout()
        _save_and_close(fig, os.path.join(out_dir, f"{_safe_fname(feat_col, maxlen=80)}.png"), dpi=100)


# ---------------------------------------------------------------------------
# Subsystem 6: CCS scatter
# ---------------------------------------------------------------------------


def plot_ccs_scatter(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    gt_peptides: list[str] | None = None,
    ccs_tol_pct: float | None = None,
    title_extra: str = "",
    filename: str = "ccs_scatter.png",
) -> None:
    """
    Scatter plot of observed vs predicted CCS for all candidates.

    Requires ``im2deep_observed_ccs`` and ``im2deep_predicted_ccs`` columns in
    ``features_df`` (added by ``compute_im2deep_features``). Silently skips if
    neither column is present.

    Points are coloured by target/decoy status. "Winner" means the feature's
    best candidate AND q_value <= fdr_threshold. R1 winners (best
    candidate but below FDR threshold) are shown at intermediate size.

    When ``ccs_tol_pct`` is provided, fan-shaped CCS filter boundaries are drawn:
    ``obs = pred × (1 ± ccs_tol_pct/100)``. These diverge from the origin.
    Saved to ``{out_dir}/{filename}``.
    """
    if "im2deep_observed_ccs" not in features_df.columns:
        return
    if "im2deep_predicted_ccs" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    obs = pd.to_numeric(feat["im2deep_observed_ccs"], errors="coerce").values
    pred = pd.to_numeric(feat["im2deep_predicted_ccs"], errors="coerce").values
    is_decoy = _flag(feat, "is_decoy")
    is_winner = _flag(res, "is_tdc_winner")
    # passes_fdr = TDC winner AND passes FDR; r1_only = winner but below FDR
    passes_fdr = is_winner & (_num(res, "q_value") <= fdr_threshold)
    r1_only = is_winner & ~passes_fdr

    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return

    obs_v, pred_v = obs[valid], pred[valid]
    decoy_v = is_decoy[valid]
    fdr_v = passes_fdr[valid]
    r1_v = r1_only[valid]
    bg_v = ~fdr_v & ~r1_v

    fig, ax = plt.subplots(figsize=(7, 6))

    # Non-winners at the back, then R1 winners (best per feature, below FDR), then
    # FDR-passing winners on top. The R1 layer is drawn only when it has points.
    _fdr_lbl = f" (FDR ≤ {fdr_threshold:.0%})"
    for layer, suffix, kw in [
        (bg_v, "", dict(s=6, alpha=0.25, linewidths=0)),
        (r1_v, " (R1 winner)", dict(s=15, alpha=0.5, linewidths=0)),
        (fdr_v, _fdr_lbl, dict(s=50, alpha=0.9, linewidths=0.7, zorder=5)),
    ]:
        if layer is r1_v and not r1_v.any():
            continue
        for dec, color, edge, name in [
            (False, "steelblue", "navy", "Target"), (True, "tomato", "darkred", "Decoy"),
        ]:
            m = layer & (decoy_v == dec)
            edges = {"edgecolors": edge} if layer is fdr_v else {}
            ax.scatter(pred_v[m], obs_v[m], color=color, label=name + suffix, **edges, **kw)

    # GT peptide overlay
    if gt_peptides:
        gt_set = set(gt_peptides)
        pep_col = feat.get("peptide", pd.Series(dtype=str)).values
        gt_mask = np.array([p in gt_set for p in pep_col], dtype=bool) & valid
        if gt_mask.any():
            gt_pred = pred[gt_mask]
            gt_obs = obs[gt_mask]
            gt_names = pep_col[gt_mask]
            ax.scatter(
                gt_pred, gt_obs,
                s=150, marker="*", color="black", edgecolors="darkorange", linewidths=0.8,
                zorder=10,
            )

    # y = x reference line
    lo = min(float(pred_v.min()), float(obs_v.min()))
    hi = max(float(pred_v.max()), float(obs_v.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, label="y = x")

    # Fan-shaped CCS tolerance boundaries (diverge from origin)
    if ccs_tol_pct is not None:
        pred_range = np.linspace(float(pred_v.min()), float(pred_v.max()), 300)
        fac = ccs_tol_pct / 100.0
        ax.plot(pred_range, pred_range * (1 + fac), color="darkorange", lw=1.2,
                ls="--", alpha=0.85, label=f"±{ccs_tol_pct:.1f}% CCS threshold")
        ax.plot(pred_range, pred_range * (1 - fac), color="darkorange", lw=1.2,
                ls="--", alpha=0.85)

    # Linear regression across all valid points
    try:
        coeffs = np.polyfit(pred_v, obs_v, 1)
        x_fit = np.linspace(float(pred_v.min()), float(pred_v.max()), 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), color="gray", lw=1.5,
                ls="-", alpha=0.7,
                label=f"fit: y={coeffs[0]:.3f}x{coeffs[1]:+.1f}")
    except Exception:
        pass

    # Correlation annotation
    corr_all = float(np.corrcoef(pred_v, obs_v)[0, 1]) if len(pred_v) > 1 else float("nan")
    if fdr_v.sum() > 1:
        corr_fdr = float(np.corrcoef(pred_v[fdr_v], obs_v[fdr_v])[0, 1])
        corr_str = f"r (all) = {corr_all:.3f}   r (FDR ≤ {fdr_threshold:.0%}) = {corr_fdr:.3f}"
    else:
        corr_str = f"r = {corr_all:.3f}"

    title = f"Observed vs Predicted CCS\n{corr_str}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_xlabel("Predicted CCS (Å²)", fontsize=10)
    ax.set_ylabel("Observed CCS (Å²)", fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, filename))


# ---------------------------------------------------------------------------
# Subsystem 7: IDs vs FDR curve
# ---------------------------------------------------------------------------


def plot_ids_vs_fdr(
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_max: float = 0.20,
) -> None:
    """
    Save a curve of target identifications as a function of FDR threshold.

    Backend-agnostic: the model name is inferred from the ``*_score_r1``
    column in ``result_df``; no explicit model identifier is required.  Plots
    the TDC q-value.  Vertical lines mark 1 % and 5 % FDR.  Only TDC winner
    target rows are considered.

    Output: ``{out_dir}/ids_vs_fdr.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    # Infer model name from the first *_score_r1 column present.
    r1_cols = [c for c in result_df.columns if c.endswith("_score_r1")]
    model_name = r1_cols[0].removesuffix("_score_r1") if r1_cols else "model"

    if "q_value" not in result_df.columns:
        logger.warning("plot_ids_vs_fdr: no q_value column — skipping")
        return
    target_winners = _flag(result_df, "is_tdc_winner") & ~_flag(result_df, "is_decoy")
    vals = _num(result_df, "q_value")[target_winners]
    vals = vals[np.isfinite(vals)]

    fdr_grid = np.linspace(0.0, fdr_max, 500)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colour = "steelblue"
    if len(vals):
        ax.plot(fdr_grid * 100, [(vals <= t).sum() for t in fdr_grid],
                label="TDC q-value", color=colour, lw=2)
        for thresh, ls in [(0.01, "--"), (0.05, ":")]:
            if thresh <= fdr_max:
                n_at = int((vals <= thresh).sum())
                ax.axvline(thresh * 100, color=colour, lw=0.8, ls=ls, alpha=0.6)
                ax.annotate(f"{n_at}", xy=(thresh * 100, n_at), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color=colour)

    ax.set_xlabel("FDR threshold (%)", fontsize=10)
    ax.set_ylabel("Target identifications", fontsize=10)
    ax.set_title(f"{model_name} — IDs vs FDR", fontsize=11)
    ax.set_xlim(0, fdr_max * 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "ids_vs_fdr.png"))


# ---------------------------------------------------------------------------
# Subsystem 8: Protein colocalization by scoring group
# ---------------------------------------------------------------------------

_COLOC_COLS = [
    ("protein_colocalization",         "Protein coloc. (mean r)"),
    ("protein_colocalization_max",     "Protein coloc. (max r)"),
    ("protein_colocalization_median",  "Protein coloc. (median r)"),
    ("protein_colocalization_n_partners", "Protein coloc. (n partners)"),
]

_GROUP_ORDER  = ["ID @ 1% FDR", "ID @ 5% FDR", "R1 winner (below FDR)", "Non-winner"]
_GROUP_COLORS = ["seagreen",    "mediumseagreen", "darkorange",          "steelblue"]


def plot_protein_colocalization_by_group(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    fdr_threshold_loose: float = 0.05,
) -> None:
    """
    Box + strip plot of protein-level colocalization values split into four groups:
      - ID @ 1% FDR  : TDC winner with q_value <= fdr_threshold
      - ID @ 5% FDR  : TDC winner with fdr_threshold < q_value <= fdr_threshold_loose
      - R1 winner     : TDC winner, but q_value > fdr_threshold_loose
      - Non-winner    : lost its feature's target-decoy competition

    Only target (non-decoy) rows are shown, since decoy colocalization values
    reflect the null model rather than biology.

    Skips silently if none of the four colocalization columns are present.
    Output: ``{out_dir}/protein_colocalization_by_group.png``
    """
    present = [col for col, _ in _COLOC_COLS if col in features_df.columns]
    if not present:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_decoy = _flag(feat, "is_decoy")
    is_winner = _flag(res, "is_tdc_winner")
    q = _num(res, "q_value")

    passes_fdr_strict = is_winner & (q <= fdr_threshold)
    passes_fdr_loose  = is_winner & (q > fdr_threshold) & (q <= fdr_threshold_loose)
    r1_only           = is_winner & (q > fdr_threshold_loose)

    group_label = np.where(
        passes_fdr_strict, _GROUP_ORDER[0],
        np.where(passes_fdr_loose, _GROUP_ORDER[1],
        np.where(r1_only, _GROUP_ORDER[2], _GROUP_ORDER[3])),
    )

    # Restrict to targets only
    target_mask = ~is_decoy

    n_cols = len(present)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    rng = np.random.default_rng(0)

    for ax, col in zip(axes, present):
        label = dict(_COLOC_COLS)[col]
        vals  = pd.to_numeric(feat[col], errors="coerce").values.astype(float)

        group_data: dict[str, np.ndarray] = {}
        for grp in _GROUP_ORDER:
            mask = target_mask & (group_label == grp)
            v = vals[mask]
            group_data[grp] = v[np.isfinite(v)]

        positions = list(range(len(_GROUP_ORDER)))

        # Box plots
        bp_data = [group_data[g] for g in _GROUP_ORDER]
        bp = ax.boxplot(
            bp_data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.8),
            whiskerprops=dict(lw=1.0),
            capprops=dict(lw=1.0),
            boxprops=dict(lw=1.0),
        )
        for patch, color in zip(bp["boxes"], _GROUP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        # Strip (jitter) overlay
        for pos, (grp, color) in enumerate(zip(_GROUP_ORDER, _GROUP_COLORS)):
            v = group_data[grp]
            if len(v) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(v))
            ax.scatter(
                pos + jitter, v,
                s=12, alpha=0.55, color=color, linewidths=0, zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [g.replace(" ", "\n") for g in _GROUP_ORDER],
            fontsize=7,
        )
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

        # Horizontal reference line at r=0 for correlation columns
        if "n_partners" not in col:
            ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)

    # Sample size annotations — drawn after tight_layout sets final y limits.
    for ax, col in zip(axes, present):
        vals = pd.to_numeric(feat[col], errors="coerce").values.astype(float)
        y_lo, y_top = ax.get_ylim()
        y_ann = y_top + (y_top - y_lo) * 0.02
        for pos, grp in enumerate(_GROUP_ORDER):
            mask = target_mask & (group_label == grp)
            n = int(np.isfinite(vals[mask]).sum())
            ax.text(pos, y_ann, f"n={n}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"Protein colocalization by scoring group "
        f"(targets only, strict FDR {fdr_threshold:.0%} / loose FDR {fdr_threshold_loose:.0%})",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "protein_colocalization_by_group.png"))


# ---------------------------------------------------------------------------
# Subsystem 9: Target vs Decoy m/z distribution
# ---------------------------------------------------------------------------


def plot_target_decoy_mz_distribution(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    n_bins: int = 60,
) -> None:
    """
    Three-panel figure showing target vs decoy m/z coverage.

    Panel 1 — Per-feature competition status (histogram):
      Each MALDI feature is counted once:
        - steelblue  : target-only  (no decoy candidate matched this feature)
        - mediumpurple: contested   (at least one target AND one decoy)
        - tomato     : decoy-only   (no target candidate matched this feature)

    Panel 2 — All candidates, per-candidate density (target vs decoy):
      Overlapping density histograms with dashed median lines.

    Panel 3 — R1 winners only (best candidate per feature):
      Same layout as Panel 2 but restricted to is_tdc_winner rows.
      Shows the effective T:D ratio entering FDR estimation.

    Output: ``{out_dir}/target_decoy_mz_distribution.png``
    """
    if "feature_mz" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    is_decoy = _flag(feat, "is_decoy")
    is_winner = _flag(res, "is_tdc_winner")

    fmz = pd.to_numeric(feat["feature_mz"], errors="coerce").values
    finite = np.isfinite(fmz)

    # --- Per-feature competition classification ---
    target_fmz_set = set(fmz[~is_decoy & finite].tolist())
    decoy_fmz_set  = set(fmz[is_decoy  & finite].tolist())

    contested_arr   = np.array(sorted(target_fmz_set & decoy_fmz_set))
    target_only_arr = np.array(sorted(target_fmz_set - decoy_fmz_set))
    decoy_only_arr  = np.array(sorted(decoy_fmz_set  - target_fmz_set))
    all_fmz_arr     = np.concatenate([contested_arr, target_only_arr, decoy_only_arr])

    # Per-candidate m/z arrays
    t_mz     = fmz[~is_decoy & finite]
    d_mz     = fmz[ is_decoy & finite]
    t_win    = fmz[~is_decoy & is_winner & finite]
    d_win    = fmz[ is_decoy & is_winner & finite]

    if len(all_fmz_arr) == 0:
        return

    lo = float(np.percentile(all_fmz_arr, 1))
    hi = float(np.percentile(all_fmz_arr, 99))
    if lo >= hi:
        lo, hi = float(all_fmz_arr.min()), float(all_fmz_arr.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # ------------------------------------------------------------------
    # Panel 1: per-feature competition status
    # ------------------------------------------------------------------
    ax = axes[0]
    if len(target_only_arr):
        ax.hist(target_only_arr, bins=bins, alpha=0.70, color="steelblue",
                label=f"Target-only ({len(target_only_arr)})")
    if len(contested_arr):
        ax.hist(contested_arr, bins=bins, alpha=0.70, color="mediumpurple",
                label=f"Contested ({len(contested_arr)})")
    if len(decoy_only_arr):
        ax.hist(decoy_only_arr, bins=bins, alpha=0.70, color="tomato",
                label=f"Decoy-only ({len(decoy_only_arr)})")

    n_total = len(target_only_arr) + len(contested_arr) + len(decoy_only_arr)
    ax.set_title(
        f"Per-feature competition status  "
        f"({n_total} features: {len(target_only_arr)} T-only, "
        f"{len(contested_arr)} contested, {len(decoy_only_arr)} D-only)",
        fontsize=9,
    )
    ax.set_ylabel("Number of features", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Panel 2: all candidates; Panel 3: R1 winners
    # ------------------------------------------------------------------
    for ax, t_arr, d_arr, lbl, title in [
        (axes[1], t_mz, d_mz, "", "All candidates"),
        (axes[2], t_win, d_win, " R1 winners", "R1 winners (best per feature)"),
    ]:
        for arr, color, name in [(t_arr, "steelblue", "Target"), (d_arr, "tomato", "Decoy")]:
            if len(arr):
                ax.hist(arr, bins=bins, density=True, alpha=0.50, color=color,
                        label=f"{name}{lbl} (n={len(arr)})")
                ax.axvline(float(np.median(arr)), color=color, lw=1.5, ls="--", alpha=0.85)
        ratio_str = (
            f"T:D = {len(t_arr)}/{len(d_arr)} = {len(t_arr)/len(d_arr):.2f}:1"
            if len(d_arr) else f"T:D = {len(t_arr)}/0"
        )
        ax.set_title(f"{title}  ({ratio_str})", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    axes[2].set_xlabel("Feature m/z", fontsize=9)

    fig.suptitle("Target vs Decoy m/z Distributions", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "target_decoy_mz_distribution.png"))


# ---------------------------------------------------------------------------
# Subsystem 10: Candidate competition per feature
# ---------------------------------------------------------------------------


def plot_candidate_competition(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    ccs_tol_pct: float | None = None,
) -> None:
    """
    Four-panel figure showing how many target and decoy candidates compete at
    each MALDI m/z feature.

    Panel [0,0] — Target candidate count distribution:
        Histogram of how many features have 0, 1, 2, 3, 4+ target candidates.
    Panel [0,1] — Decoy candidate count distribution:
        Same for decoy candidates.
    Panel [1,0] — T vs D balance scatter:
        Each point = one (n_targets, n_decoys) combination; size ∝ number of
        features at that combination.  Diagonal marks perfect 1:1 balance.
    Panel [1,1] — Sorted competition landscape:
        Each feature as a vertical pair of bars: n_targets (blue, above axis)
        and n_decoys (orange, below axis), sorted by total candidates
        descending.  Capped at the 200 most-contested features for clarity.

    When ``ccs_tol_pct`` is not None the title notes that a CCS filter was
    applied before calling this function.

    Output: ``{out_dir}/candidate_competition.png``
    """
    if "feature_mz" not in features_df.columns or "is_decoy" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    is_decoy = _flag(feat, "is_decoy")

    feat_col = "feature_idx" if "feature_idx" in feat.columns else "feature_mz"

    # --- Per-feature target/decoy counts ---
    n_tgt = feat[~is_decoy].groupby(feat_col).size().rename("n_targets")
    n_dec = feat[is_decoy].groupby(feat_col).size().rename("n_decoys")
    all_features = feat[feat_col].unique()
    per_feat = (
        pd.DataFrame(index=all_features)
        .join(n_tgt, how="left")
        .join(n_dec, how="left")
        .fillna(0)
        .astype(int)
    )
    per_feat["n_total"] = per_feat["n_targets"] + per_feat["n_decoys"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Candidate competition per MALDI feature"
        + (f"  [CCS filter applied: ±{ccs_tol_pct:.1f}%]" if ccs_tol_pct is not None else ""),
        fontsize=11,
    )

    # [0,0] / [0,1]  Target / decoy candidate count distributions
    for ax, col, color, name, short in [
        (axes[0][0], "n_targets", "steelblue", "Target", "target"),
        (axes[0][1], "n_decoys", "tomato", "Decoy", "decoy"),
    ]:
        cap = min(int(per_feat[col].max()) if len(per_feat) else 4, 6)
        ax.hist(np.clip(per_feat[col].values, 0, cap), bins=np.arange(0, cap + 2) - 0.5,
                color=color, edgecolor="white", linewidth=0.5)
        ax.set_xticks(np.arange(0, cap + 1))
        ax.set_xticklabels([str(i) if i < cap else f"{cap}+" for i in range(cap + 1)])
        ax.set_xlabel(f"{name} candidates per feature")
        ax.set_ylabel("Number of features")
        ax.set_title(
            f"{name} candidate distribution\n"
            f"median={per_feat[col].median():.1f}  "
            f"mean={per_feat[col].mean():.2f}  "
            f"0-{short} features: {int((per_feat[col]==0).sum())}",
            fontsize=8,
        )

    # ------------------------------------------------------------------ #
    # [1,0]  T vs D balance scatter                                       #
    # ------------------------------------------------------------------ #
    ax = axes[1][0]
    td_counts = per_feat.groupby(["n_targets", "n_decoys"]).size().reset_index(name="count")
    sc = ax.scatter(
        td_counts["n_targets"],
        td_counts["n_decoys"],
        s=np.clip(td_counts["count"], 1, None) * 12,
        c=np.log1p(td_counts["count"]),
        cmap="Blues",
        edgecolors="steelblue",
        linewidths=0.6,
        alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="log(n features + 1)", fraction=0.046, pad=0.04)
    # Ideal 1:1 diagonal
    _lim = max(int(td_counts[["n_targets", "n_decoys"]].max().max()), 1)
    ax.plot([0, _lim], [0, _lim], color="gray", lw=0.8, ls="--", alpha=0.6, label="1:1")
    ax.set_xlim(left=-0.3)
    ax.set_ylim(bottom=-0.3)
    ax.set_xlabel("n_targets per feature")
    ax.set_ylabel("n_decoys per feature")
    ax.set_title("Target vs decoy balance per feature\n(bubble size ∝ n features)", fontsize=8)
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------ #
    # [1,1]  Sorted competition landscape (top-N most contested)          #
    # ------------------------------------------------------------------ #
    ax = axes[1][1]
    _TOP = 150
    sorted_pf = per_feat.sort_values("n_total", ascending=False).head(_TOP).reset_index(drop=True)
    x = np.arange(len(sorted_pf))
    ax.bar(x, sorted_pf["n_targets"].values, color="steelblue", label="Targets", width=1.0)
    ax.bar(x, -sorted_pf["n_decoys"].values, color="tomato", label="Decoys", width=1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(1, color="steelblue", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(-1, color="tomato", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel(f"Feature rank (by total candidates, top {min(_TOP, len(per_feat))} shown)")
    ax.set_ylabel("n_candidates  (targets ↑  /  decoys ↓)")
    _n_feat = len(per_feat)
    _td_ratio = per_feat["n_decoys"].sum() / max(per_feat["n_targets"].sum(), 1)
    ax.set_title(
        f"Competition landscape  ({_n_feat} features total)\n"
        f"total T={per_feat['n_targets'].sum()}  D={per_feat['n_decoys'].sum()}  "
        f"D:T ratio={_td_ratio:.2f}",
        fontsize=8,
    )
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "candidate_competition.png"))


# ---------------------------------------------------------------------------
# Subsystem 11: Score PP plot
# ---------------------------------------------------------------------------


def _pp_curve(
    target_scores: np.ndarray,
    decoy_scores: np.ndarray,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (F_decoy(t), F_target(t)) evaluated on a grid of n_points thresholds."""
    t_sorted = np.sort(target_scores)
    d_sorted = np.sort(decoy_scores)
    lo = float(min(t_sorted[0], d_sorted[0]))
    hi = float(max(t_sorted[-1], d_sorted[-1]))
    if lo >= hi:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    thresholds = np.linspace(lo, hi, n_points)
    x = np.searchsorted(d_sorted, thresholds, side="right") / len(d_sorted)
    y = np.searchsorted(t_sorted, thresholds, side="right") / len(t_sorted)
    # Prepend (0,0) so the curve starts at the origin
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    return x, y


def _draw_pp_panel(
    ax: "matplotlib.axes.Axes",
    target_scores: np.ndarray,
    decoy_scores: np.ndarray,
    title: str,
    n_points: int = 500,
) -> None:
    """Draw a single PP-plot panel onto ax."""
    n_t = len(target_scores)
    n_d = len(decoy_scores)
    if n_t == 0 or n_d == 0:
        ax.set_visible(False)
        return

    x, y = _pp_curve(target_scores, decoy_scores, n_points=n_points)

    # Reference line 1: diagonal y = x (what a fully null distribution would look like)
    ax.plot([0, 1], [0, 1], color="grey", lw=1.0, ls="--", label="y = x (null)")

    # Reference line 2: y = (1 − π₁) × x — expected slope in the null-dominated region.
    # π₁ estimated under the TDC assumption: decoys approximate null targets.
    pi1_hat = max(0.0, 1.0 - n_d / n_t)
    slope = 1.0 - pi1_hat
    ax.plot(
        [0, 1], [0, slope],
        color="darkorange", lw=1.2, ls=":",
        label=f"y = (1−π₁)·x  [π₁≈{pi1_hat:.2f}]",
    )

    ax.plot(x, y, color="steelblue", lw=1.8, label=f"T (n={n_t})  vs  D (n={n_d})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("F_decoy(t)", fontsize=9)
    ax.set_ylabel("F_target(t)", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")


def plot_score_pp(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    n_points: int = 500,
) -> None:
    """
    PP plot of score distributions: F_decoy(t) on the x-axis vs F_target(t)
    on the y-axis, sweeping threshold t across all observed scores.

    Two panels:
      Left  — ``*_score_r1`` on all candidates.
      Right — ``score`` on the winners (the TDC input set).

    Reference lines on each panel:
      - Dashed grey  y = x: the curve if targets and decoys were identically
        distributed (pure null, no true positives).
      - Dotted orange y = (1−π₁)·x: expected null-component slope in the
        bottom-left region, where π₁ = max(0, 1 − n_decoy / n_target) is
        the fraction of estimated true positives among targets.

    A healthy TDC run: the curve starts at (0, 0), tracks the orange dotted
    line in the low-score region (where both distributions are dominated by
    incorrect matches), then peels away toward (1, 1) as the target CDF
    accumulates true positives in the high-score tail.

    Output: ``{out_dir}/score_pp_plot.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    assert len(feat) == len(res), f"Length mismatch: {len(feat)} vs {len(res)}"

    is_decoy = _flag(feat, "is_decoy")
    is_winner = _flag(res, "is_tdc_winner")

    # Detect score columns dynamically
    r1_col = next((c for c in res.columns if c.endswith("_score_r1")), None)
    w_col = "score" if "score" in res.columns else None
    if r1_col is None and w_col is None:
        logger.debug("score_pp: no score columns found, skipping")
        return

    panels = [
        (r1_col, np.ones(len(res), dtype=bool), f"All candidates — {r1_col}"),
        (w_col, is_winner, "Winners — score"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (col, subset_mask, title) in zip(axes, panels):
        if col is None:
            ax.set_visible(False)
            continue
        scores = _num(res, col)
        m = subset_mask & np.isfinite(scores)
        _draw_pp_panel(ax, scores[~is_decoy & m], scores[is_decoy & m], title, n_points=n_points)

    fig.suptitle("Score PP plot: target vs decoy empirical CDFs", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "score_pp_plot.png"))


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    n_bins: int = 60,
) -> None:
    """
    Overlapping target/decoy score histograms.

    Up to two panels (left to right):
      ``*_score_r1`` — all candidates (including non-winners).
      ``score``      — winners only (non-winners have NaN ``score``).

    Each panel uses normalised counts (density=True) so target and decoy
    distributions are comparable when their sizes differ. A vertical dashed
    line marks the score threshold corresponding to q_value ≤ 0.01 on winners (if
    computable). Panels with no finite scores are hidden.

    Output: ``{out_dir}/score_distributions.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_decoy = _flag(feat, "is_decoy")
    is_winner = _flag(res, "is_tdc_winner")
    _src_sd = feat.get("source", pd.Series("", index=feat.index)).fillna("").values
    entrapment_mask_sd = (~is_decoy) & (_src_sd == "entrapment_shuffled")
    has_entrapment_sd = entrapment_mask_sd.any()
    real_target_mask_sd = (~is_decoy) & (~entrapment_mask_sd)

    r1_cols = [c for c in res.columns if c.endswith("_score_r1")]
    r1_col = r1_cols[0] if r1_cols else None
    w_col = "score" if "score" in res.columns else None
    panels = [
        (_num(res, col), mask, col, label)
        for col, mask, label in [
            (r1_col, np.ones(len(res), dtype=bool), "All candidates"),
            (w_col, is_winner, "Winners"),
        ]
        if col
    ]

    if not panels:
        logger.debug("score_distributions: no score columns found, skipping")
        return

    # Q=0.01 threshold from the winners
    q_threshold_score = None
    if w_col and "q_value" in res.columns:
        w_scores = _num(res, w_col)
        q_vals = _num(res, "q_value")
        mask = is_winner & real_target_mask_sd & np.isfinite(w_scores) & np.isfinite(q_vals)
        if mask.any():
            passing = w_scores[mask & (q_vals <= 0.01)]
            if len(passing):
                q_threshold_score = passing.min()

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    axes = axes[0]

    colours = {"T": "#2196F3", "D": "#F44336", "E": "goldenrod"}

    for ax, (scores, subset_mask, col_label, subset_label) in zip(axes, panels):
        t_scores = scores[real_target_mask_sd & subset_mask]
        d_scores = scores[is_decoy & subset_mask]
        e_scores = scores[entrapment_mask_sd & subset_mask] if has_entrapment_sd else np.array([])
        t_finite = t_scores[np.isfinite(t_scores)]
        d_finite = d_scores[np.isfinite(d_scores)]
        e_finite = e_scores[np.isfinite(e_scores)] if len(e_scores) else np.array([])

        if not len(t_finite) and not len(d_finite):
            ax.set_visible(False)
            continue

        all_finite = np.concatenate([t_finite, d_finite] + ([e_finite] if len(e_finite) else []))

        # IQR-based x-axis limits: robust to heavy-tailed and skewed distributions.
        # Whisker = Q1 - 3*IQR … Q3 + 3*IQR, then clipped to data range.
        q1, q3 = np.percentile(all_finite, [25, 75])
        iqr = q3 - q1
        lo = max(all_finite.min(), q1 - 3.0 * iqr)
        hi = min(all_finite.max(), q3 + 3.0 * iqr)
        if lo >= hi:
            lo, hi = all_finite.min(), all_finite.max()
        bins = np.linspace(lo, hi, n_bins + 1)

        classes = [(t_finite, "T", "Target", "-"), (d_finite, "D", "Decoy", "-"),
                   (e_finite, "E", "Entrapment", "--")]
        for vals, key, name, _ in classes:
            if len(vals):
                ax.hist(vals, bins=bins, density=True, color=colours[key], alpha=0.45,
                        label=f"{name} (n={len(vals):,})")

        # KDE overlay for clearer shape visualization.
        try:
            from scipy.stats import gaussian_kde
            x_kde = np.linspace(lo, hi, 300)
            for vals, key, _, ls in classes:
                if len(vals) >= 5:
                    ax.plot(x_kde, gaussian_kde(vals)(x_kde), color=colours[key], lw=1.5, linestyle=ls)
        except Exception:
            pass

        if q_threshold_score is not None and col_label == w_col:
            ax.axvline(
                q_threshold_score, color="black", linestyle="--", linewidth=1.0,
                label=f"q≤0.01 ({q_threshold_score:.3f})",
            )

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_xlim(lo, hi)
        ax.set_title(f"{col_label}\n({subset_label})", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Score distributions — target vs decoy", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "score_distributions.png"))


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _make_gt_subset(
    gt_peptides: list[str],
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
) -> tuple["pd.DataFrame | None", list[str]]:
    """
    Build a subset DataFrame for GT peptides, parallel to _sample_subset output.

    Returns (gt_subset, not_found) where not_found lists GT peptides absent from
    features_df entirely.
    """
    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    peptide_vals = feat.get("peptide", pd.Series(dtype=str)).values
    gt_set = set(gt_peptides)
    not_found = [p for p in gt_peptides if p not in peptide_vals]

    matched_idx = [i for i, p in enumerate(peptide_vals) if p in gt_set]
    if not matched_idx:
        return None, list(gt_set)

    return _annotate_subset(feat, res, matched_idx, "GT"), not_found


def _save_gt_not_found_figures(peptides: list[str], subdirs: list[str]) -> None:
    """Save a 'not a candidate' placeholder figure for each unfound GT peptide."""
    for sub in subdirs:
        os.makedirs(sub, exist_ok=True)
        for pep in peptides:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(
                0.5, 0.5,
                f"GT peptide '{pep}' is not a candidate",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            ax.axis("off")
            fig.suptitle(f"GT: {pep}", fontsize=10)
            fname = f"GT_T_000_{_safe_fname(pep)}.png"
            _save_and_close(fig, os.path.join(sub, fname), dpi=80)


# ---------------------------------------------------------------------------
# PEP mixture model visualization
# ---------------------------------------------------------------------------

def plot_pep_mixture(
    result_df: pd.DataFrame,
    out_dir: str,
    model_name: str = "model",
    n_bins: int = 50,
) -> None:
    """
    Overlay histogram of target and decoy winner scores with the Gaussian f0/f1
    curves ``estimate_pep`` fits, and a secondary y-axis PEP curve.

    X-axis uses IQR-based limits (Q1 − 3×IQR … Q3 + 3×IQR) to handle
    heavy-tailed score distributions robustly.

    Reads ``pep`` and ``score`` from ``result_df`` (winners only).
    Output: ``{out_dir}/pep_mixture.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    winners = result_df[_flag(result_df, "is_tdc_winner")].copy()
    if len(winners) == 0:
        logger.debug("plot_pep_mixture: no winners, skipping")
        return

    if "score" not in winners.columns or "pep" not in winners.columns:
        logger.debug("plot_pep_mixture: score or pep column missing, skipping")
        return

    is_decoy_w = _flag(winners, "is_decoy")
    scores = pd.to_numeric(winners["score"], errors="coerce").values
    pep_vals = pd.to_numeric(winners["pep"], errors="coerce").values

    finite = np.isfinite(scores) & np.isfinite(pep_vals)
    if finite.sum() < 4:
        logger.debug("plot_pep_mixture: too few finite winners, skipping")
        return

    scores_f = scores[finite]
    pep_f = pep_vals[finite]
    is_decoy_f = is_decoy_w[finite]

    t_scores = scores_f[~is_decoy_f]
    d_scores = scores_f[is_decoy_f]

    # IQR-based x limits: robust to heavy-tailed scores.
    q1, q3 = np.percentile(scores_f, [25, 75])
    iqr = q3 - q1
    lo = max(float(scores_f.min()), q1 - 3.0 * iqr)
    hi = min(float(scores_f.max()), q3 + 3.0 * iqr)
    if lo >= hi:
        lo, hi = float(scores_f.min()) - 0.5, float(scores_f.max()) + 0.5
    score_range = np.linspace(lo, hi, 300)

    # Shared setup for both methods (mirrors estimate_pep)
    median_t = float(np.median(t_scores)) if len(t_scores) >= 2 else float(np.mean(scores_f))
    high_t = t_scores[t_scores > median_t] if len(t_scores) >= 2 else t_scores
    if len(high_t) < 2:
        high_t = t_scores
    pi0 = is_decoy_f.sum() / len(is_decoy_f)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bins = np.linspace(lo, hi, n_bins + 1)
    ax1.hist(t_scores, bins=bins, alpha=0.4, color="steelblue", label="Targets", density=True)
    ax1.hist(d_scores, bins=bins, alpha=0.4, color="tomato", label="Decoys", density=True)

    # Gaussian mixture: reconstruct f0/f1 and PEP curve analytically.
    from scipy.stats import norm
    mu0 = float(np.mean(d_scores)) if len(d_scores) >= 2 else float(np.mean(scores_f))
    sigma0 = max(float(np.std(d_scores)), 1e-6) if len(d_scores) >= 2 else 1.0
    mu1 = float(np.mean(high_t)) if len(high_t) >= 1 else mu0 + 1.0
    sigma1 = max(float(np.std(high_t)), 1e-6) if len(high_t) >= 2 else 1.0
    f0_curve = norm.pdf(score_range, mu0, sigma0)
    f1_curve = norm.pdf(score_range, mu1, sigma1)
    numer = pi0 * f0_curve
    denom = numer + (1.0 - pi0) * f1_curve
    with np.errstate(invalid="ignore", divide="ignore"):
        pep_curve = np.where(denom > 0, numer / denom, 1.0)

    ax1.plot(score_range, f1_curve, color="steelblue", lw=1.5, ls="--", label="f1 (signal)")
    ax1.plot(score_range, f0_curve, color="tomato", lw=1.5, ls="--", label="f0 (null)")

    ax2 = ax1.twinx()
    ax2.plot(score_range, pep_curve, color="black", lw=2, label="PEP")
    sort_idx = np.argsort(scores_f)
    ax2.scatter(scores_f[sort_idx], pep_f[sort_idx],
                s=6, color="grey", alpha=0.4, zorder=3)
    title_suffix = "Gaussian mixture"

    ax1.set_xlabel("score")
    ax1.set_ylabel("Density")
    ax1.set_xlim(lo, hi)

    ax2.set_ylabel("PEP")
    ax2.set_ylim(-0.05, 1.15)
    ax2.axhline(0.05, color="black", lw=0.8, ls=":", alpha=0.6)
    ax2.axhline(0.20, color="black", lw=0.8, ls=":", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    ax1.set_title(
        f"PEP — {model_name} winners (n={len(scores_f)}) [{title_suffix}]"
    )
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "pep_mixture.png"))


# ---------------------------------------------------------------------------
# Subsystem 12: Ion image Pearson r distribution
# ---------------------------------------------------------------------------

def plot_ion_image_pearson_distribution(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    fdr_threshold: float = 0.05,
) -> None:
    """
    Distribution of pairwise ion image Pearson r for same-protein vs different-protein
    peptide pairs, restricted to target IDs at FDR (``q_value``) <= fdr_threshold.

    Output: ``{out_dir}/ion_image_pearson_distribution.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_winner = _flag(res, "is_tdc_winner")
    q = _num(res, "q_value")
    is_decoy = _flag(feat, "is_decoy")

    id_mask = is_winner & (q <= fdr_threshold) & ~is_decoy
    id_idx = np.where(id_mask)[0]
    if len(id_idx) < 2:
        logger.debug(
            "plot_ion_image_pearson_distribution: fewer than 2 target IDs at FDR %.2f, skipping",
            fdr_threshold,
        )
        return

    id_mzs = (
        feat["feature_mz"].values[id_idx]
        if "feature_mz" in feat.columns
        else np.full(len(id_idx), float("nan"))
    )
    id_proteins = (
        feat["protein"].fillna("unknown").values[id_idx]
        if "protein" in feat.columns
        else np.full(len(id_idx), "unknown")
    )

    flat_imgs: list[np.ndarray] = []
    valid_pos: list[int] = []
    for pos, mz in enumerate(id_mzs):
        if not np.isfinite(float(mz)):
            continue
        img_idx = _find_image_idx(float(mz), ion_image_mzs)
        if img_idx is None:
            continue
        flat_imgs.append(ion_images[img_idx].ravel().astype(float))
        valid_pos.append(pos)

    if len(valid_pos) < 2:
        return

    n = len(valid_pos)
    valid_proteins = np.array([id_proteins[p] for p in valid_pos])
    r, iu, ju = _pairwise_r(np.array(flat_imgs))
    same = valid_proteins[iu] == valid_proteins[ju]
    same_r = r[same].tolist()
    diff_r = r[~same].tolist()

    if not same_r and not diff_r:
        return

    rng = np.random.default_rng(42)
    max_bg = 10_000
    if len(diff_r) > max_bg:
        diff_r = rng.choice(diff_r, max_bg, replace=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 61)
    if diff_r:
        ax.hist(
            diff_r, bins=bins, alpha=0.55, color="steelblue", density=True,
            label=f"Different protein ({len(diff_r):,} pairs)",
        )
    if same_r:
        ax.hist(
            same_r, bins=bins, alpha=0.70, color="tomato", density=True,
            label=f"Same protein ({len(same_r):,} pairs)",
        )
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Pearson r of ion images")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Ion image colocalization — peptide pairs at ≤{fdr_threshold:.0%} FDR\n"
        f"(n={n} target IDs)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "ion_image_pearson_distribution.png"))


# ---------------------------------------------------------------------------
# Subsystem 13: Protein spatial coherence scatter
# ---------------------------------------------------------------------------

def plot_protein_spatial_coherence(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    fdr_threshold: float = 0.05,
) -> None:
    """
    Per-protein scatter: number of unique peptides at FDR (``q_value``) <= fdr_threshold
    (x-axis) vs mean pairwise ion image Pearson r (y-axis).  Singletons are shown at
    y = -0.15 with jitter.  Target and decoy proteins are coloured separately.

    Output: ``{out_dir}/protein_spatial_coherence.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_winner = _flag(res, "is_tdc_winner")
    q = _num(res, "q_value")
    is_decoy = _flag(feat, "is_decoy")

    id_mask = is_winner & (q <= fdr_threshold)
    id_idx = np.where(id_mask)[0]
    if len(id_idx) < 2:
        return

    id_mzs   = (
        feat["feature_mz"].values[id_idx]
        if "feature_mz" in feat.columns
        else np.full(len(id_idx), float("nan"))
    )
    id_prots  = (
        feat["protein"].fillna("unknown").values[id_idx]
        if "protein" in feat.columns
        else np.full(len(id_idx), "unknown")
    )
    id_decoys = is_decoy[id_idx]

    prot_imgs: dict[str, list[np.ndarray]] = {}
    prot_is_decoy: dict[str, bool] = {}
    for mz, prot, dec in zip(id_mzs, id_prots, id_decoys):
        if not np.isfinite(float(mz)):
            continue
        img_idx = _find_image_idx(float(mz), ion_image_mzs)
        if img_idx is None:
            continue
        img = ion_images[img_idx].ravel().astype(float)
        if prot not in prot_imgs:
            prot_imgs[prot] = []
            prot_is_decoy[prot] = bool(dec)
        prot_imgs[prot].append(img)
        if not bool(dec):
            prot_is_decoy[prot] = False  # target evidence takes priority

    rows = []
    for prot, imgs in prot_imgs.items():
        n_pep = len(imgs)
        if n_pep == 1:
            mean_r = float("nan")
        else:
            rs = _pairwise_r(np.vstack(imgs))[0]
            rs = rs[np.isfinite(rs)]
            mean_r = float(np.mean(rs)) if rs.size else float("nan")
        rows.append({
            "protein": prot,
            "n_peptides": n_pep,
            "mean_r": mean_r,
            "is_decoy": prot_is_decoy[prot],
        })

    if not rows:
        return

    df_p  = pd.DataFrame(rows)
    multi  = df_p[df_p["n_peptides"] > 1]
    single = df_p[df_p["n_peptides"] == 1]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 5))

    for dec, color, lbl in [(False, "tomato", "target"), (True, "steelblue", "decoy")]:
        sub = multi[multi["is_decoy"] == dec]
        if not sub.empty:
            ax.scatter(
                sub["n_peptides"], sub["mean_r"],
                c=color, alpha=0.75, s=50, zorder=3,
                label=f"Multi-peptide {lbl} (n={len(sub)})",
            )

    for dec, color, lbl in [(False, "tomato", "target"), (True, "steelblue", "decoy")]:
        sub = single[single["is_decoy"] == dec]
        if not sub.empty:
            jx = rng.uniform(-0.08, 0.08, len(sub))
            jy = rng.uniform(-0.02, 0.02, len(sub))
            ax.scatter(
                sub["n_peptides"].values + jx, -0.15 + jy,
                c=color, alpha=0.45, s=25, marker="x", zorder=2,
                label=f"Singleton {lbl} (n={len(sub)})",
            )

    if not multi.empty:
        top_n = multi.nlargest(min(10, len(multi)), "n_peptides")
        for _, r in top_n.iterrows():
            short = str(r["protein"]).split("|")[-1][:20]
            ax.annotate(
                short, (r["n_peptides"], r["mean_r"]),
                textcoords="offset points", xytext=(5, 3),
                fontsize=6, alpha=0.8,
            )

    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Unique peptides at FDR threshold")
    ax.set_ylabel("Mean pairwise ion image Pearson r")
    ax.set_title(
        f"Protein spatial coherence at ≤{fdr_threshold:.0%} FDR\n"
        f"(n={len(df_p)} proteins; singletons shown at y=−0.15)"
    )
    ax.set_xlim(left=0.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "protein_spatial_coherence.png"))


# ---------------------------------------------------------------------------
# Subsystem 15: Per-candidate SHAP explanations (LinearExplainer)
# ---------------------------------------------------------------------------

def debug_pfm_explanations(
    result_df: pd.DataFrame,
    X: np.ndarray,
    svm_pipeline,
    feature_names: list[str],
    ion_images: np.ndarray | None,
    feature_mzs: np.ndarray | None,
    output_dir: str,
    n_decoys: int = 10,
    fdr_threshold: float | None = None,
    max_targets: int = 200,
    kernel_background: int = 25,
    kernel_nsamples: int = 512,
) -> None:
    """
    Per-PFM SHAP explanation figures for the rescoring model.

    For a set of selected peptide-feature matches (PFMs) — target winners passing
    FDR plus a random sample of decoy winners — this computes SHAP values and saves
    a three-panel figure per candidate plus a summary TSV.

    The explainer follows the estimator. With ``coef_`` (lda, svm) it is
    ``shap.LinearExplainer`` (``feature_perturbation="interventional"``) on the bare
    coefficients, which is exact. Without it (rbf_svm, whose decision function lives
    in kernel space) it is ``shap.KernelExplainer`` on the estimator's
    ``decision_function``, with a ``kernel_background``-row k-means summary of the
    training matrix and ``kernel_nsamples`` coalitions per candidate. That is
    affordable only because it runs on the selected candidates rather than on every
    candidate row, which is the whole point of restricting it (H-model-2).

    ``result_df`` must be aligned row-for-row with ``X`` (the raw, pre-pipeline
    feature matrix the model was trained on, one row per winner).  ``feature_names``
    names the columns of ``X``.  ``svm_pipeline`` is the fitted sklearn ``Pipeline``
    (imputer → scaler → [poly] → linear estimator).

    Selection
    ---------
    The reported population, which is peptide-level since F-029: where
    ``is_peptide_winner``/``peptide_q_value`` are present they are used, and the
    per-feature ``is_tdc_winner``/``q_value`` only otherwise. Explaining feature
    winners while the run reports peptides would explain the same peptide several
    times over and misstate how many identifications were covered.

    Targets: winners with q <= ``fdr_threshold`` (default 0.01), falling back to
    0.05 when fewer than one target passes at 1%, best q first, at most
    ``max_targets``.  Decoys: ``n_decoys`` random winners with ``is_decoy=True``
    (``random.seed(42)``).

    Outputs (``<output_dir>/pfm_explanations/``)
    --------------------------------------------
    One PNG per candidate, ``{rank:03d}_{peptide}_{feature_mz:.4f}_{target|decoy}.png``
    (rank by ``q_value`` for targets, sampling order for decoys), each with:
      Left   — the candidate feature's ion image (``hot`` colormap, gamma 0.5).
      Middle — SHAP waterfall: top-15 per-feature contributions sorted by |SHAP|,
               positive (toward target) in steelblue, negative in tomato, with the
               base value and final score annotated.
      Right  — percentile rank of each top-15 feature value within the training
               distribution (0–100 horizontal bar with a marker).
    Plus ``summary.tsv`` with one row per explained candidate.
    """
    import random

    try:
        import shap
    except ImportError:
        logger.warning(
            "debug_pfm_explanations: the 'shap' package is not installed — "
            "skipping PFM SHAP explanations (pip install shap)."
        )
        return

    out_dir      = os.path.join(output_dir, "pfm_explanations")
    shap_data_dir = os.path.join(out_dir, "shap_data")
    os.makedirs(out_dir,       exist_ok=True)
    os.makedirs(shap_data_dir, exist_ok=True)

    res = result_df.reset_index(drop=True)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] != len(res):
        logger.warning(
            "debug_pfm_explanations: X has %d rows but result_df has %d — "
            "cannot align; skipping.",
            X.shape[0], len(res),
        )
        return

    # The reported population — peptide-level where the run computed it (F-029),
    # per-feature otherwise.
    winner_col, q_col = "is_peptide_winner", "peptide_q_value"
    if winner_col not in res.columns or q_col not in res.columns:
        winner_col, q_col = "is_tdc_winner", "q_value"
    if winner_col in res.columns:
        winner_mask = res[winner_col].fillna(False).astype(bool).values
    else:
        winner_mask = np.ones(len(res), dtype=bool)
    is_decoy = _flag(res, "is_decoy")
    q_value = _num(res, q_col)

    # --- Select targets at FDR (fall back 1% → 5%) ---
    if fdr_threshold is None:
        thr = 0.01
        target_pos = np.where(winner_mask & ~is_decoy & (q_value <= thr))[0]
        if len(target_pos) < 1:
            thr = 0.05
            target_pos = np.where(winner_mask & ~is_decoy & (q_value <= thr))[0]
    else:
        thr = float(fdr_threshold)
        target_pos = np.where(winner_mask & ~is_decoy & (q_value <= thr))[0]
    # Rank targets by q_value ascending; the cap bounds the KernelExplainer cost.
    target_pos = target_pos[np.argsort(q_value[target_pos], kind="stable")][:max_targets]

    # --- Sample decoy winners ---
    decoy_candidates = np.where(winner_mask & is_decoy)[0].tolist()
    random.seed(42)
    if len(decoy_candidates) > n_decoys:
        decoy_pos = sorted(random.sample(decoy_candidates, n_decoys))
    else:
        decoy_pos = decoy_candidates

    logger.info(
        "debug_pfm_explanations: explaining %d targets (q<=%.2g) and %d decoys",
        len(target_pos), thr, len(decoy_pos),
    )
    if len(target_pos) == 0 and len(decoy_pos) == 0:
        logger.warning("debug_pfm_explanations: no candidates to explain — skipping.")
        return

    # --- Pipeline decomposition: pre-processing (imputer/scaler/[poly]) + estimator ---
    try:
        pre = svm_pipeline[:-1]
        estimator = svm_pipeline[-1]
        Xt_all = np.asarray(pre.transform(X), dtype=np.float64)
    except Exception as exc:
        logger.warning(
            "debug_pfm_explanations: could not decompose pipeline / estimator (%s) — skipping.",
            exc,
        )
        return
    is_linear = hasattr(estimator, "coef_")
    if is_linear:
        coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
        intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    else:
        coef = np.array([])

    # The transformed matrix may have more columns than feature_names when a
    # polynomial-interaction step expands the inputs; in that case raw-value and
    # percentile annotations fall back to the transformed feature space.
    if Xt_all.shape[1] == len(feature_names):
        est_names = list(feature_names)
        raw_aligned = True
    else:
        est_names = [f"f{j}" for j in range(Xt_all.shape[1])]
        raw_aligned = False
        logger.info(
            "debug_pfm_explanations: transformed matrix has %d columns vs %d raw "
            "feature names (poly expansion?) — using transformed values for labels.",
            Xt_all.shape[1], len(feature_names),
        )

    # --- SHAP explainer ---
    # Linear estimators: LinearExplainer on the bare coefficients,
    # feature_perturbation="interventional" with the full transformed training
    # matrix as background. Exact and effectively free.
    #
    # Kernel estimators (rbf_svm) have no coef_, so this used to skip entirely.
    # KernelExplainer covers them, and is affordable here for one reason: it runs
    # on the selected candidates only -- the few hundred that get reported -- not
    # on the ~12 K candidate rows (H-model-2). It is still ~n_background *
    # kernel_nsamples decision_function evaluations per candidate, so the
    # background is a k-means summary of the training matrix rather than all of
    # it, which is what shap's own documentation recommends for this explainer.
    try:
        if is_linear:
            explainer = shap.LinearExplainer(
                (coef, intercept), Xt_all, feature_perturbation="interventional"
            )
            base_value = float(np.asarray(explainer.expected_value).ravel()[0])

            def _shap_row(xt):
                return np.asarray(explainer.shap_values(xt)).reshape(-1)
        else:
            bg = shap.kmeans(Xt_all, min(kernel_background, Xt_all.shape[0]))
            explainer = shap.KernelExplainer(estimator.decision_function, bg)
            base_value = float(np.asarray(explainer.expected_value).ravel()[0])
            logger.info(
                "debug_pfm_explanations: %s has no coef_ — using KernelExplainer "
                "(%d background rows, %d samples per candidate) on %d candidates",
                type(estimator).__name__, kernel_background, kernel_nsamples,
                len(target_pos) + len(decoy_pos),
            )

            def _shap_row(xt):
                return np.asarray(
                    explainer.shap_values(xt, nsamples=kernel_nsamples, silent=True)
                ).reshape(-1)
    except Exception as exc:
        logger.warning("debug_pfm_explanations: SHAP explainer failed (%s) — skipping.", exc)
        return

    # Per-column training distributions for percentile ranks (raw space when aligned).
    dist_matrix = X if raw_aligned else Xt_all
    n_train = dist_matrix.shape[0]

    summary_rows: list[dict] = []
    top_k = 15

    def _explain_one(pos: int, rank: int, kind: str) -> None:
        row = res.iloc[pos]
        peptide = str(row.get("peptide", "unknown"))
        protein = str(row.get("protein", ""))
        feature_mz = _get(row, "feature_mz")
        qv = _get(row, q_col)

        xt = Xt_all[pos : pos + 1]
        shap_vals = _shap_row(xt)
        final_score = base_value + float(shap_vals.sum())

        order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        sel_names = [est_names[j] for j in order]
        sel_shap = shap_vals[order]
        if raw_aligned:
            sel_raw = X[pos, order]
        else:
            sel_raw = Xt_all[pos, order]

        # ----- Figure -----
        _BG      = "#F7F7F7"
        _POS_COL = "#4C9BE8"   # steel blue — toward target
        _NEG_COL = "#E8654C"   # coral     — away from target

        fig, (ax_img, ax_shap, ax_pct) = plt.subplots(
            1, 3, figsize=(16, 6),
            gridspec_kw={"width_ratios": [1.0, 1.8, 0.9]},
            facecolor=_BG,
        )
        fig.patch.set_facecolor(_BG)

        # Left: ion image (gamma 0.5, hot)
        img2d = None
        if ion_images is not None and feature_mzs is not None and np.isfinite(feature_mz):
            idx = _find_image_idx(float(feature_mz), feature_mzs)
            if idx is not None:
                img2d = np.asarray(ion_images[idx])
        ax_img.set_facecolor("black")
        for _sp in ax_img.spines.values():
            _sp.set_visible(False)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        if img2d is not None:
            _p99 = np.percentile(img2d[img2d > 0], 99) if (img2d > 0).any() else 1.0
            _imd = np.clip(img2d / _p99, 0, 1) ** 0.5
            _im  = ax_img.imshow(_imd, cmap="hot", vmin=0, vmax=1, aspect="auto",
                                 interpolation="nearest")
            _cax = ax_img.inset_axes([0.02, 0.02, 0.06, 0.35])
            _cb  = fig.colorbar(_im, cax=_cax)
            _cb.set_ticks([0, 1]); _cb.set_ticklabels(["0", "p99"], fontsize=6, color="white")
            _cb.outline.set_edgecolor("white")
            _cb.ax.tick_params(colors="white", length=2)
            ax_img.set_title(f"{peptide}\n{feature_mz:.4f} Da", fontsize=9,
                             fontweight="bold", color="#222222", pad=4)
        else:
            ax_img.text(0.5, 0.5, "No ion image", ha="center", va="center",
                        transform=ax_img.transAxes, color="gray", fontsize=9)
            ax_img.set_title(f"{peptide}", fontsize=9, fontweight="bold",
                             color="#222222", pad=4)

        # Middle: SHAP waterfall (top 15 by |SHAP|)
        ypos   = np.arange(len(order))[::-1]  # largest |SHAP| at top
        colors = [_POS_COL if v >= 0 else _NEG_COL for v in sel_shap]
        ax_shap.set_facecolor("white")
        ax_shap.barh(ypos, sel_shap, color=colors, height=0.65,
                     edgecolor="white", linewidth=0.4, zorder=3)
        ax_shap.axvline(0, color="#333333", lw=1.0, zorder=4)
        ax_shap.axvline(base_value,  color="#888888", lw=0.8, ls="--", zorder=2)
        ax_shap.axvline(final_score, color="#222222", lw=1.2, ls=":",  zorder=2)
        ax_shap.set_yticks(ypos)
        ax_shap.set_yticklabels(
            [n.replace("_", " ") for n in sel_names],
            fontsize=7.5,
        )
        ax_shap.set_xlabel("SHAP contribution  (→ target)", fontsize=8, color="#444444")
        _xlim = (
            min(sel_shap.min() - 0.3, base_value  - 0.3),
            max(sel_shap.max() + 0.3, final_score + 0.3),
        )
        ax_shap.set_xlim(*_xlim)
        ax_shap.text(base_value,  1.01, f"base\n{base_value:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color="#888888",
                     transform=ax_shap.get_xaxis_transform())
        ax_shap.text(final_score, 1.01, f"score\n{final_score:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color="#222222", fontweight="bold",
                     transform=ax_shap.get_xaxis_transform())
        ax_shap.tick_params(axis="x", labelsize=7, colors="#555555")
        ax_shap.tick_params(axis="y", colors="#222222", length=0)
        ax_shap.xaxis.grid(True, color="#dddddd", lw=0.5, zorder=0)
        ax_shap.set_axisbelow(True)
        for _sp in ["top", "right", "left"]: ax_shap.spines[_sp].set_visible(False)
        ax_shap.spines["bottom"].set_color("#cccccc")

        # Right: percentile rank within training distribution
        pct = np.full(len(order), np.nan)
        for i, j in enumerate(order):
            colvals = dist_matrix[:, j]
            finite  = colvals[np.isfinite(colvals)]
            v = sel_raw[i]
            if finite.size > 0 and np.isfinite(v):
                pct[i] = 100.0 * np.count_nonzero(finite <= v) / finite.size
        ax_pct.set_facecolor("white")
        ax_pct.barh(ypos, np.full(len(order), 100.0), color="#eeeeee", height=0.65, zorder=1)
        ax_pct.barh(ypos, np.nan_to_num(pct),         color="#cccccc", height=0.65, zorder=2)
        for yp, p, col in zip(ypos, pct, colors):
            if np.isfinite(p):
                ax_pct.plot(p, yp, "o", color=col, ms=7, zorder=5,
                            markeredgecolor="white", markeredgewidth=0.5)
                _offset, _ha = (-3, "right") if p > 88 else (2, "left")
                ax_pct.text(p + _offset, yp, f"{p:.0f}", va="center",
                            fontsize=6.5, color="#333333", ha=_ha)
        ax_pct.set_xlim(0, 100)
        ax_pct.set_ylim(ax_shap.get_ylim())
        ax_pct.set_yticks([])
        ax_pct.set_xticks([0, 25, 50, 75, 100])
        ax_pct.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.5, color="#555555")
        ax_pct.set_xlabel("Percentile in training dist.", fontsize=8, color="#444444")
        ax_pct.tick_params(axis="x", length=2)
        for _sp in ["top", "right", "left"]: ax_pct.spines[_sp].set_visible(False)
        ax_pct.spines["bottom"].set_color("#cccccc")
        ax_pct.xaxis.grid(True, color="#eeeeee", lw=0.5, zorder=0)
        ax_pct.set_axisbelow(True)

        # Save per-candidate SHAP data for later reproduction
        _sel_coef = coef[order] if len(coef) == len(est_names) else np.full(len(order), np.nan)
        pd.DataFrame({
            "feature":        sel_names,
            "shap_value":     sel_shap,
            "raw_value":      sel_raw,
            "percentile_rank": pct,
            "coef":           _sel_coef,
        }).assign(
            peptide=peptide, protein=protein,
            feature_mz=feature_mz, q_value=qv,
            final_score=final_score, base_value=base_value, kind=kind,
        ).to_csv(
            os.path.join(
                shap_data_dir,
                f"{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}_{kind}.tsv"
                if np.isfinite(feature_mz)
                else f"{rank:03d}_{_safe_fname(peptide)}_{kind}.tsv",
            ),
            sep="\t", index=False,
        )

        _q_s = f"q = {qv:.4f}" if np.isfinite(qv) else "q = NA"
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        fig.text(
            0.5, 0.97,
            f"[{kind}]  {peptide}  ·  {protein}  ·  m/z {feature_mz:.4f}"
            f"  ·  {_q_s}  ·  score = {final_score:.3f}",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#111111",
        )
        fname = (
            f"{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}_{kind}.png"
            if np.isfinite(feature_mz)
            else f"{rank:03d}_{_safe_fname(peptide)}_{kind}.png"
        )
        _save_and_close(fig, os.path.join(out_dir, fname), dpi=150)

        srow = {
            "peptide": peptide,
            "protein": protein,
            "feature_mz": feature_mz,
            "q_value": qv,
            "is_decoy": bool(row.get("is_decoy", False)),
            "final_score": final_score,
        }
        for t in range(3):
            if t < len(order):
                srow[f"shap{t+1}_feature"] = sel_names[t]
                srow[f"shap{t+1}_value"] = float(sel_shap[t])
                srow[f"shap{t+1}_feature_value"] = float(sel_raw[t])
            else:
                srow[f"shap{t+1}_feature"] = ""
                srow[f"shap{t+1}_value"] = np.nan
                srow[f"shap{t+1}_feature_value"] = np.nan
        summary_rows.append(srow)

    for rank, pos in enumerate(target_pos):
        try:
            _explain_one(int(pos), rank, "target")
        except Exception as exc:
            logger.debug("debug_pfm_explanations: target row %d failed: %s", pos, exc)
            plt.close("all")
    for rank, pos in enumerate(decoy_pos):
        try:
            _explain_one(int(pos), rank, "decoy")
        except Exception as exc:
            logger.debug("debug_pfm_explanations: decoy row %d failed: %s", pos, exc)
            plt.close("all")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "summary.tsv"), sep="\t", index=False
        )
    logger.info(
        "debug_pfm_explanations: wrote %d figures + summary.tsv to %s",
        len(summary_rows), out_dir,
    )


# ---------------------------------------------------------------------------
# Target / decoy 3D scatter: m/z × ion mobility × intensity
# ---------------------------------------------------------------------------

def plot_mz_mobility_intensity_scatter(
    features_df: pd.DataFrame,
    out_dir: str,
    filename: str = "mz_mobility_intensity_scatter.png",
) -> None:
    """
    Scatter plot of candidates in (m/z, observed CCS, log-intensity) space.

    x: ``feature_mz``; y: ``im2deep_observed_ccs`` (falls back to
    ``im2deep_predicted_ccs``); colour: log10(``feature_intensity_p90`` + 1)
    mapped to a diverging colormap; marker shape: target (circle) vs decoy
    (cross). Silently skips when neither CCS column is present.

    Saved to ``{out_dir}/{filename}``.
    """
    ccs_col = None
    for c in ("im2deep_observed_ccs", "im2deep_predicted_ccs"):
        if c in features_df.columns:
            ccs_col = c
            break
    if ccs_col is None or "feature_mz" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    fmz = pd.to_numeric(feat["feature_mz"], errors="coerce").values
    ccs = pd.to_numeric(feat[ccs_col], errors="coerce").values
    is_decoy = _flag(feat, "is_decoy")

    # intensity column: p90 preferred, then raw, then ones
    int_col = next(
        (c for c in ("feature_intensity_p90", "feature_intensity") if c in feat.columns),
        None,
    )
    if int_col is not None:
        raw_int = pd.to_numeric(feat[int_col], errors="coerce").values
        raw_int = np.where(np.isfinite(raw_int) & (raw_int >= 0), raw_int, 0.0)
    else:
        raw_int = np.ones(len(feat), dtype=float)
    log_int = np.log10(raw_int + 1.0)

    valid = np.isfinite(fmz) & np.isfinite(ccs)
    if not valid.any():
        return

    fmz_v, ccs_v, li_v, dec_v = fmz[valid], ccs[valid], log_int[valid], is_decoy[valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    vmin, vmax = float(np.percentile(li_v, 5)), float(np.percentile(li_v, 95))
    if vmin >= vmax:
        vmin, vmax = 0.0, max(1.0, float(li_v.max()))

    for mask, marker, label in [
        (~dec_v, "o", "Target"),
        (dec_v,  "x", "Decoy"),
    ]:
        if not mask.any():
            continue
        sc = ax.scatter(
            fmz_v[mask], ccs_v[mask],
            c=li_v[mask], cmap="viridis",
            vmin=vmin, vmax=vmax,
            s=6 if marker == "o" else 8,
            alpha=0.4 if marker == "o" else 0.6,
            marker=marker,
            linewidths=0.5,
            label=label,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"log₁₀({int_col or 'intensity'} + 1)", fontsize=9)

    ccs_label = "Observed CCS (Å²)" if ccs_col == "im2deep_observed_ccs" else "Predicted CCS (Å²)"
    ax.set_xlabel("Feature m/z (Da)", fontsize=10)
    ax.set_ylabel(ccs_label, fontsize=10)
    ax.set_title("Target vs decoy: m/z × ion mobility × intensity", fontsize=11)
    ax.legend(markerscale=2, fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def save_debug_figures(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    *,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    maldi_envelopes: dict | None = None,
    feature_names: list[str] | None = None,
    model_name: str = "model",
    importances: np.ndarray | None = None,
    importance_names: list[str] | None = None,
    structure_coefs: np.ndarray | None = None,
    structure_names: list[str] | None = None,
    debug_dir: str = "debug",
    n_subset: int = 50,
    seed: int = 42,
    gt_peptides: list[str] | None = None,
    ccs_tol_pct: float | None = None,
) -> None:
    """
    Generate all debug figures and save them under ``debug_dir``.

    Parameters
    ----------
    features_df
        Full candidate DataFrame (all rows, before any filtering), same row
        order as ``result_df``.
    result_df
        Scoring output from ``rescore()`` (same row order as ``features_df``).
    ion_images
        MALDI ion images array, shape (n_features, H, W) float32.
    ion_image_mzs
        m/z values corresponding to ``ion_images`` rows (length n_features).
    maldi_envelopes
        Dict mapping float feature_mz → [m0_mean, m1_mean, m2_mean].
    feature_names
        Feature names used by the model (for importance plots).
    model_name
        Scoring model identifier used in output file names.
    importances
        Feature importance array aligned with ``importance_names``.
    importance_names
        Feature names aligned with importance arrays; defaults to
        ``feature_names`` when omitted.
    debug_dir
        Root output directory.  Sub-directories are created automatically.
    n_subset
        Number of candidates to sample for per-candidate figures.
    seed
        Random seed for reproducible sampling.
    """
    os.makedirs(debug_dir, exist_ok=True)
    # Empty ion_images/ first. Figure names carry a rank, and a protein's rank moves
    # between runs, so re-running into an existing directory leaves both copies under
    # different names with nothing to tell them apart. kidney's E025 directory held 103
    # files for 59 figures for exactly that reason. The sibling figure directories
    # (features/, isotope_envelopes/, feature_importance/, pfm_explanations/) have the
    # same property and are deliberately left alone here — see PROGRESS.md.
    _reset_figure_dir(os.path.join(debug_dir, "ion_images"))

    subset = _sample_subset(features_df, result_df, n=n_subset, seed=seed)
    logger.info("Debug viz: sampled %d candidates from %d", len(subset), len(features_df))

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)
    is_winner = _flag(res, "is_tdc_winner")
    q = _num(res, "q_value")

    # feature_mz -> q-value and peptide of that feature's TDC winner. Used by
    # plot_ion_image_colocalization to rank and annotate the co-feature panels.
    feature_qvals: dict[float, float] = {}
    feature_peptides: dict[float, str] = {}
    if "feature_mz" in feat.columns:
        for i in np.where(is_winner)[0]:
            mz = float(feat["feature_mz"].iat[i])
            feature_qvals[mz] = float(q[i])
            if "peptide" in feat.columns:
                feature_peptides[mz] = str(feat["peptide"].iat[i])

    def _protein_image_subset() -> pd.DataFrame:
        # Every (unsampled) FDR <= 5% winner plus the sampled R1/L rows, collapsed to
        # one row per protein: each ion-image figure already shows the whole protein,
        # so its best-q peptide represents it. With no winner at 5%, the five
        # lowest-PEP winners stand in, as in _sample_subset.
        id_mask = is_winner & (q <= 0.05)
        if not id_mask.any() and "pep" in res.columns:
            id_mask = np.zeros(len(res), dtype=bool)
            id_mask[_pep_top(is_winner, _num(res, "pep"))] = True
        id_subset = _annotate_subset(feat, res, np.where(id_mask)[0], "ID")
        out = _one_row_per_peptide(pd.concat(
            [id_subset, subset[subset["_group"].isin(["R1", "L"])]], ignore_index=True,
        ))
        # Cap the figure count: above 300 proteins, subsample 200 (reproducibly).
        if len(out) > 300:
            keep = sorted(np.random.default_rng(seed).choice(len(out), size=200, replace=False).tolist())
            logger.info(
                "Ion image colocalization: %d proteins exceed 300; subsampled to 200 for visualization",
                len(out),
            )
            out = out.iloc[keep].reset_index(drop=True)
        out["_rank"] = np.arange(1, len(out) + 1)
        logger.info(
            "Ion image colocalization: one figure per protein — %d proteins "
            "(%d with an ID at ≤5%% FDR)", len(out), int((out["_group"] == "ID").sum()),
        )
        return out

    def j(*parts: str) -> str:
        return os.path.join(debug_dir, *parts)

    steps: list = []
    if ion_images is not None:
        steps += [
            ("Ion image colocalization figures", lambda: plot_ion_image_colocalization(
                _protein_image_subset(), features_df, ion_images, ion_image_mzs,
                out_dir=j("ion_images"),
                feature_qvals=feature_qvals, feature_peptides=feature_peptides,
            )),
            ("Ion image Pearson distribution", lambda: plot_ion_image_pearson_distribution(
                features_df, result_df, ion_images, ion_image_mzs, out_dir=debug_dir,
            )),
            ("Protein spatial coherence", lambda: plot_protein_spatial_coherence(
                features_df, result_df, ion_images, ion_image_mzs, out_dir=debug_dir,
            )),
        ]
    steps += [
        ("Feature diagnostic figures", lambda: plot_feature_diagnostics(
            subset, features_df, ion_images, ion_image_mzs, maldi_envelopes,
            out_dir=j("features"),
        )),
        ("Isotope envelope figures", lambda: plot_isotope_envelope_figures(
            subset, maldi_envelopes, out_dir=j("isotope_envelopes"),
        )),
        ("Feature distribution figures", lambda: plot_feature_distributions(
            features_df, result_df, out_dir=j("feature_distributions"),
            feature_names=feature_names, gt_peptides=gt_peptides,
        )),
        ("CCS scatter", lambda: plot_ccs_scatter(
            features_df, result_df, out_dir=debug_dir,
            gt_peptides=gt_peptides, ccs_tol_pct=ccs_tol_pct,
        )),
        ("m/z × mobility × intensity scatter", lambda: plot_mz_mobility_intensity_scatter(
            features_df, out_dir=debug_dir,
        )),
        ("IDs vs FDR curve", lambda: plot_ids_vs_fdr(result_df, out_dir=debug_dir)),
        ("Protein colocalization by group", lambda: plot_protein_colocalization_by_group(
            features_df, result_df, out_dir=debug_dir,
        )),
    ]
    steps += [
        ("T/D m/z distribution", lambda: plot_target_decoy_mz_distribution(
            features_df, result_df, out_dir=debug_dir,
        )),
        ("Candidate competition", lambda: plot_candidate_competition(
            features_df, result_df, out_dir=debug_dir, ccs_tol_pct=ccs_tol_pct,
        )),
        ("Score PP plot", lambda: plot_score_pp(
            features_df, result_df, out_dir=debug_dir,
        )),
        ("PEP mixture plot", lambda: plot_pep_mixture(
            result_df, out_dir=debug_dir, model_name=model_name,
        )),
        ("Score distributions", lambda: plot_score_distributions(
            features_df, result_df, out_dir=debug_dir,
        )),
    ]
    if importances is not None:
        steps.append(("Feature importance figures", lambda: plot_feature_importance(
            importance_names or feature_names or [], importances,
            out_dir=j("feature_importance"), model_name=model_name,
            structure_coefs=structure_coefs, structure_names=structure_names,
        )))

    # --- Ground-truth peptide figures (after the main set: ion_images/ dedups by protein) ---
    if gt_peptides:
        try:
            gt_subset, not_found = _make_gt_subset(gt_peptides, features_df, result_df)
        except Exception as exc:
            logger.warning("GT debug figures failed: %s", exc)
            gt_subset, not_found = None, []
        if not_found:
            logger.info(
                "GT peptides not found as candidates (%d): %s", len(not_found), ", ".join(not_found),
            )
            steps.append(("GT not-a-candidate figures", lambda: _save_gt_not_found_figures(
                not_found, subdirs=[j("features"), j("isotope_envelopes")],
            )))
        if gt_subset is not None:
            logger.info(
                "GT debug viz: %d rows for %d GT peptides",
                len(gt_subset), len(gt_peptides) - len(not_found),
            )
            steps += [
                ("GT feature diagnostic figures", lambda: plot_feature_diagnostics(
                    gt_subset, features_df, ion_images, ion_image_mzs, maldi_envelopes,
                    out_dir=j("features"),
                )),
                ("GT isotope envelope figures", lambda: plot_isotope_envelope_figures(
                    gt_subset, maldi_envelopes, out_dir=j("isotope_envelopes"),
                )),
            ]
            if ion_images is not None:
                steps.append(("GT ion image figures", lambda: plot_ion_image_colocalization(
                    gt_subset, features_df, ion_images, ion_image_mzs,
                    out_dir=j("ion_images"),
                    feature_qvals=feature_qvals, feature_peptides=feature_peptides,
                )))

    for name, draw in steps:
        try:
            draw()
            logger.info("Debug viz: %s done", name)
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)
