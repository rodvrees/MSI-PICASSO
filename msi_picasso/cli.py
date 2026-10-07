"""Command-line interface for MSI-PICASSO."""

import argparse
import inspect
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from msi_picasso import __version__

logger = logging.getLogger(__name__)


class _Tee:
    """Write to multiple streams at once (stdout/stderr -> terminal + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ---------------------------------------------------------------------------
# MALDI data loading
# ---------------------------------------------------------------------------


def _read_feature_mzs(path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Load feature m/z values from a plain text file or a SCiLS Lab CSV export.

    Plain text: one m/z value per line, no header. Returns (mzs, None, None).
    SCiLS CSV: semicolon-delimited; lines starting with '#' are comments;
               the first non-comment line is a header (first column = 'm/z').
               If a 'CCS [Å²]' column is present it is returned as the second
               element. If an 'Intensity' column is present (e.g. the SCiLS
               'Intensity [Regions]' column) it is returned as the third element.
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    data_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    if not data_lines:
        raise ValueError(f"No data lines found in {path!r}")

    if ";" in data_lines[0]:
        # SCiLS-style: first non-comment line is header, rest are data rows.
        header = data_lines[0].split(";")
        rows = data_lines[1:]
        mzs = np.array([float(row.split(";")[0]) for row in rows if row.strip()], dtype=np.float64)

        def _extract_col(col_idx: int) -> np.ndarray:
            vals: list[float] = []
            for row in rows:
                if row.strip():
                    parts = row.split(";")
                    try:
                        vals.append(float(parts[col_idx]))
                    except (IndexError, ValueError):
                        vals.append(np.nan)
            return np.array(vals, dtype=np.float64)

        # Find CCS column
        ccs_col_idx = None
        for i, col in enumerate(header):
            if "CCS" in col and "Å" in col:  # Å = U+00C5
                ccs_col_idx = i
                break
        ccs: np.ndarray | None = _extract_col(ccs_col_idx) if ccs_col_idx is not None else None

        # Find intensity column — prefer 'Intensity [Regions]', accept any 'Intensity' column
        # that is not an interval-width or CCS column.
        intensity_col_idx = None
        for i, col in enumerate(header):
            col_lower = col.lower()
            if "intensity" in col_lower and "interval" not in col_lower and "width" not in col_lower:
                intensity_col_idx = i
                break
        intensities: np.ndarray | None = (
            _extract_col(intensity_col_idx) if intensity_col_idx is not None else None
        )
    else:
        mzs = np.array([float(ln) for ln in data_lines], dtype=np.float64)
        ccs = None
        intensities = None

    return mzs, ccs, intensities


# ---------------------------------------------------------------------------
# Digest parameter inference from LC-MS/MS identifications
# ---------------------------------------------------------------------------


def _infer_digest_params(
    lcms_ids,
    missed_cleavages_override: int | None,
    min_length_override: int | None,
    max_length_override: int | None,
) -> tuple[int, int, int]:
    """
    Infer missed_cleavages, min_length, and max_length from LC-MS/MS peptides.

    Override values take priority. Falls back to standard defaults
    (2 / 7 / 30) if the peptide table is empty.
    """
    seqs = lcms_ids.peptides["sequence"] if len(lcms_ids.peptides) > 0 else None

    if seqs is not None and len(seqs) > 0:
        inferred_min = int(seqs.str.len().min())
        inferred_max = int(seqs.str.len().max())

        def _count_mc(seq: str) -> int:
            return sum(1 for aa in seq[:-1] if aa in "KR")

        inferred_mc = int(seqs.apply(_count_mc).max())
    else:
        inferred_min, inferred_max, inferred_mc = 7, 30, 2

    min_length = (
        min_length_override if min_length_override is not None else inferred_min
    )
    max_length = (
        max_length_override if max_length_override is not None else inferred_max
    )
    missed_cleavages = (
        missed_cleavages_override
        if missed_cleavages_override is not None
        else inferred_mc
    )

    logger.info(
        f"Digest parameters: min_length={min_length}, max_length={max_length}, "
        f"missed_cleavages={missed_cleavages}"
        + (" (inferred from LC-MS/MS IDs)" if seqs is not None else " (defaults)")
    )
    return min_length, max_length, missed_cleavages


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_results(
    result,
    output_dir: str,
) -> None:
    """Write rescoring results to TSV files in ``output_dir``.

    ``ms1rescore_matches.tsv`` gets every candidate, targets and decoys, with its
    q-value annotation and no filtering: downstream consumers filter on
    ``is_peptide_winner``/``peptide_q_value`` (or the feature-level equivalents for
    results predating F-029) and ``is_decoy`` themselves.

    ``ms1rescore_peptides.tsv`` is the confident-identification list, so it is
    filtered: targets only, one row per peptide, at 1% FDR. It previously did none
    of those things despite its name -- it kept every peptide-feature pair, applied
    the feature-level q-value, and **did not exclude decoys**, so amyloidosis E018
    shipped 1294 rows covering 308 peptides of which 11 were decoys, where the
    reported count was 247 target peptides. See PROGRESS.md F-035.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ms1rescore_matches.tsv")
    result.to_csv(out_path, sep="\t", index=False)
    n_winners = result.get("is_tdc_winner", result["is_decoy"].apply(lambda x: not x)).sum()
    logger.info(f"  Wrote {len(result)} candidates ({n_winners} TDC winners) → {out_path}")

    # Prefer the peptide-level population, matching the reported ID count (F-029).
    # Fall back to feature level for raw-query results, which have no peptide-level
    # columns; the log line says which was used.
    if "is_peptide_winner" in result.columns and "peptide_q_value" in result.columns:
        win_col, q_col, level = "is_peptide_winner", "peptide_q_value", "peptide-level"
    elif "is_tdc_winner" in result.columns and "q_value" in result.columns:
        win_col, q_col, level = "is_tdc_winner", "q_value", "feature-level"
    else:
        return

    confident = result[
        result[win_col].astype(bool)
        & ~result["is_decoy"].astype(bool)
        & (result[q_col] <= 0.01)
    ]
    cols = [c for c in ["feature_idx", "feature_mz", "feature_ccs", "peptide", "protein",
                        q_col] if c in confident.columns]
    peptides = (
        confident[cols]
        .sort_values(q_col)
        .drop_duplicates(subset=["peptide"], keep="first")
    )
    peptides_out = os.path.join(output_dir, "ms1rescore_peptides.tsv")
    peptides.to_csv(peptides_out, sep="\t", index=False)
    logger.info(
        "  Wrote %d confident target peptides at 1%% FDR (%s) → %s",
        len(peptides), level, peptides_out,
    )

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MSI-PICASSO",
        description=(
            "Symmetric target-decoy rescoring for MALDI-MSI MS1 data. "
            "Matches MALDI features to an in-silico tryptic digest and "
            "rescores candidates using MALDI-intrinsic features, with "
            "LC-MS/MS identifications as the candidate source."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"MSI-PICASSO {__version__}"
    )

    parser.add_argument(
        "-c",
        "--config-file",
        default=None,
        metavar="PATH",
        help=(
            "JSON or TOML configuration file. Values here override defaults but "
            "are overridden by explicit CLI arguments."
        ),
    )

    # --- Required inputs ---
    req = parser.add_argument_group("required inputs")
    req.add_argument(
        "--fasta",
        "-f",
        required=False,
        default=None,
        metavar="PATH",
        help=(
            "Protein FASTA file (forward sequences only; decoys are generated). "
            "Required when --digest is specified. Ignored otherwise."
        ),
    )
    req.add_argument(
        "--digest",
        action="store_true",
        default=False,
        help=(
            "Digest the provided --fasta to create additional candidates beyond "
            "the LC-MS/MS identified peptides. Requires --fasta. Without this flag, "
            "only LC-MS/MS identified peptides (--lcms-peptides or --msf) are used "
            "as candidates and --fasta is not needed."
        ),
    )
    req.add_argument(
        "--extra-fasta",
        metavar="PATH",
        default=None,
        help=(
            "Additional FASTA file whose proteins are always included in the "
            "candidate database (e.g. contaminants, spike-ins). Works with both "
            "Strategy A and C. Peptides already present in the primary database "
            "are not duplicated."
        ),
    )

    # --- MALDI input ---
    maldi_group = parser.add_argument_group("MALDI input")
    maldi_group.add_argument(
        "--maldi-d",
        metavar="PATH",
        help=(
            "Bruker .d directory (required). Ion images, including the adduct and "
            "isotopologue extra images, are extracted from it."
        ),
    )

    maldi_group.add_argument(
        "--feature-mzs",
        metavar="PATH",
        default=None,
        help=(
            "Peak list from the TIMSImaging fork's 2D peak picking; ion images and "
            "spatial features are extracted at these m/z values. Semicolon-delimited, "
            "'#' comment lines, first data column = m/z, optional 'CCS [Å²]' column."
        ),
    )
    maldi_group.add_argument(
        "--maldi-query-raw",
        action="store_true",
        default=None,
        help=(
            "Raw-query mode (superseded; kept to reproduce older results). Instead of "
            "using a feature list, generate candidates first and extract ion images "
            "directly from the --maldi-d .d at the candidate-derived m/z values."
        ),
    )
    maldi_group.add_argument(
        "--raw-query-cache-dir",
        default=None,
        help=(
            "Directory in which to cache the raw-query observed-centroid/CCS alphatims "
            "pass (the dominant cost of a raw-query run). Keyed by the .d path, the "
            "candidate m/z grid and the window parameters, so a re-run that changes only "
            "scoring settings reuses it; a changed candidate set misses and recomputes."
        ),
    )

    # --- Raw extraction parameters ---
    raw_grp = parser.add_argument_group(
        "raw MALDI extraction",
        description="Parameters for ion image extraction from the Bruker .d directory.",
    )
    raw_grp.add_argument(
        "--extraction-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "m/z window for raw ion image extraction (ppm). Controls which raw "
            "data points contribute to each ion image. Should be slightly wider "
            "than the instrument's typical peak width. Default: 25.0."
        ),
    )
    raw_grp.add_argument(
        "--matching-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "m/z window for candidate matching (ppm). Applied when linking "
            "peptide candidates to detected MALDI features. Default: 20.0."
        ),
    )
    raw_grp.add_argument(
        "--save-npz",
        metavar="PATH",
        help=(
            "Save extracted features and ion images as an NPZ file to this path "
            "(for offline inspection; the CLI does not read it back)."
        ),
    )
    raw_grp.add_argument(
        "--save-spatial",
        metavar="PATH",
        help=(
            "Save the computed spatial features TSV to this path "
            "(columns: feature_mz, n_pixels_detected, fraction_detected, "
            "mean_intensity, intensity_p90, intensity_sum, "
            "spatial_autocorrelation, intensity_cv)."
        ),
    )

    # --- Candidate generation ---
    cand = parser.add_argument_group("candidate generation")
    cand.add_argument(
        "--ppm-tolerance",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Mass tolerance for MALDI-to-database matching (ppm).",
    )
    cand.add_argument(
        "--missed-cleavages",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Maximum missed cleavages for in-silico digest. When "
            "--lcms-peptides is provided (Strategy C), inferred from the "
            "maximum number of internal K/R in identified sequences. "
            "Default for Strategy A (full FASTA): 2."
        ),
    )
    cand.add_argument(
        "--min-length",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Minimum peptide length. Inferred from LC-MS/MS IDs when "
            "--lcms-peptides is provided. Default for Strategy A: 7."
        ),
    )
    cand.add_argument(
        "--max-length",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Maximum peptide length. Inferred from LC-MS/MS IDs when "
            "--lcms-peptides is provided. Default for Strategy A: 30."
        ),
    )
    cand.add_argument(
        "--decoy-method",
        choices=("substitution", "mz_shuffle"),
        default=None,
        help=(
            "Decoy generation strategy. 'substitution' (default): interior residues of "
            "each target peptide are substituted (see --substitution-*). 'mz_shuffle': "
            "derangement of the peptide->feature assignment, so each real target "
            "peptide is relocated onto another peptide's real feature (co-located "
            "1 target + 1 decoy per feature); feature-quality features are then "
            "symmetric and the ranker must discriminate on the peptide match "
            "(CCS/isotope)."
        ),
    )
    cand.add_argument(
        "--substitution-n-residues",
        type=int,
        default=None,
        metavar="INT",
        help="substitution only: number of interior residues to substitute per peptide (default 1).",
    )
    cand.add_argument(
        "--substitution-seed",
        type=int,
        default=None,
        metavar="INT",
        help="substitution only: random seed for per-peptide RNG (default 42).",
    )
    cand.add_argument(
        "--substitution-no-collision-filter",
        action="store_true",
        default=None,
        help=(
            "substitution only: disable the collision filter that rejects decoys "
            "whose [M+H]+ falls within matching_ppm of a target or an already-assigned "
            "decoy. Enabled by default; pass this flag to disable."
        ),
    )
    cand.add_argument(
        "--substitution-collision-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "substitution only: m/z separation (ppm) the collision filter enforces "
            "between a decoy and any target. Defaults to matching_ppm, which is 0 in "
            "raw-query mode and therefore disables the filter; set at least the "
            "extraction window so a decoy cannot capture a target's ion image."
        ),
    )
    cand.add_argument(
        "--substitution-mass-shift-min-da",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "substitution only: minimum absolute mass shift in Da. Default: auto "
            "(matching_ppm × mh_mz / 1e6 per peptide). For single-residue substitution "
            "this threshold is always satisfied automatically (N↔D minimum is ~0.96 Da)."
        ),
    )
    cand.add_argument(
        "--substitution-mass-shift-max-da",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "substitution only: maximum absolute NET mass shift in Da; a substitution "
            "exceeding it is rejected and resampled, so no decoy is lost. Default: unset "
            "(no cap). H-fdr-10: a large shift leaves the decoy's elemental composition far "
            "from its source target's, and the isotope-envelope features read composition, "
            "so those decoys separate from targets for a construction reason rather than a "
            "spectral one. Measured AUC 0.58-0.66 above ~120 Da against 0.50-0.53 below "
            "(PROGRESS.md F-036). Two substitutions cannot exceed ~258 Da, so a cap above "
            "that is a no-op."
        ),
    )
    cand.add_argument(
        "--substitution-residue-weighting",
        choices=("uniform", "target_frequency"),
        default=None,
        help=(
            "substitution only: how the replacement residue is drawn. 'uniform' "
            "(default) draws evenly over the 18-letter alphabet, so every residue "
            "appears 5.6%% of the time whatever its real abundance. "
            "'target_frequency' draws from the empirical residue frequency of the "
            "target peptides instead (H-decoy-15). Cys and Met are 2 of 18 letters "
            "but only 1.4-1.8%% of real residues, so the uniform draw over-produces "
            "sulfur by 6-8x and leaves decoys with twice the sulfur of targets; "
            "features that read composition then partially read the target/decoy "
            "label (PROGRESS.md F-036, F-042). Acts on every composition axis."
        ),
    )
    cand.add_argument(
        "--substitution-preserve-sulfur",
        action="store_true",
        default=None,
        help=(
            "substitution only: never substitute Cys or Met in or out, so every "
            "decoy carries exactly its source target's sulfur count. Narrower than "
            "--substitution-residue-weighting but exact on the axis that dominates "
            "isotope-envelope shape. The two compose."
        ),
    )
    cand.add_argument(
        "--feature-mzs-keep",
        default=None,
        metavar="PATH",
        help=(
            "Peak list naming the peaks to KEEP ion images for, produced by "
            "scripts/prefilter_peaklist.py. Ion images are extracted at every m/z in "
            "--feature-mzs, the on-tissue TIC mask is computed over all of them, and "
            "only the peaks listed here are retained. Only 18-34%% of peaks are ever "
            "matched by a candidate, so this cuts ion-image memory 3-5x with no change "
            "to any feature. Do NOT pass the reduced list as --feature-mzs instead: the "
            "mask is a sum over every peak and drops with them (PROGRESS.md F-048)."
        ),
    )
    cand.add_argument(
        "--protein-size-residualize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Replace the size-driven protein-level features with size-free companions "
            "(log_protein_n_features, protein_colocalization_n_partners, protein_coverage, "
            "is_single_peptide_protein, protein_best_ratio -> *_sizeresid), ranking each "
            "within bins of the protein's tryptic count. Default ON. In feature-list mode "
            "targets and decoys match detected peaks at the same rate, so these features "
            "mostly read how many peptides a protein HAS rather than whether it is present "
            "(PROGRESS.md F-045). Use --no-protein-size-residualize to reproduce a run "
            "from before E025."
        ),
    )

    # --- Rescoring ---
    rescore_grp = parser.add_argument_group("rescoring")
    rescore_grp.add_argument(
        "--model",
        choices=("lda", "svm", "rbf_svm"),
        default=None,
        help=(
            "Rescoring backend. 'lda' (default): sklearn LinearDiscriminantAnalysis "
            "with median imputation and standardization. 'svm': sklearn LinearSVC "
            "(penalty=l2, squared_hinge, C=--svm-c). 'rbf_svm': sklearn "
            "SVC(kernel='rbf') (nonlinear, continuous scores); tuned by --rbf-svm-* "
            "flags. All three share the same semi-supervised loop."
        ),
    )
    rescore_grp.add_argument(
        "--svm-c",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Regularization strength C for the --model svm (LinearSVC) backend. Default 1.0.",
    )
    rescore_grp.add_argument(
        "--rbf-svm-c",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Regularization strength C for the --model rbf_svm backend. Default 1.0 "
             "(C~5-10 often improves yield).",
    )
    rescore_grp.add_argument(
        "--rbf-svm-gamma",
        type=str,
        default=None,
        metavar="SCALE|AUTO|FLOAT",
        help="RBF kernel gamma for the --model rbf_svm backend: 'scale' (default), "
             "'auto', or a float (e.g. 0.01-0.03 on standardized features often "
             "outperforms 'scale').",
    )
    rescore_grp.add_argument(
        "--init-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "FDR threshold used for best-feature seed initialization "
            "and pairwise combination search (default 0.2)."
        ),
    )
    rescore_grp.add_argument(
        "--train-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "FDR threshold for the per-iteration pseudo-label update (default 0.05)."
        ),
    )
    rescore_grp.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="INT",
        help="Maximum pseudo-label iterations (default 5).",
    )
    rescore_grp.add_argument(
        "--init-ppm-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "ppm_error_abs threshold for the fallback seed, used only when the "
            "best-feature seed search finds no passing targets."
        ),
    )
    rescore_grp.add_argument(
        "--only-main-features",
        action="store_true",
        help=(
            "Replace MALDI_INTRINSIC_FEATURES with a reduced set of ~19 "
            "non-collinear representative features (MAIN_FEATURES). "
            "Removes redundancy within collinear groups (ppm, isotope_theo, "
            "sequence_comp, etc.) before training. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--use-protein-level-feats",
        action="store_true",
        help=(
            "Include protein-level features (protein_n_features, protein_coverage, "
            "protein_rank, protein_best_ratio, protein_colocalization_*) in the "
            "rescoring model. These features aggregate signal across all candidates "
            "sharing a protein, which can break the TDC null model symmetry. "
            "Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--use-spatial-ranker-features",
        action="store_true",
        default=None,
        help=(
            "Include spatial ranker features (spatial_autocorrelation, spatial_morans_i, "
            "spatial_gearys_c, fraction_detected, intensity_cv, and protein_colocalization_*) "
            "in the rescoring model. Both decoy methods put each decoy on a real MALDI "
            "feature, so these have a symmetric null. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--n-debug",
        type=int,
        default=None,
        metavar="INT",
        help="Number of candidates to sample for per-candidate debug figures (default 50).",
    )
    rescore_grp.add_argument(
        "--debug-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for debug candidate sampling (default 42).",
    )
    rescore_grp.add_argument(
        "--debug-gt",
        metavar="PATH",
        default=None,
        help=(
            "Path to a plain-text file with one ground-truth peptide sequence per line. "
            "If --verbose is set, debug figures are generated for each GT peptide "
            "found among the candidates (prefixed GT_). Peptides absent from the "
            "candidate set produce a 'not a candidate' placeholder figure. "
            "Ignored when --verbose is not set."
        ),
    )
    rescore_grp.add_argument(
        "--features-preset",
        choices=("all", "main"),
        default=None,
        help=(
            "'all' (default): use MALDI_INTRINSIC_FEATURES. 'main': use the "
            "reduced MAIN_FEATURES set. Overridden by --only-main-features."
        ),
    )
    rescore_grp.add_argument(
        "--features-exclude",
        nargs="*",
        default=None,
        metavar="FEATURE",
        help=(
            "Space-separated list of feature names to exclude from the ranker. "
            "Example: --features-exclude peptide_length n_proline. "
            "Useful for ablation studies without editing source code."
        ),
    )
    rescore_grp.add_argument(
        "--seed-features",
        nargs="*",
        default=None,
        metavar="FEATURE",
        help=(
            "Restrict best-feature seed initialization to only these features "
            "(single/pairwise/tree sweeps), still respecting the composition-leak "
            "skip guard. Empty (default) seeds from all eligible features. Use to "
            "seed from tissue-independent axes (e.g. --seed-features "
            "im2deep_abs_delta_ccs_pct_resid ppm_error_pct) when colocalization is "
            "non-discriminative."
        ),
    )
    rescore_grp.add_argument(
        "--r1-seed-percentile",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Fallback top-N ppm percentile for initial positive seed when the "
            "ppm/isotope threshold yields zero positives (default 0.10 = top 10%%)."
        ),
    )
    rescore_grp.add_argument(
        "--min-seed-positives",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Minimum number of pseudo-positive targets required from "
            "the single-feature sweep before the pairwise combination search is "
            "triggered. When fewer than this many targets pass at q<=init_fdr, all "
            "unique feature pairs are tried. Default: 50."
        ),
    )

    rescore_grp.add_argument(
        "--train-fdr-escalate",
        action="store_true",
        default=None,
        help=(
            "H-fdr-2: if init_fdr/train_fdr would otherwise yield zero pseudo-positives "
            "at the seed step or a pseudo-label iteration, retry at increasing "
            "thresholds (steps of 0.005, capped at 0.5) instead of giving up. A no-op "
            "whenever the configured threshold already succeeds. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--pseudo-label-growth-cap",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "H-fdr-5: stop the self-training loop (keeping the previous iteration's "
            "model) once the pseudo-positive count exceeds this multiple of the "
            "initial seed size, regardless of convergence. Guards against the loop "
            "amplifying a weak or leaking seed into a runaway positive set. Unset "
            "(default) disables the cap."
        ),
    )
    rescore_grp.add_argument(
        "--model-repeats",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "H-fdr-6: average the scores of this many independent replicate fits, "
            "each with its own CV partition. The self-training loop does not converge, "
            "so one fit's q-value floor is a draw rather than a property of the data -- "
            "on kidney, twelve partitions gave 0 to 63 peptides at 5%% FDR with nothing "
            "else changed. 1 (default) is a single fit and reproduces results predating "
            "this."
        ),
    )
    rescore_grp.add_argument(
        "--winner-percentile",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Round-1 winner filter: drop features whose winner score falls below "
            "this quantile of all winner scores (default 0.02)."
        ),
    )
    rescore_grp.add_argument(
        "--match-ccs",
        action="store_true",
        default=False,
        help=(
            "After IM2Deep finetuning, filter candidates by predicted CCS. The tolerance "
            "threshold is data-driven: p95 absolute %%CCS error on single-candidate (m/z "
            "unambiguous) matches × --ccs-window-multiplier. Requires observed CCS values "
            "in the feature m/z file and IM2Deep installed."
        ),
    )
    rescore_grp.add_argument(
        "--ccs-window-multiplier",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "CCS filter threshold = multiplier × p95 |delta_CCS%%| on single-candidate "
            "calibration matches. Default 2.0. "
            "Ignored when --ccs-window-pct is set."
        ),
    )
    rescore_grp.add_argument(
        "--save-ion-images",
        action="store_true",
        default=None,
        help=(
            "Write the full ion-image array to 2_ion_images.npy. Off by default: nothing "
            "in the pipeline reads it back, it is there for the notebooks, and it is the "
            "largest thing a run writes (123 GB for her2 at min_regions=1). Leaving it on "
            "filled the disk and killed four runs."
        ),
    )
    rescore_grp.add_argument(
        "--ccs-window-pct",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Fixed CCS filter threshold in percent, overriding --ccs-window-multiplier. "
            "The multiplier scales a p95 measured on whichever calibration peptides the "
            "run has, and a denser peak list raises it (amyloidosis 3.18%% at "
            "min_regions=2, 4.06%% at 1), so two runs cannot share a threshold unless it "
            "is fixed. Keep it above the ground truth's own CCS error: the reachable "
            "confirmed peptides need 1.84%% on amyloidosis, 1.45%% on her2, 0.90%% on "
            "kidney (PROGRESS.md F-050)."
        ),
    )
    rescore_grp.add_argument(
        "--mob-coloc",
        action="store_true",
        default=None,
        help=(
            "Compute per-candidate mobility-filtered colocalization features (the *_mob "
            "features). Reads the raw Bruker .d via alphatims and filters peaks to each "
            "candidate's predicted 1/K0 window. Requires IM2Deep predicted CCS (i.e. "
            "observed CCS available) and the raw .d (--maldi-d). Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--mob-protein-coloc",
        action="store_true",
        default=None,
        help=(
            "Additionally compute protein-level mobility-gated colocalization "
            "(protein_colocalization_mob*). Requires --mob-coloc. Rebuilds per-candidate "
            "M0 images from the raw TDF and computes within-protein pairwise Pearson r "
            "on those images. Slow for large datasets; disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--mob-window-multiplier",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Mobility-window half-width = multiplier × p95 |delta 1/K0| on the calibration "
            "set, for --mob-coloc. Default 2.0."
        ),
    )
    rescore_grp.add_argument(
        "--mob-quality-mz-window-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Raw-query only: outer m/z half-window (ppm) for the intrinsic 2D peak-quality "
            "features (mob_2d_concentration, mob_k0_spread, mob_mz_spread_ppm, mob_peak_snr). "
            "Effectively bounded by --extraction-ppm (peaks are collected there). Default 25.0."
        ),
    )
    rescore_grp.add_argument(
        "--mob-quality-k0-tol",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Raw-query only: 1/K0 half-width (V·s/cm²) of the peak band / concentration box "
            "for the intrinsic 2D peak-quality features. Default 0.02."
        ),
    )
    rescore_grp.add_argument(
        "--coloc-tic-quantile",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "On-tissue pixel quantile for colocalization features. Pearson r is computed "
            "only on pixels whose total-ion-current exceeds this quantile of measured TIC, "
            "removing the shared on/off-tissue component that inflates every correlation. "
            "0.0 (default) keeps all measured pixels (drops only unmeasured padding); e.g. "
            "0.25 additionally trims low-signal tissue edges. Range [0, 1)."
        ),
    )
    rescore_grp.add_argument(
        "--coloc-measured-mask",
        action="store_true",
        default=None,
        help=(
            "Restrict colocalization to pixels that were actually rastered, using the "
            "pixel coordinate list from the MALDI data source rather than the TIC > 0 "
            "heuristic. Useful for partial-raster acquisitions where only a sub-region "
            "of the slide was scanned."
        ),
    )
    rescore_grp.add_argument(
        "--cosine-coloc",
        action="store_true",
        default=None,
        help=(
            "Compute median-thresholded cosine within-protein colocalization features "
            "(protein_colocalization_cosine*; PROGRESS.md H-feat-3/H-decoy-9). Each on-tissue "
            "ion image is thresholded at its own median, then compared by cosine similarity — "
            "Ovchinnikova et al. 2020 (ColocML) validated this at Spearman 0.794 against 42 "
            "expert raters, matching a trained deep model. Takes only the observed ion images, "
            "so it is safe by construction under every decoy method including mz_shuffle. "
            "Requires ion images; protein-level, so also needs --use-protein-level-feats. "
            "Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--coloc-tic-normalize",
        action="store_true",
        default=None,
        help=(
            "Per-pixel TIC-normalize ion images before computing protein colocalization "
            "(divide each pixel by its total signal across images), removing the shared "
            "'more tissue = more of everything' brightness envelope. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--coloc-common-mode",
        action="store_true",
        default=None,
        help=(
            "Subtract the per-pixel mean image (the shared tissue outline) before computing "
            "protein colocalization, so only the protein-specific residual is correlated. "
            "Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--drop-zero-signal",
        action="store_true",
        default=None,
        help=(
            "Remove candidates whose MALDI feature has zero total signal "
            "(feature_intensity_sum == 0) before scoring. Under raw-query mode these "
            "candidates have no MALDI evidence and their co-located target/decoy pairs "
            "differ only on peptide_length, which leaks the mz_shuffle derangement. "
            "Symmetric: both target and decoy at a zero-signal feature are dropped. "
            "Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--entrapment",
        action="store_true",
        default=None,
        help=(
            "Inject shuffled pseudo-protein peptides (derived from LC-MS/MS confirmed "
            "sequences, one pseudo-protein per identified protein) as entrapment "
            "pseudo-targets alongside the main candidates. Reports how many entrapment "
            "peptides are identified at 1%%, 5%%, and 10%% FDR and writes "
            "entrapment_result.tsv. Requires --lcms-peptides or --msf."
        ),
    )

    # --- Strategy C: LC-MS/MS-guided candidates ---
    strat_c = parser.add_argument_group(
        "Strategy C — LC-MS/MS-guided candidates (optional)",
        description=(
            "When --lcms-peptides is provided, candidates are generated by "
            "digesting only the identified proteins and adding directly "
            "identified peptides, rather than the full FASTA (Strategy A). "
            "Min/max length and missed cleavages are inferred from the "
            "identified sequences unless overridden."
        ),
    )
    strat_c.add_argument(
        "--lcms-peptides",
        metavar="PATH",
        help="Peptide-level LC-MS/MS results file. Activates Strategy C.",
    )
    strat_c.add_argument(
        "--lcms-proteins",
        metavar="PATH",
        help=(
            "Protein-level results file (optional). Proteins are derived "
            "from the peptide table when omitted."
        ),
    )
    strat_c.add_argument(
        "--lcms-psms",
        metavar="PATH",
        help="PSM-level file for RT and intensity aggregation (optional).",
    )
    strat_c.add_argument(
        "--lcms-id-format",
        choices=("percolator", "mzidentml", "psm_utils", "msf", "ms2rescore"),
        default=None,
        help=(
            "Format of the LC-MS/MS identification files. "
            "Use 'msf' to read directly from a ProteomeDiscoverer .msf file "
            "(the same file passed to --msf can be reused). "
            "Use 'ms2rescore' to read ms2rescore .psms.tsv output directly."
        ),
    )
    strat_c.add_argument(
        "--psm-utils-reader",
        metavar="READER",
        default=None,
        help=(
            "psm_utils reader to use when --lcms-id-format is 'psm_utils'. "
            "Accepts a filetype key (e.g. 'maxquant', 'tsv', 'fragpipe') or "
            "a reader class name (e.g. 'MSMSReader', 'TSVReader'). "
            "When omitted, the reader is inferred from the file extension."
        ),
    )
    strat_c.add_argument(
        "--protein-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Protein FDR threshold for Strategy C protein filtering.",
    )
    strat_c.add_argument(
        "--peptide-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Peptide FDR threshold for Strategy C candidate inclusion.",
    )

    # --- Optional extras ---
    extras = parser.add_argument_group("optional extras")
    extras.add_argument(
        "--im2deep-calibration",
        choices=["linear", "spline", "finetune"],
        default=None,
        metavar="METHOD",
        help=(
            "CCS calibration strategy for IM2Deep predictions when observed CCS "
            "values are provided (via --feature-mzs). "
            "'linear' applies a global additive shift (default); "
            "'spline' fits a piecewise spline for non-linear bias correction; "
            "'finetune' adapts the neural network weights to the observed MALDI CCS "
            "via transfer learning (requires ≥ 100 single-candidate calibration peptides)."
        ),
    )
    extras.add_argument(
        "--spatial-features",
        metavar="PATH",
        help=(
            "Pre-computed per-feature spatial statistics TSV "
            "(fraction_detected, intensity_cv, spatial_autocorrelation, etc.)."
        ),
    )
    extras.add_argument(
        "--msf",
        metavar="PATH",
        help=(
            "ProteomeDiscoverer .msf file. Used as the LC-MS/MS identification source "
            "(format 'msf') when --lcms-peptides is not given."
        ),
    )
    # --- Output ---
    out_grp = parser.add_argument_group("output")
    out_grp.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output directory. Written files: ms1rescore_matches.tsv (every "
            "candidate) and ms1rescore_peptides.tsv (target peptides at 1%% FDR)."
        ),
    )
    out_grp.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging (default: INFO).",
    )

    return parser


# Parser options that are not config keys. extraction_ppm lives in the
# maldi_extraction subtable and is merged separately in main().
_CLI_ONLY = frozenset({"help", "version", "extraction_ppm"})


def _cli_config_source(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """The explicitly given CLI options, as the highest-priority config source.

    Every parser option outside ``_CLI_ONLY`` is a config key. store_true flags
    default to False (not None) in argparse, so "not given" and "explicitly False"
    look the same; they are converted False -> None so the config file wins when
    the flag is absent.
    """
    store_true = {a.dest for a in parser._actions if isinstance(a, argparse._StoreTrueAction)}
    return argparse.Namespace(**{
        a.dest: (True if getattr(args, a.dest) else None)
        if a.dest in store_true else getattr(args, a.dest)
        for a in parser._actions if a.dest not in _CLI_ONLY
    })


# Config keys whose rescore() parameter has a different name.
CONFIG_TO_RESCORE = {
    "fasta": "fasta_path",
    "extra_fasta": "extra_fasta_path",
    "lcms_peptides": "lcms_peptides_path",
    "lcms_proteins": "lcms_proteins_path",
    "lcms_psms": "lcms_psms_path",
    "use_protein_level_feats": "use_protein_level_features",
    "im2deep": "im2deep_kwargs",
}


def rescore_kwargs_from_config(cfg: dict) -> dict:
    """The ``rescore()`` keyword arguments that come straight from a merged config.

    Every config key (renamed through ``CONFIG_TO_RESCORE``) that names a
    ``rescore()`` parameter. Values computed at run time (MALDI arrays, digest
    settings, parsed IDs) are added by the caller.
    """
    from msi_picasso.pipeline import rescore

    params = inspect.signature(rescore).parameters
    renamed = {CONFIG_TO_RESCORE.get(k, k): v for k, v in cfg.items()}
    return {k: v for k, v in renamed.items() if k in params}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import json as _json
    from pathlib import Path as _Path
    from msi_picasso.config_parser import parse_configurations

    parser = build_parser()
    args = parser.parse_args()

    # --- Cascade config: defaults → config file → CLI args ---
    _config_sources = []
    if getattr(args, "config_file", None):
        _config_sources.append(args.config_file)

    _top_ns = _cli_config_source(parser, args)
    _config_sources.append(_top_ns)
    _ms1cfg = parse_configurations(_config_sources)["MSI-PICASSO"]

    # Extraction params: config defaults overridden by non-None CLI args.
    _extraction = dict(_ms1cfg.get("maldi_extraction", {}))
    if args.extraction_ppm is not None:
        _extraction["extraction_ppm"] = args.extraction_ppm

    # Convenience aliases from config
    output_dir = _ms1cfg["output_dir"]
    if Path(output_dir).exists() and not Path(output_dir).is_dir():
        parser.error(f"Output path {output_dir!r} exists and is not a directory.")
    elif not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)
    verbose = _ms1cfg["verbose"]

    # Write full merged config to output dir for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    _Path(output_dir, ".full_config.json").write_text(
        _json.dumps({"MSI-PICASSO": _ms1cfg}, indent=2, default=str)
    )

    _log_fh = open(_Path(output_dir, "run.log"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Third-party loggers that emit excessive DEBUG noise regardless of user intent.
    # `shap` is the worst of them: KernelExplainer logs its sampling weights per
    # explained candidate, which was 2394 lines — 45% of a kidney run's log — all of it
    # `subset_size = 1` and `weight_vector = array([...])` with nothing run-specific in it.
    for _noisy in ("numba", "numba.core", "imzy", "koyo", "shap",
                   "matplotlib", "matplotlib.font_manager", "matplotlib.pyplot",
                   "matplotlib.backends", "PIL"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # --- Validate argument combinations ---
    if _ms1cfg["digest"] and not _ms1cfg.get("fasta"):
        parser.error("--digest requires --fasta.")
    if not _ms1cfg.get("maldi_d"):
        parser.error("No MALDI input specified. Provide --maldi-d (or set maldi_d in the config file).")

    lcms_id_source = _ms1cfg.get("lcms_peptides")
    if not _ms1cfg["digest"] and not lcms_id_source and not _ms1cfg.get("msf"):
        parser.error(
            "No candidate source: provide --lcms-peptides (or --msf) to use "
            "LC-MS/MS identified peptides, or add --digest with --fasta to "
            "perform an in-silico digest."
        )

    # --- Load MALDI data ---
    spatial_features = None
    maldi_envelopes = None
    _ccs_arr: np.ndarray | None = None
    _ccs_source_mzs: np.ndarray | None = None  # mzs aligned with _ccs_arr; may differ from maldi_mzs
    _measured_pixel_mask: "np.ndarray | None" = None  # built when --coloc-measured-mask is set

    _maldi_raw_path: str = _ms1cfg["maldi_d"]
    _feature_mzs_path: str | None = _ms1cfg.get("feature_mzs")
    _maldi_query_raw = bool(_ms1cfg.get("maldi_query_raw"))
    if _maldi_query_raw:
        # Raw-query mode: defer extraction to rescore(), which queries the .d at
        # the candidate-derived m/z grid after candidate generation.
        logger.info(
            "Raw-query mode (--maldi-query-raw): MALDI ion images will be extracted "
            "from %s at candidate m/z values during candidate generation.",
            _maldi_raw_path,
        )
        logger.warning(
            "Raw-query mode is SUPERSEDED by feature-list extraction and is kept "
            "only to reproduce results predating it. It cannot produce negative "
            "evidence (every candidate is 'observed' by construction, F-012), it "
            "yields no target-decoy competition (0.00%% of features carry both a "
            "target and a decoy, F-010), and its candidates are one per feature so "
            "peptide-level FDR is a no-op. Use --feature-mzs with a peak list from "
            "the TIMSImaging fork for new work."
        )
        _tic_image = None
        _tic_n_features = None
        maldi_mzs = np.array([], dtype=np.float64)
        ion_images = None
        ion_image_mzs = None
        extra_ion_images = None
    else:
        from msi_picasso.maldi_extraction import extract_maldi_data

        logger.info(
            "Feature-list mode: ion images and spatial features are extracted at the "
            "m/z values supplied via --feature-mzs (from the TIMSImaging fork's 2D "
            "peak picking). LC-MS/MS identifications are used for candidate generation "
            "and prior features only, not for feature selection."
        )
        precomputed_mzs = None
        if _feature_mzs_path:
            logger.info(f"Loading pre-computed feature m/z values from {_feature_mzs_path}")
            try:
                precomputed_mzs, _ccs_arr, _ = _read_feature_mzs(_feature_mzs_path)
                _ccs_source_mzs = precomputed_mzs
            except Exception as exc:
                logger.error(f"Could not read --feature-mzs {_feature_mzs_path!r}: {exc}")
                sys.exit(1)
            logger.info(f"  {len(precomputed_mzs)} features loaded (skipping detection)")

        _keep_mask = None
        _keep_path = _ms1cfg.get("feature_mzs_keep")
        if _keep_path and precomputed_mzs is not None:
            try:
                _keep_mzs, _, _ = _read_feature_mzs(_keep_path)
            except Exception as exc:
                logger.error(f"Could not read --feature-mzs-keep {_keep_path!r}: {exc}")
                sys.exit(1)
            # Match by nearest within a hair's breadth rather than by equality: a
            # keep list written through a CSV can lose the last bit of a float
            # (measured: 17 of 7842 kidney m/z off by ~5e-13). The tolerance is
            # 1e-9 relative, about a thousandth of a ppm, far below any real peak
            # spacing, so it identifies the same peak and nothing else.
            _order = np.argsort(precomputed_mzs)
            _srt = precomputed_mzs[_order]
            _pos = np.clip(np.searchsorted(_srt, _keep_mzs), 1, len(_srt) - 1)
            _left = np.abs(_keep_mzs - _srt[_pos - 1])
            _right = np.abs(_keep_mzs - _srt[np.minimum(_pos, len(_srt) - 1)])
            _near = np.where(_left <= _right, _pos - 1, np.minimum(_pos, len(_srt) - 1))
            _dist = np.minimum(_left, _right)
            _hit = _dist <= 1e-9 * np.abs(_keep_mzs)
            _keep_mask = np.zeros(len(precomputed_mzs), dtype=bool)
            _keep_mask[_order[_near[_hit]]] = True
            _missing = int((~_hit).sum())
            if _missing:
                logger.error(
                    "  %d of %d m/z in --feature-mzs-keep are absent from --feature-mzs. "
                    "The two lists must come from the same peak-finding run.",
                    _missing, len(_keep_mzs),
                )
                sys.exit(1)
            logger.info(
                "  Keeping ion images for %d of %d peaks (%.1f%%) from %s; the "
                "on-tissue mask is still computed over all of them.",
                int(_keep_mask.sum()), len(precomputed_mzs),
                100.0 * _keep_mask.mean(), _keep_path,
            )

        logger.info(f"Extracting MALDI features from raw data: {_maldi_raw_path}")
        (maldi_mzs, ion_images, extra_ion_images, spatial_features, maldi_envelopes,
         _raw_pixel_coords, _tic_image, _tic_n_features) = extract_maldi_data(
            _maldi_raw_path,
            feature_mzs=precomputed_mzs,
            keep_mask=_keep_mask,
            extraction_ppm=_extraction["extraction_ppm"],
            output_npz=_ms1cfg.get("save_npz"),
            output_spatial_tsv=_ms1cfg.get("save_spatial"),
            output_dir=output_dir,
            verbose=verbose,
            save_ion_images=bool(_ms1cfg["save_ion_images"]),
        )
        ion_image_mzs = maldi_mzs if ion_images is not None else None
        logger.info(
            f"  {len(maldi_mzs)} features extracted"
            + (f", ion image shape: {ion_images.shape[1:]}" if ion_images is not None else "")
        )
        if bool(_ms1cfg.get("coloc_measured_mask", False)) and ion_images is not None:
            _xc, _yc = _raw_pixel_coords
            _H, _W = ion_images.shape[1], ion_images.shape[2]
            _measured_pixel_mask = np.zeros(_H * _W, dtype=bool)
            _measured_pixel_mask[np.asarray(_yc, dtype=np.int64) * _W + np.asarray(_xc, dtype=np.int64)] = True
            logger.info(f"  Measured-pixel mask: {int(_measured_pixel_mask.sum())}/{_measured_pixel_mask.size} pixels rastered")

    # --- Optional spatial features (explicit file overrides extracted ones) ---
    if _ms1cfg.get("spatial_features"):
        logger.info(f"Loading spatial features from {_ms1cfg['spatial_features']}")
        spatial_features = pd.read_csv(_ms1cfg["spatial_features"], sep="\t")

    # --- Resolve Strategy C source ---
    # If --lcms-peptides is not given but --msf is, use the MSF for Strategy C.
    lcms_peptides_path = _ms1cfg.get("lcms_peptides")
    lcms_id_format = _ms1cfg["lcms_id_format"]
    if lcms_peptides_path is None and _ms1cfg.get("msf") is not None:
        lcms_peptides_path = _ms1cfg.get("msf")
        lcms_id_format = "msf"
        logger.info(
            f"No --lcms-peptides provided; using --msf ({_ms1cfg['msf']}) "
            f"as Strategy C ID source (format='msf')."
        )

    # --- Resolve digest parameters ---
    lcms_ids = None
    if lcms_peptides_path:
        from msi_picasso.lcms_ids import parse_lcms_ids

        logger.info("Parsing LC-MS/MS identifications for Strategy C...")
        lcms_ids = parse_lcms_ids(
            proteins_path=_ms1cfg.get("lcms_proteins"),
            peptides_path=lcms_peptides_path,
            psms_path=_ms1cfg.get("lcms_psms"),
            protein_fdr=_ms1cfg["protein_fdr"],
            peptide_fdr=_ms1cfg["peptide_fdr"],
            format=lcms_id_format,
            psm_utils_reader=_ms1cfg.get("psm_utils_reader"),
        )
        if verbose:
            logger.debug("Writing parsed LC-MS/MS IDs to debug_lcms_ids.tsv")
            lcms_ids.peptides.to_csv(
                f"{output_dir}/4_debug_lcms_ids.tsv", sep="\t", index=False
            )
        min_length, max_length, missed_cleavages = _infer_digest_params(
            lcms_ids,
            missed_cleavages_override=_ms1cfg["missed_cleavages"],
            min_length_override=_ms1cfg["min_length"],
            max_length_override=_ms1cfg["max_length"],
        )
    else:
        min_length = _ms1cfg["min_length"]
        max_length = _ms1cfg["max_length"]
        missed_cleavages = _ms1cfg["missed_cleavages"]

    logger.info(
        f"Parameters extracted: min_length={min_length}, "
        f"max_length={max_length}, missed_cleavages={missed_cleavages}"
    )

    # --- Build observed CCS dict from loaded CCS array ---
    # extract_maldi_data drops zero-signal features, so maldi_mzs can be shorter than
    # the --feature-mzs list the CCS array is aligned with; re-index by m/z.
    observed_ccs: dict | None = None
    if _ccs_arr is not None:
        _ref_mzs = _ccs_source_mzs if _ccs_source_mzs is not None else maldi_mzs
        if _ref_mzs is not None and len(_ccs_arr) == len(_ref_mzs):
            from msi_picasso.utils import values_at_mz

            _ccs = values_at_mz(_ccs_arr, _ref_mzs, maldi_mzs)
            observed_ccs = {int(i): float(_ccs[i]) for i in np.flatnonzero(np.isfinite(_ccs))}
            if observed_ccs:
                logger.info(
                    f"  CCS values loaded for {len(observed_ccs)}/{len(maldi_mzs)} features"
                )
            else:
                logger.warning(
                    "  CCS array found but no m/z values matched maldi_mzs; "
                    "CCS features will be skipped"
                )

    # --- Load GT peptides (only relevant when debug is enabled) ---
    gt_peptides: list[str] | None = None
    _debug_gt_path = _ms1cfg.get("debug_gt")
    if verbose and _debug_gt_path:
        try:
            with open(_debug_gt_path) as _fh:
                gt_peptides = [line.strip() for line in _fh if line.strip()]
            logger.info("GT peptides loaded: %d from %s", len(gt_peptides), _debug_gt_path)
        except Exception as _exc:
            logger.warning("Could not read --debug-gt file %s: %s", _debug_gt_path, _exc)

    # --- Run pipeline ---
    from msi_picasso.pipeline import rescore

    logger.info("Starting MSI-PICASSO pipeline...")
    _kwargs = rescore_kwargs_from_config(_ms1cfg)
    _kwargs.update(
        maldi_mzs=maldi_mzs,
        ion_images=ion_images,
        ion_image_mzs=ion_image_mzs,
        extra_ion_images=extra_ion_images,
        spatial_features=spatial_features,
        maldi_envelopes=maldi_envelopes,
        maldi_query_raw=_maldi_query_raw,
        maldi_d_path=_maldi_raw_path,
        tdf_path=_maldi_raw_path,
        extraction_ppm=_extraction["extraction_ppm"],
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
        lcms_ids=lcms_ids,
        lcms_peptides_path=lcms_peptides_path,
        lcms_id_format=lcms_id_format,
        debug_dir=os.path.join(output_dir, "debug") if verbose else None,
        observed_ccs_per_feature=observed_ccs,
        gt_peptides=gt_peptides,
        coloc_measured_pixel_mask=_measured_pixel_mask,
        tic_image=_tic_image,
        tic_n_features=_tic_n_features,
        substitution_collision_filter=not _ms1cfg["substitution_no_collision_filter"],
    )
    result_df, _features_df = rescore(**_kwargs)

    # --- Write results ---
    logger.info(f"Writing results to {os.path.abspath(output_dir)}")
    if verbose:
        logger.debug("Writing complete result DataFrame to debug_result_df.tsv")
        result_df.to_csv(f"{output_dir}/5_debug_result_df.tsv", sep="\t", index=False)
    _write_results(result_df, output_dir)
    if gt_peptides and "is_tdc_winner" in result_df.columns:
        gt_set = set(gt_peptides)
        is_target = ~result_df["is_decoy"].astype(bool)
        winners = result_df[result_df["is_tdc_winner"] & is_target]
        logger.info(
            "%d/%d GT peptides are feature-level winners.",
            winners.drop_duplicates(subset=["peptide"])["peptide"].isin(gt_set).sum(),
            len(gt_set),
        )

        # Report GT recovery on the SAME population the reported ID count uses.
        # F-029 moved ID counting to peptide level but left this block at feature
        # level, so every log from E015 on printed the two on different footings.
        # amyloidosis E018 read 7/10 GT at 1% FDR against E016's 8/10 -- looking
        # like GT had been lost while the count rose, which PROGRESS.md sec.1 calls a
        # red flag for an over-optimistic FDR. At peptide level the same two runs
        # give 6/10 and 7/10: GT rose with the count. See PROGRESS.md F-034.
        if "is_peptide_winner" in result_df.columns:
            pep_col, q_col, level = "is_peptide_winner", "peptide_q_value", "peptide-level"
        else:
            pep_col, q_col, level = "is_tdc_winner", "q_value", "feature-level"
        reported = result_df[result_df[pep_col] & is_target]
        for alpha in (0.01, 0.05, 0.10):
            passing = reported[reported[q_col] <= alpha].drop_duplicates(subset=["peptide"])
            logger.info(
                "%d/%d GT peptides at %g%% FDR (%s, same population as the reported "
                "ID count).",
                passing["peptide"].isin(gt_set).sum(), len(gt_set), alpha * 100, level,
            )
    logger.info("Done.")


if __name__ == "__main__":
    main()
