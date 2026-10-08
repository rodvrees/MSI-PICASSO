# MSI-PICASSO — CLAUDE.md

**Scope of this document:** what the code *is* and what it must *never* do.
It contains no experimental results. Everything about what has been tried, what it
produced, and what to try next lives in **`/home/robbe/MALDI_MSI_score/PROGRESS.md`** —
read that first for project state. Keeping results out of here is deliberate: the previous
version of this file mixed findings from superseded decoy methods into the reference
material, and those findings were then applied to configurations where they did not hold.

Superseded material is archived in `/home/robbe/MALDI_MSI_score/docs/archive/`
(`CLAUDE_pre_3008.md`, `ARCHITECTURE.md`, `AUDIT.md`, `FIX_KIDNEY_DISCREPANCY.md`).

---

## Write so a new reader can follow it

This applies to this file, to PROGRESS.md, and to how results are explained in
conversation.

The reader is a competent scientist who has not been living inside this project. They know
mass spectrometry and proteomics. They do not know what *this project* means by "the noise
band", "the spread", "a draw", "the head of the ranking", or "the q-value floor". Those are
local shorthand, and every one of them was invented here.

Rules:

1. **Explain a term the first time it appears in a document**, in a clause, not a footnote.
   "The q-value floor (the best q-value the run can reach at all, no matter the threshold)
   was 0.0536."
2. **PROGRESS.md §0.1 is the glossary.** Add a term there when you coin one. If a term is
   not worth a glossary entry, it is not worth using.
3. **Prefer the plain word.** "Varies from run to run" beats "exhibits inter-replicate
   variance". "One result out of many possible ones" beats "a single draw". Keep the
   technical term where it is genuinely standard (q-value, FDR, target-decoy competition,
   PSM) and explain the ones that are not.
4. **Short sentences.** One claim each. Long sentences in this project usually mean a
   caveat has been folded into a finding; split them and state the caveat separately.
5. **Give the number and what it means.** "3.06x" is not a result on its own. "3.06 times
   larger than the biggest value random labels ever produced, so it is unlikely to be
   chance" is.
6. **No figurative language.** No "rides on", "collapses", "destroys", "the head is
   poisoned". These read as measurements when they are not. F-002 said the collision filter
   "destroys kidney"; the measured effect was roughly half the size that word implies.

The reason is concrete rather than stylistic. This document set has repeatedly had findings
misapplied because the wording was compact enough to be memorable and vague enough to be
wrong: F-004 carried an `mz_shuffle` fix into `substitution` configs, and the scope tags in
PROGRESS.md §0 exist because of it. Vague wording is how that happens.

---

## Purpose

`MSI-PICASSO` is a symmetric target-decoy rescoring package for MALDI-MSI MS1 data. It
takes MALDI data (a raw Bruker `.d`) plus an LC-MS/MS PSM table and produces FDR-controlled
MALDI peptide identifications by semi-supervised rescoring.

**Goal:** raise the number of confident IDs on the ground-truth datasets — amyloidosis,
her2, kidney (all TIMS, in-situ iprm-PASEF MS/MS ground truth), and `SC_Heeren` (no ion
mobility). Nothing is fixed: decoy method, scoring backend, and feature set are all in play.

**Why it was built this way:** the prior approach (ms2rescore "Approach B") used
ProteomeDiscoverer output for candidates and features, which leaked the label — `lcms_xcorr`,
a search-engine score, had AUC 0.993 against the PD target/decoy label. This package removes
all PD-derived scores; every feature is computed from the raw `.d` (and optionally raw
LC-MS/MS) under the symmetry invariant below.

---

## Critical invariants

Violating any of these silently corrupts the FDR. They are listed first because they are
the only part of this file that is genuinely load-bearing.

1. **No feature computation function takes `is_decoy`.** Targets and decoys receive features
   through identical code paths. This is enforced at the API level — grep for `is_decoy` in
   `maldi_features.py`, `feature_generator.py`, `lcms_evidence.py`, `utils.py` and
   `maldi_query.py` returns nothing.
2. **A decoy row's `feature_mz` must be the anchor that decoy was actually scored at** —
   for `substitution` the substituted peptide's [M+H]+. Never *another candidate's* anchor. Load-bearing for raw-query mode, which extracts each
   candidate's ion image at its own anchor.

   **`mz_shuffle` is the deliberate exception and must not be "fixed".** Its decoys are
   co-located on a target's feature by design, so a `mz_shuffle` decoy's `feature_mz` *is*
   its paired target's — measured at spread exactly 0 across 5077/5077 kidney and 3131/3131
   her2 pairs. That co-location is what restores the target-decoy competition F-010 found
   missing (F-021), and computing `ppm_error` against the decoy's own mass instead would
   make the null anti-conservative (F-019, and `generate_mz_shuffle_candidates`' docstring).
   In feature-list mode the anchor is the matched detected peak's m/z, which is likewise the
   candidate's own match.
3. **Composition features must stay OUT of the seed and out of the ranker.** Several decoy
   methods alter elemental composition, so composition features separate target from decoy
   as an artifact of decoy construction, with no spectral evidence involved. Seeding on them
   makes the FDR anti-conservative.

   Two separate mechanisms do the excluding and it is worth knowing which is which, because
   the wording here used to be wrong in a way PROGRESS.md then repeated. `_BEST_FEAT_SKIP`
   (`pipeline.py:320`, 11 entries) excludes columns from the **seed search only**.
   Eligibility for the **ranker** is set by membership of `MALDI_INTRINSIC_FEATURES`. `n_S`
   is in neither, which is what actually keeps it out — not `_BEST_FEAT_SKIP`, which does not
   contain it.

   **Anything derived from composition counts as a composition feature**, and the
   isotope-envelope family is derived from composition. F-036 measured `theo_isotope_kl`
   reading sulfur content at AUC 0.61-0.70 among targets alone, while `substitution` decoys
   carry twice the sulfur of targets — so it partially reads the label. F-039 replaced that
   family with averagine-referenced versions that do not.

   **This is not a property of the isotope family, and there is a check for it.** F-042 found
   four more ranker features doing the same thing at about a tenth of the size
   (`log_maldi_intensity_p90`, `protein_colocalization_weighted_max`, `spatial_gearys_c`,
   `adduct_colocalization_chca_mob`), and traced the cause to the decoy generator rather than
   to the features: `generate_substitution_candidates` draws the replacement residue uniformly
   over 18 amino acids, so 11.1% of substitutions add a sulfur residue against a natural
   frequency of 1.4-1.8%. **Fixed at the generator as of E024** by
   `substitution_residue_weighting = "target_frequency"` plus
   `substitution_preserve_sulfur = true`, which the checked-in configs set; both default off so
   earlier results reproduce. Removing the asymmetry gains identifications on kidney and her2
   but costs her2 both of its reachable confirmed ground-truth peptides, which are sulfur-free
   and were being flattered by the old ranking — read PROGRESS.md F-043 before changing either
   option. **Run `scripts/audit_composition_leak.py` when adding a feature to
   the ranker or changing the decoy generator.** It tests the whole chain — does the feature
   read composition, and does that come out as target/decoy separation — because reading
   composition alone is not enough to leak: the protein-level features read sulfur strongly
   and stay symmetric, since a decoy protein is the same size as its target protein.
4. **`is_decoy` must be cast to `bool` dtype** before returning a candidates frame.
   `pd.concat` with an empty frame yields `object`-dtype booleans, which break
   `~df["is_decoy"]` indexing downstream.
5. **Explicit falsy config values are honored** by `_merge` in `config_parser.py`
   (`config_parser.py:45`): any non-None value wins, including 0 and `false`; `None` only
   fills a key that does not exist yet. `--winner-percentile 0` is therefore not masked by
   the default of 0.02. The schema still rejects out-of-range values, so `matching_ppm` 0
   is allowed while a negative value raises.
6. **`maturin develop` must target the venv explicitly** (see Build below), or the extension
   installs into the base pyenv Python and the package silently falls back to slow Python
   paths.
7. **`has_oxidized_met` is really `has_methionine`** (`maldi_features.py:1346`). Plain-sequence
   candidates never carry `M[Oxidation]` annotations, so the literal definition is
   unreachable. Rename it or wire up modification detection; do not trust the name.

---

## Repository layout

The git repo is `MSI-PICASSO/` only. The parent `/home/robbe/MALDI_MSI_score/` is **not**
under version control, so `configs/`, `results/`, `scripts/`, `data/` and `docs/` are
untracked by design. Run provenance is instead recovered from the `.full_config.json` that
`cli.py` writes into every output directory.

```
MSI-PICASSO/
├── pyproject.toml              # testpaths = ["msi_picasso/tests"]
├── CLAUDE.md                   # this file
├── msi_picasso/
│   ├── cli.py             1475 # argparse CLI (`picasso` / `msi-picasso`), config cascade, MALDI loading
│   ├── config_parser.py    102 # _merge + jsonschema validation
│   ├── candidates.py      1319 # FASTA digest, substitution + mz_shuffle decoys, match_to_maldi_features
│   ├── lcms_ids.py         722 # parse LC-MS/MS IDs -> identified proteins + peptides
│   ├── maldi_extraction.py 730 # ion images + spatial feats at a GIVEN feature list (imzy)
│   ├── maldi_query.py      502 # raw-query mode + observed centroids/CCS (alphatims) + disk cache
│   ├── maldi_features.py  2302 # all MALDI-side features: mass accuracy, colocalization, isotope
│   ├── feature_generator.py 469# feature-group constants + compute_all_features
│   ├── pipeline.py        2595 # rescore() orchestrator, scoring backends, TDC q-values, PEP
│   ├── debug_viz.py       3277 # debug figures (--verbose)
│   ├── utils.py             95 # shared math (brainpy isotopes, compositions, values_at_mz)
│   ├── package_data/           # config_default.json + config_schema.json
│   └── tests/                  # 44 test modules
└── msi-picasso-rs/             # PyO3 + rayon extension, crate "MSI-PICASSO-rs"
    └── src/{lib,digest,ion_image,mob_coloc}.rs   # 166 + 54 + 125 + 297 lines
```

**The Rust crate imports as `ms1rescore_rs`** — the module name was never renamed with the
package. The peptide-mass, m/z-matching and ion-image call sites wrap it in
`try/except ImportError` with a Python fallback. The mobility-colocalization kernel
(`mob_coloc_features`) has no fallback: without the extension, Step 6c fails and
`rescore()` skips the `*_mob` features with a warning.

---

## Pipeline flow

1. **Entry**: `cli.main()` (`cli.py:1133`) → `parse_configurations(...)["MSI-PICASSO"]`.
2. **Config cascade** (`config_parser.py:58`): `package_data/config_default.json` → user
   JSON/TOML → explicit CLI args. `_cli_config_source` (`cli.py:1086`) turns every parser
   option into a config key, converting `store_true` flags `False → None` so argparse
   defaults do not clobber config-file values.
3. **MALDI input**: `--maldi-d` (a Bruker `.d`) is the only MALDI input.
   **`maldi_d` + `--feature-mzs <peak list>` is the current mode and the one new work
   must use.** The peak list comes from the TIMSImaging fork (see "Reading MALDI data").
   `maldi_query_raw=true` selects the superseded raw-query mode, which defers extraction
   into `rescore()` and warns at startup; it is kept only to reproduce results predating
   feature-list extraction. The CLI parses the LC-MS/MS IDs once and passes them to
   `rescore()` as `lcms_ids`.
4. **`rescore()`** (`pipeline.py:1460`):
   - Step 1 candidate generation; 1b extra FASTA; 1c decoy generation (`pipeline.py:1874`)
   - Raw-query extraction (`pipeline.py:1966`): `query_raw_maldi()` +
     `extract_observed_feature_stats_raw()`, then symmetric `ppm_error` recomputation from
     the observed centroids
   - Calibration-peptide selection for IM2Deep finetuning (`pipeline.py:2093`)
   - Step 6 `compute_all_features()`; then optional `drop_zero_signal`, CCS filter,
     mobility-filtered colocalization (6c), spatial-ranker features
   - Step 8 rescoring, winner selection, TDC q-values, PEP (`pipeline.py:2457`)
5. **Output**: `_write_results()` (`cli.py:155`), debug figures via `save_debug_figures()`.

The step numbers in the log skip 2-5 and 7. Those steps (LC-MS/MS evidence and a PSMList
build) were removed; the remaining numbers were kept so old and new logs line up.

### Extraction mode: feature list, not raw query

Every candidate used to be queried at its own m/z, and the window integrator returned the
weighted mean of whatever fell inside it — so an empty window returned the local noise and
**no candidate could fail to be observed** (measured no-observation rate 0.000, F-012).
Each candidate then owned its own `feature_idx`, so target and decoy never competed
(0.00% of features carried both, F-010).

Feature-list mode fixes both. A peak list is built once per dataset by the TIMSImaging
fork's 2D (m/z, 1/K0) picker; candidates are matched to it within `matching_ppm`;
unmatched candidates are dropped. Measured consequences:

- target-decoy competition rises from 0.00% to 19–25% of occupied features, **earned from
  the data** rather than imposed by a co-located decoy generator;
- `protein_coverage` becomes a real feature — F-013 had recorded it constant at 1.000 and
  dead weight, because in raw-query every candidate matched by construction;
- the ~40 minute per-candidate alphatims pass disappears (observed CCS comes from the peak
  list), replaced by a ~2 minute peak-finding pass.

**Matching is not evidence, and must not be treated as such.** Targets and decoys survive
feature matching at the same rate (67.7% against 67.9% on amyloidosis, 50.7% against 50.7% on
kidney) and their mass errors are indistinguishable at every tolerance from 1 to 10 ppm. That is
the decoys working correctly: a false identification is a wrong peptide whose mass matches a real
peak, which is what `substitution` models. **No tolerance change repairs it and none should be
attempted** — the ground truth needs up to 8.86 ppm on amyloidosis, so tightening loses real IDs.
All discrimination comes from the ion images. Expect any feature built on mass agreement or on
match counts to be symmetric, and measure it before building on it. See PROGRESS.md F-047.

**Two things change meaning and must not be forgotten:**

1. **A peptide matches several features** (2.1–4.8 on the three datasets, max 15), so a
   candidate row is a *peptide-feature match*, not a peptide. FDR is therefore computed at
   peptide level by `_peptide_level_qvalues` after winner selection — count IDs with
   `is_peptide_winner & ~is_decoy & peptide_q_value <= alpha`. Counting rows overstates by
   the multiplicity.
2. **`matching_ppm` must be non-zero.** The old baselines set it to 0 (exact), which in
   feature-list mode matches nothing at all. 10 ppm is the tightest value that keeps every
   reachable ground-truth peptide on all three datasets.

### The debug table records what the ranker saw

`13_debug_features.tsv` is the input to every offline analysis in PROGRESS.md, so it must
match the matrix the ranker was fitted on. `rescore()` log1p-transforms five heavy-tail
columns (`_HEAVY_TAIL_FEATURES`) before fitting; **that transform must stay above the debug
write**, and a regression test (`test_debug_table_fidelity.py`) asserts the source order. It
did not, once, and a refit from the table silently failed to reproduce the run it came from
(PROGRESS.md F-032). This is the same class of ordering bug as F-018.

### Scoring

One round: every candidate is scored → per-feature winner selection
(`_select_feature_winners`) keeps the top-scoring candidate per MALDI feature → TDC
q-values over the winners, on the score each winner was selected with. Winner selection is
the target-vs-decoy competition that defines the TDC population.

A second round (retraining on the winners only) used to exist and was always switched off
by `single_round = true`; it and its polynomial interaction terms were removed. `result_df`
holds `<model>_score_r1` for every candidate and `score`, `q_value`, `pep`,
`pep_q_value` and `peptide_q_value` for the winners.

**Protein-level features are size-residualized by default** (`protein_size_residualize`, on
since E025). Each size-driven one gains a `*_sizeresid` companion — its rank within a bin of the
protein's tryptic count — and the raw column is excluded. The seed allowlist is remapped to the
companion names, because every config seeds mostly on these features and a seed that hunts for a
missing column fails silently (F-044). A companion is **not** added when the config excludes its
raw form, or config exclusions would be bypassed. Turn it off with `--no-protein-size-residualize`;
every pre-E025 config pins it false so it reproduces. See PROGRESS.md F-045 and E025 — this buys
the 5% and 10% columns and costs the 1% column and confirmed recovery on two datasets.

**Ties in `_tdc_qvalues` go to the decoys, and must keep doing so.** The candidates frame is
every target followed by every decoy, so sorting on score alone inherits that order. When
scoring degenerates — a failed seed makes `_rescore_linear_once` return all zeros — a
row-order tiebreak puts every target above every decoy and the run reports nearly all of them
at q = 1/n_targets instead of failing. `np.lexsort((~is_decoy, -scores))` makes that case
conservative, and a warning fires when every score is identical. Real fitted scores have no
ties (zero on all three E024 datasets), so this changes no result that was not already
meaningless. See PROGRESS.md F-044.

---

## Reading MALDI data

`imzy.get_reader(d_path)` dispatches Bruker `.d` (TDF/TSF, bundled `libtimsdata.so`) and
`.imzML`. `extract_maldi_data()` is the single public entry, returning
`(feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes, pixel_coords,
tic_image, tic_n_features)`.

**This package no longer finds features.** `extract_maldi_data()` requires `feature_mzs`
and only extracts ion images and spatial statistics at that list; passing none is a
`ValueError`. Peak picking is done in 2D (m/z, 1/K0) by the **TIMSImaging fork** at
`/home/robbe/TIMSImaging` (branch `feat/msi-picasso-feature-finding`), which writes a
feature list consumed here via `--feature-mzs`. The interface is that file, not an import:
`_read_feature_mzs` (`cli.py:38`) reads m/z plus optional CCS and intensity from the
semicolon layout, and the fork has a round-trip test against this reader. The intensity
column is read but not used: the per-feature intensity features come from the extracted
ion images.

The former in-package detectors (`detect_features`, and `maldi_imzml.py`'s SCiLS-style
interval extraction with its deisotoping and mass-defect filtering) were **deleted**, not
deprecated, along with their 22 config knobs — a second, m/z-only implementation was worth
less than the confusion of having two.

Performance-relevant details:

- **`_extract_centroid_fast`** (`maldi_extraction.py:33`) — TSF only. Pre-converts feature
  m/z windows to raw spectral index windows once from the reference frame's calibration,
  then makes one DLL call per pixel. Accepts <5 ppm systematic calibration error. Without
  it, imzy's two-calls-per-pixel pattern is ~3 hours for 49 K pixels on a network filesystem.
- **`_extract_profile_fast_multi`** (`maldi_extraction.py:136`) — the in-RAM path.
  Extracts all six feature sets (main, M+1, M+2, Na, K, CHCA) in a **single**
  `spectra_iter()` pass, buffering 512 pixels at a time into the Rust
  `accumulate_profile_chunk` (rayon).
- **imzy writes its own caches**: an `.icache` npz beside each imzML holding per-spectrum
  `.ibd` byte offsets and coordinates (so opening a 133 MB imzML is an npz load, not an XML
  parse), and a `.icache/frame_index_cache.npz` inside each Bruker `.d`.
- **Ion images live in RAM.** The full `(n_features, H, W)` float32 array is extracted in a
  single `spectra_iter` pass for all six feature sets. Feature-list mode makes the array
  much larger than raw-query ever did (her2: 54 K features vs 5 K candidates, 42 GB), so
  check available RAM, and use `--feature-mzs-keep` to keep images only for peaks a
  candidate can match. The former memmap option (`--images-path`) was removed: it took
  ~43 hours on her2 and extracted only the main feature set.

### The `.d` is opened more than once

`imzy` exposes neither per-peak centroid m/z nor mobility, so raw-query mode opens the `.d`
a second time with `alphatims` (`maldi_query.py:295`) for observed peak centroids, observed
CCS, and the mobility peak-quality descriptors. Mobility colocalization
(`maldi_features.py`) streams the TDF a third time.

**This second pass dominates runtime** (roughly 40 minutes of a ~69 minute amyloidosis run,
streaming ~4e9 raw peaks) while producing only a handful of arrays of `len(query_mzs)`.

**Feature-list mode does not pay it at all.** Observed CCS comes from the feature list's own
CCS column and the peak-shape descriptors come from the resolved peak, so
`extract_observed_feature_stats_raw` is not called. The fork's whole peak-finding pass over
the same `.d` measured 135 s (kidney, 5% frame sampling, 4.6e9 peaks), against the ~40 min
this replaces. Both caches below therefore apply to raw-query mode only.

Two caches exist for it:

- **`raw_query_cache`** (`pipeline.py:1470`, logic at `1991`) — an in-process dict covering
  the *whole* extraction (`maldi_mzs`, `ion_images`, `extra_ion_images`, `spatial_features`,
  `maldi_envelopes`, `ccs_arr`, `centroid_arr`, `peak_quality`). Pass `None` to always
  extract, `{}` to extract once and populate, or a populated dict to reuse without touching
  the `.d`. Used by `scripts/grid_search.py`. Dies with the process.
- **`--raw-query-cache-dir`** (`raw_query_cache_dir`) — persists just the alphatims stats to
  an `.npz` keyed by a SHA-256 of the `.d` path, the **full** query m/z grid, and the window
  parameters. Survives across processes. Hashing the whole grid means a changed candidate
  set (different decoy method, digest, or FASTA) misses rather than silently reusing stale
  statistics. This is the cache to use for the experiment cycle; the ion images are only
  ~5 minutes of work but several GB of array, so they are deliberately *not* persisted.

Both are valid for exactly one reason: the candidate m/z grid is fixed by the digest plus
the decoy method, so it is constant across runs that vary only scoring parameters.

---

## Decoy generation

`decoy_method` selects the Step-1c generator. Two are supported. **This section describes
capability only**: for how each one has actually performed, see PROGRESS.md §4, where
every finding carries the configuration it was established under.

| method | what it does | preserves | notes |
|---|---|---|---|
| `substitution` | substitutes `substitution_n_residues` interior non-K/R residues, one decoy per unique target | length, cleavage sites | package default. **Changes elemental composition unless told not to**: see invariant 3. `--substitution-residue-weighting target_frequency` draws the replacement from the target residue frequency instead of uniformly, and `--substitution-preserve-sulfur` never substitutes Cys or Met in or out; together they take the decoy/target sulfur ratio from ~2x to ~1x at no cost in decoys (F-042, F-043). **Both default off so pre-E024 results reproduce; the checked-in configs set both.** Mass shift measured p10 24 Da, median 55 Da, p90 107 Da, max ~244 Da (F-036 as corrected). CCS features stay usable and it is compatible with `--match-ccs`. |
| `mz_shuffle` | derangement of the peptide→feature assignment (mass-sorted rotation) | sequence exactly | decoys are **co-located** with targets on identical ion images, so feature-quality features are exactly symmetric. **Do not combine with `--match-ccs`**: it would remove ~all decoys by design. Mobility-gated and predicted-CCS columns, and the own-mass isotope-envelope features, are auto-excluded (`_mz_shuffle_leaking_features`). |

Every method places decoys in a **separate protein namespace** (`DECOY_…`) so
protein-level features are computed within class and a decoy is never pooled with its
source target's protein. `--entrapment` additionally injects shuffled pseudo-target
peptides (`ENTRAPMENT_…`, from `generate_entrapment_from_lcms_ids`) and reports how many
pass at each FDR.

The other decoy methods (`shuffle`, `mz_shift`, `entrapment` as a decoy method,
`balanced_shuffle`, `paired_shuffle`) were removed. Both remaining methods put each decoy
on a real MALDI feature, so `use_spatial_ranker_features` is no longer gated by decoy
method.

---

## Scoring backends

`--model` accepts **`{lda, svm, rbf_svm}`**. `_estimator_factory` (`pipeline.py:1082`)
holds the final pipeline step of each; all three go through `_rescore_linear`.

| model | estimator | notes |
|---|---|---|
| `lda` | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])` | package default; importances are `coef_[0]` |
| `svm` | `sklearn.svm.LinearSVC` | adds no dependency |
| `rbf_svm` | `sklearn.svm.SVC(kernel="rbf")` | nonlinear; no `coef_`, so importances are reported as permutation importance (below). `rbf_svm_gamma` accepts `"scale"`/`"auto"` or a float. Training is O(N²). |

The `qda` and `gbt` backends were removed.

All backends share the same semi-supervised loop: seed → pseudo-label iteration → winner
selection → TDC.

**The loop does not converge, and a single fit's reported count is a draw rather than a
measurement.** It is a chaotic map from the seed labels: the CV partition alone
(`_make_fold_ids`, `random_state`) moves the head of the ranking enough to change the
reported count several-fold on the low-count datasets. `model_repeats` (default 1, off)
averages the standardised scores of that many independent replicate fits, each with its own
partition, via the `_rescore_linear` wrapper around `_rescore_linear_once`. Averaging
*inside* a single trajectory was tried and does not converge — see PROGRESS.md F-030/F-031.
`model_repeats=1` is exactly the pre-existing single fit, so every earlier result reproduces.
Reported importances come from the first replicate; they describe one fitted model.

**Out-of-fold scoring.** `_cv_semisup_scores` (default `cv_folds=3`, stratified by
`is_decoy`) scores every candidate with a model trained on the other folds. Feature
importances come from a model fit on everything, but the FDR scores are strictly
out-of-fold.

**Three different per-feature numbers are reported, and they are not the same quantity.**
`17_debug_<model>_importances_r1.tsv` holds two of them.

| column | what it is | which backends |
|---|---|---|
| `importance` | `coef_[0]` | `lda`, `svm` |
| `importance` | permutation importance: `1 - spearman(score with that column shuffled, the unshuffled score)`, 5 shuffles, rows subsampled to 5000. 0 means the model does not use the feature (`_permutation_importance`, `pipeline.py:682`) | `rbf_svm` |
| `structure_coef` | Pearson r between the scaled feature and the score | all |

A structure coefficient is a *correlation*, not an attribution: a feature can correlate
with the score without contributing to it and contribute without correlating. The two
disagree in practice (PROGRESS.md F-041), so read both: `importance` says how much the
model uses a feature, `structure_coef` says in which direction. Permutation importance has
no sign — kidney's top feature carries three quarters of the ranking with a structure
coefficient of −0.53, meaning a high value pushes a candidate *down*.

**Neither number says a feature separates targets from decoys, and on current runs the top ones
do not.** E025's highest-importance feature has a target/decoy AUC of 0.476 while the best
separators carry importance 0.012 (PROGRESS.md F-046). Permutation importance is also sensitive
to a feature's entropy: any transform that spreads a coarse feature — ranking, binning, quantile
mapping — raises its reported importance without adding information, which is how
`protein_coverage_sizeresid` reached 0.292 from a variable with a 30% point mass at 1.0. **Read
the importance table next to the target/decoy AUC, never on its own.**

Permutation importance is averaged over the first `_PERM_IMPORTANCE_REPLICATES` (5)
replicate fits rather than all `model_repeats`, because one replicate's ranking is not
reproducible on kidney (F-041) and a full 20 would add 23 minutes to an amyloidosis run.
Coefficients and structure coefficients still come from replicate 0 only.

**SHAP works for every backend** (`debug_pfm_explanations`, `--verbose`): `LinearExplainer`
where there is a `coef_`, `KernelExplainer` on `decision_function` otherwise. The kernel
path is affordable only because it runs on the reported candidates — at most `max_targets`
of them, 1.3 s each on the E021 models — and not on the ~12 K candidate rows.

**Seed** — `_find_best_feature_labels` (`pipeline.py:391`) sweeps each feature and
both ranking directions, counting targets at q ≤ `train_fdr`. Sub-ULP random noise breaks
ties so row order cannot bias the result. If the best single feature yields fewer than
`min_seed_positives` targets it escalates to pairwise sums/differences on standardised
columns, then to a depth-3 `DecisionTreeClassifier`. Columns in `_BEST_FEAT_SKIP`
(composition and ionisation features) are excluded throughout — invariant 3.

Fallback chain when that yields nothing: `ppm_error_abs < init_ppm_threshold` OR
`n_candidates == 1`; then the top `r1_seed_percentile` of targets by `ppm_error_abs`.

`train_fdr_escalate` (off by default) retries the seed search and each pseudo-label update
at thresholds raised in steps of 0.005 when the configured one yields no positives.

There is no post-scoring reweighting. The additive LC-MS/MS and spatial log-priors were
removed; every config had both weights at 0, so for every earlier run `q_value` equals the
old `reweighted_q_value`. Storey π₀ correction and the Percolator-RESET decoy split
(`decoy_split`) were removed as well.

---

## Feature groups

Defined in `feature_generator.py`; import as `from msi_picasso.feature_generator import ...`.

| constant | line | in the ranker? |
|---|---|---|
| `MALDI_INTRINSIC_FEATURES` | 48 | yes, by default |
| `SIZE_DRIVEN_PROTEIN_FEATURES` | 112 | replaced by `*_sizeresid` companions (F-045) |
| `PROTEIN_LEVEL_FEATURES` | 192 | opt-in `--use-protein-level-feats` |
| `COSINE_COLOCALIZATION_FEATURES` | 225 | opt-in `--cosine-coloc` |
| `SPATIAL_RANKER_FEATURES` | 235 | opt-in `--use-spatial-ranker-features` |
| `MOB_QUALITY_FEATURES` | 255 | yes, when present (raw-query + ion mobility) |
| `MZ_SHUFFLE_MASSNORM_ISOTOPE_FEATURES` | 272 | `mz_shuffle` only |
| `MAIN_FEATURES` | 284 | subset selector |
| `FEATURE_NAN_FILL` | 327 | per-feature NaN sentinels |

The region-colocalization groups (`--region-coloc`, `--within-region-coloc`) were removed.

`MOB_QUALITY_FEATURES` = `mob_2d_concentration`, `mob_k0_spread`, `mob_mz_spread_ppm`,
`mob_peak_snr` — intrinsic joint (m/z, intensity, 1/K0) peak-quality descriptors, available
only when the dataset has a TIMS dimension.

**LC-MS/MS ID-derived columns stay out of the ranker.** `lcms_q_value`, `lcms_pep`,
`lcms_score`, `n_psms` and `lcms_intensity` are populated on the candidates frame by
Strategy C but are in no ranker group: they would give confirmed targets different
treatment from decoys, breaking TDC symmetry.

**Why `PROTEIN_LEVEL_FEATURES` are opt-in:** they aggregate over all candidates sharing a
protein, and are only valid because decoys occupy a separate protein namespace. Even so they
can interact subtly with the decoy model.

Compose the active set via config rather than by editing the module:

```toml
[MSI-PICASSO]
features_preset  = "all"          # "all" | "main"
features_exclude = ["peptide_length", "adduct_colocalization_chca"]
```

---

## Candidate generation

**Strategy C (current)** — activated by passing `lcms_peptides_path`. Candidates are the
in-silico digest of identified proteins ∪ the directly identified LC-MS/MS peptides. The
`source` column records the origin: `protein_digest`, `lcms_confirmed`, or `decoy`.
Without `--digest` (no FASTA) all confirmed peptides are novel, so decoys are built from a
**concatenated pseudo-protein** — all target sequences concatenated, shuffled once, then
re-digested — because a per-peptide shuffle would give decoys identical elemental
composition and make isotope features non-discriminative.

**Strategy A (legacy)** — `digest_fasta()` over a whole FASTA with K/R-preserving shuffled
decoys. Used when `rescore()` gets no `lcms_peptides_path`. Add `--digest` to combine it
with Strategy C.

`protein_coverage` counts distinct observed *peptides* over the **true full tryptic digest
count** (overridden in `pipeline.py` from `peptide_db` before Step 6). Both halves matter:
the earlier `protein_n_features / candidate_pool_count` form pinned every decoy protein to
exactly 1.0 and leaked the label.

### LC-MS/MS ID formats

`lcms_id_format` ∈ `percolator` (default), `mzidentml`, `psm_utils`, `msf`, `ms2rescore`.
With `psm_utils`, `psm_utils_reader` picks the concrete reader (`fragpipe`,
`proteome_discoverer`, `tsv`, …). Accessions are normalised (`sp|P12345|GENE_HUMAN` →
`P12345`) before comparison with the FASTA; a <50% match rate warns of a database mismatch.
FragPipe reports `Retention` in **seconds** — `_parse_psm_utils` divides by 60 when the
median exceeds 200.

---

## Configuration

Priority, lowest to highest: `package_data/config_default.json` → `--config-file`
(JSON/TOML) → explicit CLI arguments (`None` never overrides). The merged config is written
to `<output_dir>/.full_config.json` at the start of every run — **this is the run provenance
record**, and `scripts/scoreboard.py --diff` reads it.

The TOML table name is `[MSI-PICASSO]`, with `[MSI-PICASSO.maldi_extraction]` and
`[MSI-PICASSO.im2deep]` subtables.

Package defaults worth knowing, because the checked-in configs override all of them:

| key | package default | current configs use |
|---|---|---|
| `model` | `lda` | `rbf_svm` |
| `decoy_method` | `substitution` | `substitution` |
| `substitution_n_residues` | `1` | `2` |
| `train_fdr` | `0.05` | `0.1` (amyloidosis) / `0.3` (her2, kidney) |
| `init_ppm_threshold` | `2.0` (`rescore()` signature says `5.0`) | `10.0` / `5.0` |
| `min_seed_positives` | `50` | `125` / `20` |
| `matching_ppm` | `20.0` | `10` (feature-list) / `0` (old raw-query configs) |

### Adding a configurable parameter

1. `package_data/config_default.json`: add the key with its default.
2. `package_data/config_schema.json`: add the type (use `["type", "null"]` to allow a CLI
   `None` passthrough). The schema rejects unknown keys, so this step is mandatory.
3. `cli.py`: add `--param-name` with `default=None` (or `action="store_true"`). Every
   parser option becomes a config key automatically (`_cli_config_source`);
   `test_cli_config.py` fails if the key is missing from the defaults.
4. `pipeline.py`: add it to the `rescore()` signature. `rescore_kwargs_from_config`
   passes every config key that names a `rescore()` parameter, so no call site changes.
   If the config key and the parameter name differ, add the pair to `CONFIG_TO_RESCORE`.
5. Add a test in `tests/test_config_parser.py`.

---

## Environment, build, tests

```bash
# interpreter: the bare pyenv python does NOT have the dependencies
/home/robbe/.pyenv/versions/MSIscore/bin/python

pip install -e MSI-PICASSO/            # or "MSI-PICASSO/[timstof]" for Bruker .d support
```

Sibling checkouts `ms2rescore/`, `ms2pip/`, `psm_utils/`, `IM2Deep/`, `ms2rescore-rs/` are
upstream tools on custom branches, all installed editable into the same env.

**Rust extension** (invariant 6):

```bash
cd MSI-PICASSO/msi-picasso-rs
VIRTUAL_ENV=/home/robbe/.pyenv/versions/3.11.11/envs/MSIscore \
  /home/robbe/.pyenv/versions/3.11.11/envs/MSIscore/bin/maturin develop --release
```

`target/` reaches 1–2 GB; `rm -rf` it if disk is tight, it rebuilds.

**Tests** — `pytest` from `MSI-PICASSO/`, `testpaths = ["msi_picasso/tests"]`.
The suite passes in full (425 passed, 1 skipped on 2026-10-07); treat a red run as a
regression.

---

## Scripts

In `/home/robbe/MALDI_MSI_score/scripts/`:

| script | purpose |
|---|---|
| `scoreboard.py` | scrape all `results/*/*/run.log` into one comparison table; `--markdown` for a PROGRESS.md row, `--diff A B` for a settings diff between two runs |
| `validate_results.py` | biological validation: marker recovery, GT recovery, LC-MS concordance. Counts on the reported population (peptide-level since F-029) |
| `diagnose_gt.py` | **STALE — does not run on any current result.** LDA-only and two-round: reads `17_debug_lda_*`, `lda_score_r2` and `reweighted_*`, which no current run writes. Its coefficient-attribution analysis is linear-model-specific, so reviving it needs a design decision first, not a rename. |
| `grid_search.py` / `analyze_grid_search.py` | parameter sweep (reuses `raw_query_cache`) and its sensitivity analysis. Its objective counts on the reported population — keep it that way or the sweep optimises something the scoreboard does not show |
| `ablation_svm.py` / `ablation_lda.py` | feature ablation, single round, `lda` or `svm` only (any other configured model runs as `lda`). `ablation_lda.py` still counts at feature level (H-code-1, not fixed — it is LDA-era and unused) |
| `audit_coloc_leak.py` | per-colocalization-column target/decoy AUC and abundance-leak check — run before promoting a coloc feature into the ranker |
| `audit_match_information.py` | per-peptide best `|ppm|` against the matched peaks, targets vs decoys vs ground truth, by tolerance. Answers whether mass agreement can discriminate (it cannot, PROGRESS.md F-047) and what tolerance the ground truth needs. `RUN=E025 python scripts/audit_match_information.py` |
| `audit_ccs_matching.py` | sizes what a CCS window would do to candidate multiplicity and to the target peptides, at any peak list. Reads the run's own predicted CCS and asserts the run's `im2deep_abs_delta_ccs_pct` comes back out of the join before reasoning from it. `--peaklist` sizes a list that has no run yet, rebuilding candidates through `target_rows` (F-049); decoy sequences depend on the peak list, so that mode is targets only. `python scripts/audit_ccs_matching.py kidney results/kidney/E025 [--peaklist peaklists/kidney_region4_minreg1.csv]` (PROGRESS.md F-050) |
| `make_peaklist.py` / `prefilter_peaklist.py` | regenerate a TIMSImaging-fork peak list at a chosen `min_regions`, and derive the keep list of peaks a candidate can match. **Validate a new list by filtering it to `n_regions >= 2` and diffing against the checked-in one**, and a new keep list with `--against-run` once a run exists — the older `--verify` passed while a list was missing 61 of a run's peaks (F-049, F-050) |
| `audit_protein_size.py` | asks whether the protein-level features measure protein presence or protein size, by re-ranking each within bins of the protein's tryptic count and re-reading its target/decoy and ground-truth AUCs (PROGRESS.md, RD's objection of 2026-09-11). `RUN=E024 python scripts/audit_protein_size.py` |
| `audit_composition_leak.py` | screens every ranker feature for composition dependence and reports how much of each one's target/decoy separation the composition asymmetry accounts for (PROGRESS.md F-042). **Run it whenever a feature is added to the ranker or the decoy generator changes.** `RUN=E021 python scripts/audit_composition_leak.py`; `--self-check` exercises the decoy pairing |
| `seed_permutation_test.py` | the F-020 label-permutation test on the seed search, over several independent permutation sets — reads the run's own ranker feature list and reproduces its reported seed. `--protein-block` (F-051) permutes at the real/decoy protein-pair level instead of per row. `--rollup {none,protein,feature_mz}` (F-052/F-053) chooses the unit `_find_best_feature_labels` counts passes at before comparing to that null. `--features <list>` overrides the run's own `seed_features` allowlist, to test a candidate pool that has no run yet |
| `screen_seed_features.py` | per-feature triage for seed-allowlist candidates (H-fdr-11 option 3, F-054): target/decoy AUC and composition-leak share (both reused from `audit_composition_leak.py`) plus a block ratio (F-051's per-protein diagnostic, generalized to every feature) for every ranker feature a run used. Reads a candidate pool for `seed_permutation_test.py --features`, it does not itself decide whether that pool clears the noise band. `python scripts/screen_seed_features.py amyloidosis kidney her2 --run E025` |
| `replicate_spread.py` | refits a past run's round 1 under N CV partitions and reports the spread of its ID counts — reads the run's own `.full_config.json`; run this before quoting or comparing any single-fit count |
| `refit_harness.py` | shared helpers for refitting a finished run's scoring step offline with one thing changed: `load`, `fit`, `rollup`, `stats`, `final_report_permutation`. Reproduces a run's reported counts exactly when nothing is varied — if it does not, stop and find out why. Always pass `model_repeats` |
| `compare_backends.py` | which scoring backend, at what cost in IDs (PROGRESS.md F-040). `RUN=E021 python scripts/compare_backends.py` |
| `importance_attribution.py` | permutation importance against the structure coefficients, and whether either is stable across replicate fits (PROGRESS.md F-041). `RUN=E021 REPS=6 python scripts/importance_attribution.py` |
| `lcms_seed_test.py` | H-seed-1: three-way seed comparison (the run's own default seed search vs an LC-MS/MS-confidence seed vs a same-size random control) via `_rescore_linear`'s `seed_mask` — no `pipeline.py` change needed. `--seed-mode {unique,confidence-only}` (F-059/F-060: literal peak-match uniqueness silently selects for depressed `protein_colocalization_top5`; confidence-only + a `--tie-break` disambiguates per peptide instead), `--pep-percentile`/`--pep-max`, `--init-fdr`/`--train-fdr`/`--cv-mode` sweeps, `--model-repeats`. `python scripts/lcms_seed_test.py amyloidosis kidney her2` |
| `trace_seed_iterations.py` | reimplements `_rescore_linear_once`'s self-training loop from imported pipeline.py helpers (including `_find_best_feature_labels_escalating` for `--seed-mode default`, the pipeline's own seed search) to report every iteration's pseudo-positive count, q-floor and GT recovery, and (F-060) the per-CV-fold-seed death rate a converged-endpoint-only view like `lcms_seed_test.py` cannot see. `--summary-only` for a fold-seed sweep; `--average N` (H-seed-2, F-061/F-062/F-063) reproduces `_rescore_linear`'s `model_repeats=N` averaging exactly, then recomputes it with degenerate replicates (`IDs@5%==0`) dropped, reporting both; `--model {rbf_svm,linear_svm}` compares backend stability on the same seed. `python scripts/trace_seed_iterations.py amyloidosis --seed-mode confidence-only --average 20` |
| `envelope_qc.py` | isotope-envelope QC |
| `visualize_ms1rescore_features.py` | per-feature, per-candidate target/decoy visualisation. **Its LC-MS/MS evidence functions import `msi_picasso.lcms_evidence`, which was removed**, so those code paths fail at runtime |

One-off analyses are parked in `scripts/archive/` (including `audit_null_model.py`,
`prototype_peaklist_query.py`, `sweep_peak_stringency.py` and `two_feature_model.py`).
`optimize_maldi_params.py` still imports the deleted `maldi_imzml` inside one function.

---

## Running

```bash
picasso -c /home/robbe/MALDI_MSI_score/configs/amyloidosis_substitution.toml
picasso -c /home/robbe/MALDI_MSI_score/configs/her2_test.toml
picasso -c /home/robbe/MALDI_MSI_score/configs/kidney_test.toml
```

For an experiment, override the output directory and reuse the extraction cache:

```bash
picasso -c configs/kidney_test.toml \
        --output-dir results/kidney/E007/ \
        --raw-query-cache-dir .rawquery_cache
```

Convention: one experiment = one ID shared across branch (`exp/E007-<slug>`), config
(`configs/kidney_E007.toml`), results dir (`results/kidney/E007/`), and PROGRESS.md entry.
