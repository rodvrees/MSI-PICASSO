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
   for `substitution` the substituted peptide's [M+H]+, for `mz_shift` the shifted m/z.
   Never *another candidate's* anchor. Load-bearing for raw-query mode, which extracts each
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
   (`pipeline.py:488`, 11 entries) excludes columns from the **seed search only**.
   Eligibility for the **ranker** is set by membership of `MALDI_INTRINSIC_FEATURES`. `n_S`
   is in neither, which is what actually keeps it out — not `_BEST_FEAT_SKIP`, which does not
   contain it.

   **Anything derived from composition counts as a composition feature**, and the
   isotope-envelope family is derived from composition. F-036 measured `theo_isotope_kl`
   reading sulfur content at AUC 0.61-0.70 among targets alone, while `substitution` decoys
   carry twice the sulfur of targets — so it partially reads the label. Those features are
   currently in the ranker; see PROGRESS.md F-036 and H-decoy-13 before adding more of the
   same shape.
4. **`is_decoy` must be cast to `bool` dtype** before returning a candidates frame.
   `pd.concat` with an empty frame yields `object`-dtype booleans, which break
   `~df["is_decoy"]` indexing downstream.
5. **Explicit falsy config values are honored** via `_apply_explicit_overrides` in
   `config_parser.py`. `--matching-ppm 0` means exact matching and is not masked by the
   default of 20.0. `cascade_config`'s own merge rule would drop it.
6. **`maturin develop` must target the venv explicitly** (see Build below), or the extension
   installs into the base pyenv Python and the package silently falls back to slow Python
   paths.
7. **`has_oxidized_met` is really `has_methionine`** (`maldi_features.py:1005`). Plain-sequence
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
│   ├── cli.py             1807 # argparse CLI (`picasso` / `msi-picasso`), MALDI input dispatch
│   ├── config_parser.py    144 # cascade_config merge + jsonschema validation
│   ├── candidates.py      1987 # FASTA digest, all decoy generators, match_to_maldi_features
│   ├── lcms_ids.py         722 # parse LC-MS/MS IDs -> identified proteins + peptides
│   ├── lcms_evidence.py    965 # raw LC-MS/MS evidence features (MS2PIP, DeepLC-anchored MS1)
│   ├── maldi_extraction.py 976 # ion images + spatial feats at a GIVEN feature list (imzy)
│   ├── maldi_query.py      465 # raw-query mode + observed centroids/CCS (alphatims) + disk cache
│   ├── maldi_features.py  2657 # all MALDI-side features: mass accuracy, colocalization, isotope
│   ├── feature_generator.py 591# feature-group constants + compute_all_features
│   ├── pipeline.py        3324 # rescore() orchestrator, scoring backends, TDC q-values, PEP
│   ├── debug_viz.py       4049 # debug figures (--verbose)
│   ├── utils.py             99 # shared math (brainpy isotopes, spectral angle, mass constants)
│   ├── package_data/           # config_default.json + config_schema.json
│   └── tests/                  # 41 test modules
└── msi-picasso-rs/             # PyO3 + rayon extension, crate "MSI-PICASSO-rs"
    └── src/{lib,digest,features,ion_image,isotope,maldi_isotope,mob_coloc,spectral,xic}.rs
```

**The Rust crate imports as `ms1rescore_rs`** — the module name was never renamed with the
package. Every call site wraps it in `try/except ImportError` with a Python fallback.

---

## Pipeline flow

1. **Entry** — `cli.main()` (`cli.py:1458`) → `parse_configurations(...)["MSI-PICASSO"]`.
2. **Config cascade** (`config_parser.py:82`) — `package_data/config_default.json` → user
   JSON/TOML → explicit CLI args. `store_true` flags are converted `False → None`
   (`cli.py:1479`) so argparse defaults do not clobber config-file values.
3. **MALDI input dispatch** (`cli.py:1613`), mutually exclusive:
   `maldi_npz | maldi_mzs | maldi_raw | maldi_d`.
   **`maldi_d` + `--feature-mzs <peak list>` is the current mode and the one new work
   must use.** The peak list comes from the TIMSImaging fork (see "Reading MALDI data").
   `maldi_query_raw=true` selects the superseded raw-query mode, which defers extraction
   into `rescore()` and warns at startup; it is kept only to reproduce results predating
   feature-list extraction.
4. **`rescore()`** (`pipeline.py:1555`):
   - Step 1 candidate generation; 1b extra FASTA; 1c decoy generation (`pipeline.py:2093`)
   - Raw-query extraction (`pipeline.py:2277`) — `query_raw_maldi()` +
     `extract_observed_feature_stats_raw()`, then symmetric `ppm_error` recomputation from
     the observed centroids
   - Calibration-peptide selection for DeepLC / IM2Deep finetuning (`pipeline.py:2412`)
   - Steps 2–5 LC-MS/MS branch, skipped when no mzML or `lcms_prior_weight == 0`
   - Step 6 `compute_all_features()`; then optional `drop_zero_signal`, CCS filter,
     mobility-filtered colocalization, spatial-ranker features
   - Step 8 build `PSMList`; Step 9 rescoring, winner selection, TDC q-values, PEP
5. **Output** — `_write_results()` (`cli.py:220`), debug figures via `save_debug_figures()`.

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

### Two-pass scoring

Round 1 scores all candidates → per-feature winner selection (`_select_feature_winners`)
keeps the top-scoring candidate per MALDI m/z → Round 2 retrains on winners only → TDC
q-values over winners.

`--single-round` skips Round 2 only. **Winner selection still runs**, so the
target-vs-decoy competition that defines the TDC population is unchanged and the FDR
semantics are identical; only the final discriminant refit is dropped.

---

## Reading MALDI data

`imzy.get_reader(d_path)` dispatches Bruker `.d` (TDF/TSF, bundled `libtimsdata.so`) and
`.imzML`. `extract_maldi_data()` is the single public entry, returning
`(feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes)`.

**This package no longer finds features.** `extract_maldi_data()` requires `feature_mzs`
and only extracts ion images and spatial statistics at that list; passing none is a
`ValueError`. Peak picking is done in 2D (m/z, 1/K0) by the **TIMSImaging fork** at
`/home/robbe/TIMSImaging` (branch `feat/msi-picasso-feature-finding`), which writes a
feature list consumed here via `--feature-mzs`. The interface is that file, not an import:
`_read_feature_mzs` (`cli.py:37`) reads m/z plus optional CCS and intensity from the
semicolon layout, and the fork has a round-trip test against this reader.

The former in-package detectors (`detect_features`, and `maldi_imzml.py`'s SCiLS-style
interval extraction with its deisotoping and mass-defect filtering) were **deleted**, not
deprecated, along with their 22 config knobs — a second, m/z-only implementation was worth
less than the confusion of having two.

Performance-relevant details:

- **`_extract_centroid_fast`** (`maldi_extraction.py:193`) — TSF only. Pre-converts feature
  m/z windows to raw spectral index windows once from the reference frame's calibration,
  then makes one DLL call per pixel. Accepts <5 ppm systematic calibration error. Without
  it, imzy's two-calls-per-pixel pattern is ~3 hours for 49 K pixels on a network filesystem.
- **`_extract_profile_fast_multi`** (`maldi_extraction.py:351`) — the default in-RAM path.
  Extracts all six feature sets (main, M+1, M+2, Na, K, CHCA) in a **single**
  `spectra_iter()` pass, buffering 512 pixels at a time into the Rust
  `accumulate_profile_chunk` (rayon).
- **imzy writes its own caches**: an `.icache` npz beside each imzML holding per-spectrum
  `.ibd` byte offsets and coordinates (so opening a 133 MB imzML is an npz load, not an XML
  parse), and a `.icache/frame_index_cache.npz` inside each Bruker `.d`.
- **RAM vs memmap** — by default the full `(n_features, H, W)` float32 array lives in RAM,
  extracted in a single `spectra_iter` pass for all six feature sets. `--images-path`
  switches to a `np.memmap` written in `image_batch_size` batches, and is a **last
  resort, not a drop-in**: it calls `reader.get_ion_images()` once per batch (543 full
  passes over her2's 52 K spectra at the default batch size — measured 4m47s per batch,
  ~43 hours, against ~10 minutes in RAM) and it extracts **only the main feature set**, so
  every isotope- and adduct-colocalization feature silently goes missing. It warns at
  runtime. Raise `image_batch_size` sharply if you must use it. Feature-list mode makes the
  array much larger than raw-query ever did (her2: 54 K features vs 5 K candidates, 42 GB),
  so check available RAM rather than reaching for this.

### The `.d` is opened more than once

`imzy` exposes neither per-peak centroid m/z nor mobility, so raw-query mode opens the `.d`
a second time with `alphatims` (`maldi_query.py:197`) for observed peak centroids, observed
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

- **`raw_query_cache`** (`pipeline.py:1566`, logic at `2292`) — an in-process dict covering
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

`decoy_method` selects the Step-1c generator. All five are supported and all remain
candidates for improving results. **This section describes capability only** — for how each
one has actually performed, see PROGRESS.md §4, where every finding carries the
configuration it was established under.

| method | what it does | preserves | notes |
|---|---|---|---|
| `substitution` | substitutes `substitution_n_residues` interior non-K/R residues, one decoy per unique target | length, cleavage sites | **changes elemental composition** — see invariant 3, and F-036 for a measured consequence. Mass shift measured p10 24 Da, median 55 Da, p90 107 Da, max ~244 Da (F-036 as corrected) — the "~1–50 Da" this table used to claim understated it. CCS features stay usable and it is compatible with `--match-ccs`. |
| `mz_shift` | shifts the query m/z by a random delta in `[delta_min, delta_max]` Da | sequence exactly | in raw-query, snapping is disabled so each decoy sits at its exact shifted m/z on a distinct feature |
| `mz_shuffle` | derangement of the peptide→feature assignment (mass-sorted rotation) | sequence exactly | decoys are **co-located** with targets on identical ion images, so feature-quality features are exactly symmetric. **Do not combine with `--match-ccs`** — it would remove ~all decoys by design. Raw CCS scalars and mobility-gated colocalizations are auto-excluded (`_MZ_SHUFFLE_CCS_LEAK_FEATURES`); only `*_resid` variants are kept. |
| `entrapment` | tryptic peptides from a foreign-organism FASTA (`entrapment_fasta`), isobaric-with-target ones filtered out | — | `protein="ENTRAPMENT_{acc}"` |
| `balanced_shuffle` / `paired_shuffle` | iterative K/R-preserving protein shuffle, keeping only decoys that match a MALDI feature | cleavage sites | achieves ~1:1 T:D on sparse feature lists. **Not compatible with `use_spatial_ranker_features`** (no consistent spatial anchor). |

Every method places decoys in a **separate protein namespace** (`DECOY_…` / `ENTRAPMENT_…`)
so protein-level features are computed within class and a decoy is never pooled with its
source target's protein.

`_SPATIAL_RANKER_OK_DECOYS` (`pipeline.py:53`) = `{entrapment, mz_shift, mz_shuffle,
substitution}`. With any other method `use_spatial_ranker_features` is force-disabled with a
`UserWarning`.

---

## Scoring backends

`--model` accepts **`{lda, qda, svm, gbt, rbf_svm}`** (`cli.py:819`).

| model | estimator | notes |
|---|---|---|
| `lda` | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` | package default; importances are `coef_[0]` |
| `svm` | `sklearn.svm.LinearSVC` | shares `_rescore_linear` with `lda`; adds no dependency |
| `rbf_svm` | `sklearn.svm.SVC(kernel="rbf")` | nonlinear; no `coef_`, so importances are reported as \|structure coefficient\|. `rbf_svm_gamma` accepts `"scale"`/`"auto"` or a float. Training is O(N²). |
| `qda` | `QuadraticDiscriminantAnalysis(reg_param=0.1)` | reuses R1 posteriors for PEP under `--single-round` |
| `gbt` | gradient-boosted trees (`_rescore_gbt`, `pipeline.py:991`) | `gbt_n_estimators`, `gbt_max_depth`, `gbt_learning_rate`; never benchmarked on the three datasets |

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

**Round-1 seed** — `_find_best_feature_labels` (`pipeline.py:456`) sweeps each feature and
both ranking directions, counting targets at q ≤ `train_fdr`. Sub-ULP random noise breaks
ties so row order cannot bias the result. If the best single feature yields fewer than
`min_seed_positives` targets it escalates to pairwise sums/differences on standardised
columns, then to a depth-3 `DecisionTreeClassifier`. Columns in `_BEST_FEAT_SKIP`
(composition and ionisation features) are excluded throughout — invariant 3.

Fallback chain when that yields nothing: `ppm_error_abs < init_ppm_threshold` OR
`n_candidates == 1`; then the top `r1_seed_percentile` of targets by `ppm_error_abs`.

**Post-scoring reweighting** (winners only) is an **additive log-prior**, not multiplicative:

```
reweighted_score = round2_score
                 + lcms_prior_weight   * log(lcms_prior)
                 + spatial_prior_weight * log(spatial_prior)
```

Multiplicative combination would invert the ranking for negative scores.

---

## Feature groups

Defined in `feature_generator.py`; import as `from msi_picasso.feature_generator import ...`.

| constant | line | in the ranker? |
|---|---|---|
| `MALDI_INTRINSIC_FEATURES` | 53 | yes, by default |
| `PROTEIN_LEVEL_FEATURES` | 103 | opt-in `--use-protein-level-feats` |
| `REGION_COLOCALIZATION_FEATURES` | 133 | opt-in `--region-coloc` |
| `WITHIN_REGION_COLOCALIZATION_FEATURES` | 144 | opt-in `--within-region-coloc` (experimental) |
| `LCMS_PRIOR_FEATURES` | 188 | no — applied as an additive log-prior |
| `SPATIAL_RANKER_FEATURES` | 210 | opt-in `--use-spatial-ranker-features` |
| `MOB_QUALITY_FEATURES` | 230 | gated by `_MOB_QUALITY_DEFAULT_DECOYS` (`pipeline.py:62`) |
| `MAIN_FEATURES` | 243 | subset selector |
| `FEATURE_NAN_FILL` | 286 | per-feature NaN sentinels |

`MOB_QUALITY_FEATURES` = `mob_2d_concentration`, `mob_k0_spread`, `mob_mz_spread_ppm`,
`mob_peak_snr` — intrinsic joint (m/z, intensity, 1/K0) peak-quality descriptors, available
only when the dataset has a TIMS dimension.

**Why `LCMS_PRIOR_FEATURES` are excluded from the ranker:** LC-MS/MS ID-derived features
(`lcms_q_value`, `lcms_pep`, `lcms_score`, `n_psms`, `lcms_intensity`) would give confirmed
targets different treatment from decoys, breaking TDC symmetry. They are populated on the
candidates frame by Strategy C but never enter the prior either.

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

| key | package default | baseline configs use |
|---|---|---|
| `model` | `lda` | `rbf_svm` |
| `decoy_method` | `balanced_shuffle` (`rescore()` signature says `shuffle`) | `substitution` |
| `substitution_n_residues` | `1` | `2` |
| `train_fdr` | `0.05` | `0.1` (amyloidosis) / `0.3` (her2, kidney) |
| `init_ppm_threshold` | `2.0` (`rescore()` signature says `5.0`) | `10.0` / `5.0` |
| `min_seed_positives` | `50` | `125` / `20` |
| `matching_ppm` | `20.0` | `0` (exact; see invariant 5) |

### Adding a configurable parameter

1. `package_data/config_default.json` — add the key with its default.
2. `package_data/config_schema.json` — add the type (use `["type", "null"]` to allow a CLI
   `None` passthrough). The schema rejects unknown keys, so this step is mandatory.
3. `cli.py` — add `--param-name` with `default=None`; add the snake_case name to
   `_TOP_LEVEL_ATTRS` (and `_STORE_TRUE_ATTRS` for boolean flags); pass it in the `rescore()`
   call at the bottom of `main()`.
4. `pipeline.py` — add it to the `rescore()` signature with the same default.
5. Add a test in `tests/test_config_parser.py`.

---

## Environment, build, tests

```bash
# interpreter — the bare pyenv python does NOT have the dependencies
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
The suite currently has known failures; see PROGRESS.md §7 before treating a red run as a
regression.

---

## Scripts

In `/home/robbe/MALDI_MSI_score/scripts/`:

| script | purpose |
|---|---|
| `scoreboard.py` | scrape all `results/*/*/run.log` into one comparison table; `--markdown` for a PROGRESS.md row, `--diff A B` for a settings diff between two runs |
| `validate_results.py` | biological validation: marker recovery, GT recovery, LC-MS concordance. Counts on the reported population (peptide-level since F-029) |
| `diagnose_gt.py` | **STALE — does not run on any current result.** LDA-only: reads `17_debug_lda_*` and `lda_score_r*`, which no `rbf_svm` run writes. Its coefficient-attribution analysis is linear-model-specific, so reviving it needs a design decision first, not a rename. |
| `grid_search.py` / `analyze_grid_search.py` | parameter sweep (reuses `raw_query_cache`) and its sensitivity analysis. Its objective counts on the reported population — keep it that way or the sweep optimises something the scoreboard does not show |
| `ablation_svm.py` / `ablation_lda.py` | feature ablation. `ablation_lda.py` still counts at feature level (H-code-1, not fixed — it is LDA-era and unused) |
| `audit_coloc_leak.py` | per-colocalization-column target/decoy AUC and abundance-leak check — run before promoting a coloc feature into the ranker |
| `seed_permutation_test.py` | the F-020 label-permutation test on the seed search, over several independent permutation sets — reads the run's own ranker feature list and reproduces its reported seed |
| `replicate_spread.py` | refits a past run's round 1 under N CV partitions and reports the spread of its ID counts — reads the run's own `.full_config.json`; run this before quoting or comparing any single-fit count |
| `refit_harness.py` | shared helpers for refitting a finished run's scoring step offline with one thing changed: `load`, `fit`, `rollup`, `stats`, `final_report_permutation`. Reproduces a run's reported counts exactly when nothing is varied — if it does not, stop and find out why. Always pass `model_repeats` |
| `compare_backends.py` | which scoring backend, at what cost in IDs (PROGRESS.md F-040). `RUN=E021 python scripts/compare_backends.py` |
| `envelope_qc.py` | isotope-envelope QC |
| `visualize_ms1rescore_features.py` | per-feature, per-candidate target/decoy visualisation |

One-off analyses are parked in `scripts/archive/`.

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
