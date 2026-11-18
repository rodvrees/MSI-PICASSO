# MSI-PICASSO — CLAUDE.md

**Scope:** what the code is and what it must never do. No results: results go to
`/home/robbe/MALDI_MSI_score/PROGRESS.md` (one dated line each) and, for the method's history, `bench/BENCH.md`.

Write for a scientist who knows MS and proteomics but not this project: explain a local term the first time it
appears, short sentences, one claim each, the number and what it means, no figurative language.

## Purpose

`picasso -c config.toml` scores MALDI-MSI MS1 data (a Bruker timsTOF `.d`) against the peptides of an LC-MS/MS PSM
file with a symmetric target-decoy method and reports peptides, and proteins, at controlled FDR. The package does
exactly the method the closed bench adopted (`bench/BENCH.md`, "Bench closed (2026-11-17)",
`bench/specs/selfwit_seeds20.toml`) and nothing else. New scoring ideas belong in a bench, not here.

## Pipeline (`cli.run`, one function per step)

1. **Targets** (`candidates.read_targets`): one row per peptide sequence of the PSM file (psm_utils reader), plus
   each LC-observed hydroxyproline form as `<seq>+h<n>` (n prolines with +15.995). Protein = first accession of the
   first PSM in q-value order.
2. **Decoys** (`candidates.decoys`, `candidates.world`): per target, random 2-residue substitutions (no K/R/C/M),
   keeping the variant closest to the target on sequence-only counts (`d1c_raw`: aromatic, proline, D+E, H; `d1b`:
   aromatic only), then on the mass-defect residual. One decoy set per seed, K = 10 seeds; a HyP target's decoy is its
   base decoy plus the same n O. Decoys within 20 ppm of their parent's mass + isotope / HyP offsets are excluded
   (they would read the parent's own ions). The query m/z of a candidate is its [M+H]+ rounded to 1e-6.
3. **Evidence** (`extract.extract`): one alphatims load; pass 1 sums per frame (= pixel) the events at M-1, M, M+1, M+2
   (+-10 ppm) of every query; pass 2 builds the M m/z profile (1 ppm bins, +-25 ppm) over the query's own top-decile
   pixels. Output: the evidence table Q (one row per query m/z) and the coloc rows (on-tissue M images after optional
   per-pixel TIC normalisation and common-mode removal = minus the mean image over all queries; unit norm).
   Tissue mask: frame TIC >= its `tic_quantile` quantile.
4. **m/z offset** (`score.mz_offset`): label-free global offset for the flank term (mode of the signed centroid over
   the top 1% rows by the other four terms; used only if z > 6 and |o| >= 1 ppm).
   It is reliable only when the top 1% rows are mostly true matches (many IDs). Check `mz_offset` in `run.json`: with
   few IDs or z near the cut, set `mz_offset_ppm = 0` unless there is independent evidence of an offset (her2 is pinned).
5. **Score** (`score.terms`, `score.m1`): five terms, fixed z-sum with signs (+, +, -, -, -):
   coloc (same-protein Pearson r of M images, weighted by sqrt(p90 p90') x the partner's evidence rank, partners
   within 20 ppm dropped, shrunk to the median with weight 4), iso (r of M and M+1 images), kl (envelope vs averagine
   KL), mm1 (r of M-1 and M, hinge 0.5), flank (|top-decile centroid - offset| ppm, hinge 2).
6. **Per-ion competition** (`score.ion_competition`): a row leaves when a higher-scoring target reads its ion
   (0.05 ppm); survivors pay a soft penalty (tau 1) for near-tie targets on their ion.
7. **q-values**: peptide q = monotone (D + 1) / (k T) over each peptide's best row, k = decoy / target peptides
   before the competition. Proteins (`score.protein_table`): picked protein FDR (`q_picked`) and non-competitive `q`.
8. **Output**: `peptides.tsv`, `proteins.tsv`, `rows.parquet` (every candidate with its terms), `evidence.parquet`,
   `run.json` (config, offset, IDs, wall time and peak RSS per step), `figures/` with `figures = true`, and
   `viewer/` for the GUI (`cli.save_viewer`, read-only): `meta.npz` (pixel layout, TIC image, every query's m/z
   profile) and `images.npy` (float16 M-1 / M / M+1 images, each scaled to its max) of the reported set (q <= 10%
   peptides, target peptides of picked q <= 10% proteins, GT, top 50 decoys); `viewer_all_targets` /
   `viewer_all_decoys` add every target / decoy query.

## Critical invariants

1. **Symmetry.** A decoy's evidence comes from the same code as a target's: every term is read at the candidate's own
   query m/z from Q. Only three places read the label: the TDC counts, the per-ion competition (only targets remove a
   row: a decoy plays a false target, which competes only with targets), and the decoy partner rule (a decoy of
   protein X pairs with X's targets except its parent, the context a false target in X would have).
2. **No tissue priors, no intensity strata.** Nothing may use cores, regions or intensity classes; the method must
   run on any MALDI dataset.
3. **Composition stays out of the score.** Decoys change composition, so any sequence/composition feature reads the
   label. The envelope term uses averagine at the query m/z, never the candidate's formula.
4. **Decoys are chosen without MALDI data.** No decoy may be picked, kept or dropped by evidence; the exclusion uses
   masses only.
5. **Bench parity.** At tag `port-parity-exact` the package reproduced the bench exactly on the bench's own evidence
   (`scripts/check_bench_parity.py A`, run from a worktree of that tag). Three later fixes change the candidate set
   on purpose: exact target masses (the bench summed 5-decimal residue masses), decoys of out-of-range parents also
   excluded, a decoy's parent partner excluded by m/z (isobaric siblings). Any other change to the method needs a
   reason in PROGRESS.md; the decoy RNG call sequence and the 1.00728 proton in the decoy floor are kept as the
   bench had them.

## Layout

```
MSI-PICASSO/
├── msi_picasso/
│   ├── candidates.py   targets, HyP forms, decoys, the run's candidate set, the exclusion
│   ├── extract.py      the .d passes (numba kernels), tissue mask, evidence table, coloc rows, pair_r
│   ├── score.py        terms, M1, per-ion competition, q-values, proteins, m/z offset
│   ├── cli.py          config (DEFAULTS), run(), main()
│   ├── figures.py      check figures (ids_vs_fdr, scores, terms, mz, proteins/, candidates/)
│   ├── gui.py          `picasso-gui` (extra `gui`): form = cli.DEFAULTS -> gui_config.toml -> `picasso -c`; live log
│   ├── gui_results.py  the GUI's results viewer (Overview / Proteins / Peptides / Run QC; dash-ag-grid + Plotly)
│   └── tests/test_msi_picasso.py   module self-checks
└── scripts/
    ├── check_bench_parity.py  A: exact port check; B: full run vs bench, per term and per ID
    └── make_bench_ref.py      reference dumps from the bench code (bench interpreter, frozen package)
```

`candidates`, `extract`, `score` and `cli` end with an `assert`-based `__main__` self-check (`cli` with no arguments). Config keys are `cli.DEFAULTS`; unknown keys are an
error. TOML table `[MSI-PICASSO]`.

## Bench and the frozen package

The bench (`/home/robbe/MALDI_MSI_score/bench/`) imports the pre-port package. It is frozen at tag
`bench-freeze-2026-11-17` (commit 572da4c), checked out as the worktree `/home/robbe/MALDI_MSI_score/MSI-PICASSO-bench`;
`bench/msi_picasso` and `scripts/msi_picasso` are symlinks to it, so `python bench/<x>.py` uses the frozen code.
The installed Rust extension `ms1rescore_rs` is only for that frozen code.

## Environment and tests

- Interpreter `/home/robbe/.pyenv/versions/MSIscore/bin/python`; editable install of this folder.
- Run from outside `/home/robbe/MALDI_MSI_score`: its `psm_utils/` checkout shadows the installed psm_utils.
- `pytest` (from `/tmp` or elsewhere): the self-checks. Parity: `check_bench_parity.py` at tag `port-parity-exact`
  (references in `/home/robbe/MALDI_MSI_score/port_ref/`).
- A full run needs about 2 minutes and 30-60 GB RAM per dataset (alphatims holds all events in memory).
