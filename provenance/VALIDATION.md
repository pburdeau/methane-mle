# Validation on October 2, 2026

## Verified

- Located all 14 figures referenced by the current paper and SI. Exact source
  PDF hashes match the baseline revision experiment directory for 13 assets and
  the very-sparse experiment directory for its comparison figure.
- The curated simulator, estimator, configuration, and model tests match their
  original source files. All 87 collected original files remain unchanged.
- All 17 existing model tests pass with Python 3.14.5, NumPy 2.4.6, pandas 3.0.1,
  Matplotlib 3.10.8, and SciPy 1.17.1.
- Twelve selected simulations from the two 12,500-row datasets reproduce all ten
  checked estimator fields at absolute/relative tolerance `1e-10`, with matching
  missing values. Selected replication IDs are 0, 499, 500, 1581, 12499, and the
  first replication with an undefined MLE in each dataset.
- The new figure runner completes successfully: 13 computational manuscript
  figures are rebuilt and the manual schematic is copied. All required output
  filenames match the current TeX sources.
- Visual comparison against reference PDFs shows matching layout and content.
  At 1200-pixel Poppler rendering, 13 of 14 assets match pixel-for-pixel; the
  threshold figure differs by a negligible rendering amount (mean absolute
  channel difference 0.000153 on a 0-255 scale). This includes the copied schematic.
- Nested-batch baseline variances are 18.3394898259 for MLE and 29.6350391871 for
  POD-weighted, giving 38.1155202457% variance reduction. The sparse scenario gives
  25.3239208076 and 31.9409403611, approximately 20.72% reduction. Exact results are
  in `validation/statistics.json`.

## Items to resolve before publication

1. **Bibliography:** `Reuland2026` is cited twice in `paper/main.tex` and is absent
   from `paper/references.bib`. No uncertain replacement entry was invented.
2. **Baseline p-value:** the paper's introduction reports p=0.40 for the baseline
   MLE bias. The archived baseline's 25 original batches give t=0.318516 and
   p=0.752849. The sparse scenario gives p approximately 0.40, as stated in the SI.
   Both baseline values imply failure to reject zero bias, so the headline
   conclusion is unchanged, but the exact baseline value needs correction or an
   explanation. Manuscript text has been preserved.
3. **Editable diagram:** the methods schematic is included as the supplied PDF.
   Its original editable drawing/source was not found. PDF inclusion and assembly
   are reproducible; authoring that schematic from source is not yet reproducible.
4. **Compilation:** manuscript compilation was attempted through the new runner;
   it stopped with a clear message because `latexmk` is not installed. No compiled
   paper or SI PDF is claimed as verified. The full Overleaf source projects and
   compilation instructions are provided.
5. **Full simulations:** the entire Monte Carlo suite has not been rerun. Existing
   datasets, selected numerical replay, tests, and figure reconstruction were
   checked. Fresh-run commands are supplied for all currently required experiments.
6. **License:** the authors should select a code/data license before public release.
   GitHub hosting and a publication DOI have not been created or assumed.

## Statistical conventions preserved

The original violin routine receives finite-filtered MLE arrays and subsequently
repartitions them for some plotted batch statistics. Its supplied `outer_stats`
still uses the original unfiltered batch boundaries for signed-bias tests. This
behavior is preserved to reproduce the reference figures. The separate
`statistics` command uses original row/batch boundaries for every method, so
minor differences in batch variance and paired-test results are expected.

The figure rounds campaign requirements to the nearest integer (e.g. 455 and
70), while a strict minimum sample requirement uses the ceiling (456 and 71 in
those examples). Both rounded and ceiling values, plus unrounded values, are
recorded in the statistics output. This is a rounding convention, not a missing
input or failed reproduction.

## Evidence files

`validation/` contains the model-test log, selected-replay comparisons, nested
statistics, figure completion record, environment records, and raster comparison
results. Temporary rebuilt PDFs, PNGs, and full logs are in `build/` and are
excluded from the future GitHub repository. Original file checksums are in
`source_manifest.json`; final package checksums are in `package_manifest.json`.
