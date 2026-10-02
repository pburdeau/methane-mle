# Methane MLE paper reproduction

Code, saved simulation results, and Overleaf sources for **Maximum likelihood
estimation improves the precision of methane emission quantification from
multi-tiered monitoring**, by Philippine Burdeau, Evan Sherwin, and Adam Brandt.

This package uses the October 2026 manuscript copies inside
`mle_methane_clean_simultaneous`. The older top-level Overleaf exports are
preserved under `provenance/top_level_overleaf` for comparison. The original
workspace has not been changed.

## Contents

- `src/`: simulator, observation model, MLE, and comparison estimators.
- `config.py`: baseline parameters and the transition-probability grid.
- `experiments/`: the experiments and plotting code needed by the manuscripts.
- `tests/`: the original 17 model tests; no pytest dependency is required.
- `data/baseline/`: 12,500 baseline replications, example realizations, and saved
  parameter sweeps, grid robustness, threshold sensitivity, and misspecification.
- `data/sparse/`: 12,500 very-sparse-cadence replications and their parameters.
- `paper/` and `si/`: manuscript sources, bibliographies, required template files,
  and only the figures actually referenced by the current manuscripts.
- `reproduce.py`: supported reproduction entry point.
- `provenance/`: source checksums, figure provenance, validation, and repairs.

All scientific data used here are synthetic. No external observational dataset,
spreadsheet, notebook, API credential, or download is needed by the selected code.

## Setup

The verification environment was Python 3.14.5 with the versions pinned in
`requirements.txt`. A TeX installation is only needed to compile the manuscripts.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all commands below from this directory. Outputs go under `build/`, which is
excluded from version control. The supplied data and reference figures are kept
unchanged.

## Check the package and estimator

```sh
python reproduce.py check
python reproduce.py tests
python reproduce.py replay
python reproduce.py statistics
```

`check` validates package hashes, every manuscript figure reference, bibliography
keys, and both 12,500-row replication files. It reports unresolved publication
items as warnings; use `--strict` to treat those as failures.

`tests` runs the original 17 model tests. `replay` reruns 12 selected simulations
across the two scenarios, including examples with undefined MLEs, and compares ten
estimator outputs with the archived values at absolute and relative tolerance
`1e-10`. This is a numerical consistency check, not a rerun of the entire study.

`statistics` computes nested-batch bias, variance, paired comparisons, and sample
size requirements from both archived datasets, saving `build/statistics.json`.

## Rebuild figures from the saved results

```sh
python reproduce.py figures
```

This reconstructs the 13 computational figures used in the paper and SI, and
copies the externally authored method schematic. Example realization plots are
regenerated from their recorded seeds and parameters. Final manuscript assets
are in `build/figures/`; auxiliary plots may also be produced.

The publication reference PDFs remain in `paper/figures` and `si/figures`.
Rebuilt PDFs can differ in metadata, fonts, and rendering across machines; do not
expect byte-identical PDFs. The original scientific plotting conventions are
preserved, including finite-MLE filtering in the baseline violin routine.

## Rerun the simulations

```sh
python reproduce.py simulations
```

This runs every experiment required by the current manuscript figures, using
fresh output directories under `build/simulations/`. It can take many hours.
It does not reuse the bundled CSVs or overwrite them. Each selected experiment
starts again when invoked; the runner is not a checkpoint/resume system.

To rerun one experiment:

```sh
python reproduce.py simulations --only baseline
python reproduce.py simulations --only sparse
python reproduce.py simulations --only sweep_T
```

Other experiment names are `sweep_theta_snap`, `sweep_tau_emit`, `sweep_p_cont`,
`sweep_T_cont`, `sweep_psnap`, `grid_robustness`, `threshold_sweep`, and `misspec`.
The last experiment includes the original eta grid and its 0.6, 0.7, 0.8 extension
with the original separate seed offset. The snapshot sweep uses T=1000, the value
in the archived manuscript dataset. Grid robustness uses 500 replications, and
linking-threshold sensitivity uses 5 x 500, as defined in the original scripts;
baseline, sparse, other sweeps, and misspecification use 25 x 500.

After all simulations finish, rebuild figures from the fresh data:

```sh
python reproduce.py figures --data-root build/simulations
```

The baseline seed is 2602. Baseline likelihood nudge is 1.015; the sparse scenario
uses 1.00, p_snap=0.0015, p_cont=0.015, and continuous POD threshold 15 kg/h.
The archived `data_parameters.csv` files omit nudge; the sparse value is confirmed
by its original run log and SI text. The baseline value is supported by the
run-directory name and current configuration, and was verified by numerical
replay. Nudge is explicitly set by the reproduction runner.

The transition grids are centered on the generating transition probabilities,
as in the original experiments. This reproduces the study's simulation setup;
application to real monitoring data would need a separately chosen search domain.
MLE outputs with insufficient detections remain missing, matching the archived
code; the baseline has 12,490 valid MLEs and the sparse scenario 12,471.

## Compile the paper and SI

A complete TeX distribution with `latexmk`, `pdflatex`, and BibTeX is needed:

```sh
python reproduce.py manuscripts
```

The runner stages both projects under `build/manuscripts`, compiles the SI first
to resolve external references, and writes `build/paper.pdf` and `build/si.pdf`.
Use `--regenerated` to compile with the rebuilt figures instead of reference PDFs.

For Overleaf, use a single project containing both the `paper/` and `si/` folders.
Compile `si/main.tex` first and then `paper/main.tex`. Alternatively, upload each
folder separately and copy the SI's compiled `main.aux` into the paper project as
`SI.aux`, replacing `\externaldocument{../si/main}` with `\externaldocument{SI}`.

## Verification and remaining items

See `provenance/VALIDATION.md` for observed results and limitations. Figure
reconstruction, model tests, and selected numerical replay have been verified.
The full simulation suite was not rerun, and manuscript compilation is unverified
because this machine does not have a TeX compiler.

The current paper cites `Reuland2026` twice, but its bibliography has no such
entry. The intended bibliographic record must be supplied before publication.
The baseline unbiasedness p-value in the paper (0.40) differs from the archived
nested-batch calculation (0.752849); the manuscript text is preserved for review.
The method schematic is available as its original PDF; no editable source was
found in this workspace.

The GitHub repository has not been created. A license should be selected by the
authors before public release. No license or publication identifier has been
invented for this package.
