# Methane MLE paper reproduction

Code, synthetic simulation results, and manuscript sources for **Maximum likelihood
estimation improves the precision of methane emission quantification from
multi-tiered monitoring**, by Philippine Burdeau, Evan Sherwin, and Adam Brandt.

This repository studies how to estimate time-averaged methane emissions from
intermittent sources observed by snapshot and continuous monitoring technologies.
The estimator links detections into emission events, combines measurements within
events, and uses inverse-probability weighting to estimate the emission-size
distribution. It then estimates ON/OFF transition probabilities from detections
and non-detections, conditional on that distribution. Emission size persists
within an event, and the event-size and transition estimates are updated
iteratively.

The simulation experiments compare this estimator with the naive empirical mean,
probability-of-detection weighting, and an ungrouped estimator. They examine
baseline and sparse monitoring, sampling cadence, campaign length, emission
duration, plume-linking thresholds, numerical accuracy, and sensor-calibration
uncertainty.

## Repository contents

- `src/`: emission simulator, observation model, estimators, and comparators.
- `config.py`: simulation parameters and numerical settings.
- `experiments/`: simulation experiments and figure-generation scripts.
- `data/baseline/` and `data/sparse/`: saved simulation results, including
  12,500 campaigns per scenario and the sensitivity experiments.
- `paper/` and `si/`: main-paper and supplementary LaTeX sources,
  bibliographies, and the 14 referenced external figure PDFs.
- `tests/`: 23 checks of the simulation model and estimation calculations.
- `reproduce.py`: commands for checking the package, running experiments,
  calculating statistics, generating figures, and compiling manuscripts.
- `provenance/`: file checksums, validation records, and figure-source information.

All study data are synthetic; no external datasets or credentials are required.
Generated outputs are written to `build/`, separately from the supplied data.

## Installation

The package has been verified with Python 3.14.5 and the dependency versions in
`requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The estimation code can use optional C acceleration, built with a system C
compiler. If a compiler is unavailable, the same likelihood recursion runs in
NumPy.

## Check the package and reproduce the statistics

```sh
python reproduce.py check --strict
python reproduce.py tests
python reproduce.py replay
python reproduce.py statistics
```

The package check verifies file hashes, manuscript figure references,
bibliography keys, and simulation records. The tests check the likelihood against
independently constructed transition matrices, irregular observation gaps,
measurement combination, and event-detection weighting. Replay checks selected
campaigns against their saved estimator outputs within documented numerical
tolerances.

The statistics command reports bias, variance, Monte Carlo standard errors,
paired estimator comparisons, parameter estimates, numerical diagnostics, and
campaign counts for precision targets. Baseline and sparse results use 25 batches
of 500 campaigns. Confidence bands describe simulation uncertainty; campaign
counts assume independent campaigns and negligible bias at the target precision.

## Reproduce the figures

```sh
python reproduce.py figures
```

Figures are generated from the supplied simulation results and saved in
`build/figures/`. The method schematic is supplied as a vector PDF, and the
main-paper graphical model is drawn directly in LaTeX. Shared plotting colours
are defined in `experiments/plot_style.py`: grey for Naive, orange for
POD weighting, blue for MLE-ungrouped, and teal for MLE.

## Run the simulations

```sh
METHANE_WORKERS=6 python reproduce.py simulations
python reproduce.py figures --data-root build/simulations
python reproduce.py statistics --data-root build/simulations
```

Worker count defaults to one. Campaign seeds and ordered results make simulation
draws independent of worker count. A full run can take several hours. To run one
experiment:

```sh
python reproduce.py simulations --only baseline
```

Available experiments are `baseline`, `sparse`, `sweep_T`, `sweep_theta_snap`,
`sweep_tau_emit`, `sweep_p_cont`, `sweep_T_cont`, `sweep_psnap`, `grid_robustness`,
`threshold_sweep`, and `misspec`.

The baseline seed is 2602. Baseline, sparse, and most parameter sweeps use
25 batches of 500 campaigns per setting; the linking-threshold sweep uses five
batches of 500. The numerical-accuracy experiment compares four optimizer
tolerances on the same 500 campaigns. Sensor-calibration experiments perturb
six sensor parameters separately.

The sparse scenario uses snapshot probability 0.0015, continuous-window trigger
probability 0.015, and a continuous-sensor detection threshold of 15 kg/h.
Numerical flags are retained in the saved results. Campaigns with observations
but no detections contribute a zero mean estimate and missing component estimates.

## Compile the main paper and supplementary information

Compilation requires latexmk with a TeX distribution, or Tectonic.

```sh
python reproduce.py manuscripts
python reproduce.py manuscripts --regenerated
```

The first command uses the supplied figure PDFs; the second uses the regenerated
figures in `build/figures/`. The runner compiles the SI first to resolve
cross-references and produces `build/paper.pdf` and `build/si.pdf`.

For Overleaf, upload `paper/` and `si/` into one project, compile `si/main.tex`,
then compile `paper/main.tex`.

See `provenance/VALIDATION.md` for simulation results and verification details.
