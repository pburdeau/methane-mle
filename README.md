# Methane MLE paper reproduction

Code, synthetic simulation results, and manuscript sources for **Maximum likelihood
estimation improves the precision of methane emission quantification from
multi-tiered monitoring**, by Philippine Burdeau, Evan Sherwin, and Adam Brandt.

This revision removes the empirical likelihood nudge while retaining plume linking,
measurement combination, event-level inverse-probability weighting (IPW), and the
outer iteration. Step 3 uses OFF plus persistent ON-size states, conditional on the
weighted empirical event-size distribution. A size persists until its event ends;
a new event draws a new size. The forward recursion is exact conditional on this
finite distribution. Estimating it from linked noisy measurements is a plug-in
approximation: the procedure is neither joint maximum likelihood nor exact EM,
and standard MLE theory does not establish its unbiasedness or efficiency.

## Contents

- `src/`: simulator, observation model, persistent-size estimator and comparators.
- `config.py`: scenario parameters and fixed numerical settings.
- `experiments/`: manuscript experiments and plotting code.
- `tests/`: 17 original model checks and six persistent-likelihood checks.
- `data/baseline/` and `data/sparse/`: revised saved results, including 12,500
  campaigns per scenario and the manuscript sensitivity experiments.
- `paper/` and `si/`: LaTeX sources, bibliographies and the 14 referenced figure PDFs.
  The main paper's graphical model is drawn within its LaTeX source.
- `reproduce.py`: reproduction entry point.
- `provenance/`: original source records, current checksums and validation notes.

All study data are synthetic; no external data or credentials are needed.
The previous implementation and results remain recoverable in Git history.
Optional legacy estimator variants in `src/estimate_variants.py` are historical;
they are not used by the revised manuscript experiments.

## Setup and checks

Verified with Python 3.14.5 and the pinned versions in `requirements.txt`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py check --strict
python reproduce.py tests
python reproduce.py replay
python reproduce.py statistics
```

The runner attempts to build the optional C acceleration with the system C
compiler. If no compiler is available, the same recursion runs in NumPy, more
slowly. No platform-specific compiled library is committed. Tests compare the
forward recursion against independently constructed dense transition matrices,
including irregular gaps, and check the original IPW recursion and no-nudge rule.
Replay checks selected campaigns across both scenarios, including zero detections.
Rates allow 0.005 kg/h absolute tolerance and transitions 0.00001, with relative
tolerance 0.00001, to accommodate numerical optimizer and backend differences.

`check` validates current file hashes, figure references, bibliography keys, and
both replication counts and method-version records. `statistics` reports the
original 25-by-500 batch design, Monte Carlo standard errors, paired comparisons,
parameter summaries, numerical flags and campaign planning calculations.
Outputs go under `build/` and do not overwrite the bundled reference data.

## Figures and simulations

```sh
python reproduce.py figures
METHANE_WORKERS=6 python reproduce.py simulations
python reproduce.py figures --data-root build/simulations
python reproduce.py statistics --data-root build/simulations
```

Worker count defaults to one; ordered tasks and per-campaign seeds make the
scientific draws independent of worker count. A full rerun can take several hours.
Each requested experiment starts again; the runner does not resume checkpoints.
For one experiment, use `python reproduce.py simulations --only baseline`.
Other names: `sparse`, `sweep_T`, `sweep_theta_snap`, `sweep_tau_emit`, `sweep_p_cont`,
`sweep_T_cont`, `sweep_psnap`, `grid_robustness`, `threshold_sweep`, `misspec`.

The baseline seed is 2602. All scenarios use a unit nudge argument; a nonunit
argument raises an error. Both transition probabilities are optimized over the
fixed domain [0.00001, 0.999999], independent of generating truth. The first
iteration uses fixed starting values and a fixed coarse search; later iterations
start at the preceding estimate. The outer iteration stops at transition changes
below 0.00001 and mean changes below 0.0001 kg/h, or after 30 iterations. Numerical
flags are retained; finite estimates are not removed because of those flags.
Campaigns with observations but zero detections contribute a zero mean estimate,
with missing component estimates; entirely unobserved campaigns remain missing.
The `has_detections` flag records availability of detected data; it does not
assert formal statistical identifiability.

Baseline, sparse and most sweeps use 25 batches of 500 campaigns per setting.
Linking-threshold sensitivity uses five batches of 500. Numerical accuracy uses
four optimizer tolerances on the same 500 datasets, retaining all empirical size
support points; its historical filename is `grid_robustness.csv`. Misspecification
uses the original eta grid and separately seeded 0.6, 0.7 and 0.8 extension for
all six sensor parameters. The snapshot sweep uses campaign length 1000.

The sparse scenario uses p_snap=0.0015, p_cont=0.015 and continuous POD threshold
15 kg/h. Figure reconstruction regenerates 13 computational assets and copies
the externally authored schematic. PDF bytes and rendering can vary by platform.
Confidence bands describe Monte Carlo performance, not uncertainty for an
individual monitoring record. Campaign counts are normal-approximation planning
values rounded upward and assume that bias is negligible at the target precision.

## Compile the paper and SI

Use Overleaf, latexmk with a TeX distribution, or Tectonic:

```sh
python reproduce.py manuscripts
python reproduce.py manuscripts --regenerated
```

The runner stages both projects under `build/manuscripts`, compiles the SI first
for external references, and writes `build/paper.pdf` and `build/si.pdf`.
For Overleaf upload both `paper/` and `si/` in one project and compile `si/main.tex`
then `paper/main.tex`. A separate combined upload ZIP has `main.tex` and `si.tex`
at the root, shared figures, and cache-safe compilation settings.

See `provenance/VALIDATION.md` for measured results and verification details.
The source manifest preserves the original assembly record; the package manifest
checks the current revision. The GitHub repository is public at
https://github.com/pburdeau/methane-mle. The authors still need to choose a license.
