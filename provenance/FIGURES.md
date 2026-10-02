# Current manuscript figure provenance

SHA-256 comparisons identified exact matches between every referenced manuscript
PDF and the named original experiment directory. All paths below are relative to
the original `mle_methane_clean_simultaneous` folder.

## Paper

- `figure0.pdf`: manual method schematic; exact match in
  `revision-psnap0125-nudge1015/figures/`. Editable source unavailable.
- `figure_baseline_violin_v2_wide.pdf`: `experiments/run_baseline.py`;
  `revision-psnap0125-nudge1015/results/data_replications.csv`.
- `figure1_histograms.pdf`: `experiments/make_figure1.py`; baseline replications.
- `figure4_composite.pdf`: `experiments/make_figure_sweeps_extended.py`, mode
  `main3`; snapshot-frequency, snapshot-threshold, and duration sweeps.
- `figure5_precision.pdf`: `experiments/make_figure_precision.py`; baseline
  replications, using 25 batches of 500 to calculate variance.

## Supplementary information

- `figure_pod_curves.pdf`: `experiments/run_baseline.py`; baseline sensor settings.
- `figure_size_distribution.pdf`: `experiments/run_baseline.py`; baseline
  lognormal distribution parameters.
- `figure_realizations.pdf`: `experiments/run_baseline.py`; selected replications
  1581, 3973, 4215, seed 2602, with emission RNG offset 100000.
- `figure_baseline_violin_v2_wide_sparse.pdf`: `experiments/run_baseline.py`;
  exact match in `revision-very-sparse-pcont0015/figures/`. Uses that directory's
  replication data and documented sparse parameters, with nudge=1.00.
- `figure1_composite.pdf`: `experiments/make_figure1.py`; baseline replications
  and `results_linear/sweep_T_summary.csv` from the baseline revision directory.
- `figure_sweeps_cont.pdf`: `experiments/make_figure_sweeps_extended.py`, mode
  `cont2`; continuous-frequency and window-length sweep summaries.
- `figure_grid_robustness.pdf`: `experiments/run_grid_robustness.py`;
  `revision-psnap0125-nudge1015/results/grid_robustness.csv`.
- `figure_threshold_sweep.pdf`: `experiments/run_threshold_sweep.py`;
  `revision-psnap0125-nudge1015/results/threshold_sweep.csv`.
- `figure_misspec_thetas_mse.pdf`: `experiments/run_misspec.py`,
  `figure_misspec_single_param_mse`; baseline revision's `misspec_onebyone.csv`.

Except for the sparse comparison, all computational reference PDFs exactly match
the baseline revision's `figures/` folder. The corresponding CSVs have been copied
to `data/baseline/results/` and `data/baseline/results_linear/`. Sparse CSVs are in
`data/sparse/results/`. All original required PDFs are kept in `paper/figures/`
or `si/figures/`.

The joint-misspecification CSV is retained as supporting data but is not required
by a current TeX figure reference. The supported full simulation runner therefore
recomputes the one-by-one misspecification analysis used in the current SI.
