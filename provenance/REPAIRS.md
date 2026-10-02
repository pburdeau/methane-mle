# Changes made in the curated package

The originals under `mle_methane_clean_simultaneous` and the top-level Overleaf
folders were left unchanged. `source_manifest.json` records the source path,
SHA-256 checksum, and size of each collected file. `package_manifest.json` records
the final curated files after the changes below.

1. The paper's external SI reference changed from `\externaldocument{SI}` to
   `\externaldocument{../si/main}` to match the packaged projects.
2. Experiment plotting cache paths use a writable temporary directory and honor
   `MPLCONFIGDIR` instead of unconditionally using `/tmp/mpl_cache`.
3. `run_grid_robustness.py` accepts `--plot-only` using its existing figure routine.
4. `run_threshold_sweep.py` separates its existing plot into `make_figure` and
   accepts `--plot-only`. The simulation and plot equations are unchanged.
5. `reproduce.py` replaces the legacy master runner for this package. It uses the
   filenames actually referenced in the current manuscripts, the extended sweep
   composites, the correct precision-figure name, and the SI sparse comparison.
   Sparse plots use a separate subdirectory so they cannot replace baseline plots.
6. The new runner stops on subprocess failures, preserves archived inputs, records
   the environment, and explicitly distinguishes plotting from fresh simulations.

The original `run_all_figures.py` is retained as
`original_run_all_figures.py` for provenance. It is not a supported entry point:
its baseline rename expects a file the current plotter does not produce, several
assembly outputs do not match TeX references, caching/force behavior is
inconsistent, and it continues after subprocess failures.

The estimator and simulator in `src/`, original configuration, and model tests
are copied without edits. Diagnostic scripts, exploratory run directories, old
figures, system metadata, and unrelated literature PDFs were excluded.
