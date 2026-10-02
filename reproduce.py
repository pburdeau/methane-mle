#!/usr/bin/env python3
"""Reproduce the supplied paper without overwriting its reference assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def run(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)],
                   cwd=ROOT, check=True)


def write_report(name, report):
    BUILD.mkdir(exist_ok=True)
    (BUILD / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def check(strict=False):
    """Audit all packaged inputs, references, and original source hashes."""
    manifest = json.loads((ROOT / "provenance/package_manifest.json").read_text())
    failures = []
    for record in manifest["files"]:
        p = ROOT / record["path"]
        if not p.is_file():
            failures.append(f"Missing: {record['path']}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(f"Changed since assembly: {record['path']}")
    warnings = []
    for doc in ("paper", "si"):
        p = ROOT / doc / "main.tex"
        source = re.sub(r"(?m)(?<!\\)%.*", "", p.read_text())
        for asset in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
            if not (p.parent / asset).is_file():
                failures.append(f"Missing figure: {doc}/{asset}")
        keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",
                              (p.parent / "references.bib").read_text()))
        cited = {key.strip() for group in re.findall(
            r"\\cite\w*\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", source)
                 for key in group.split(",")}
        for key in sorted(cited - keys):
            warnings.append(f"Missing bibliography entry in {doc}: {key}")
    for tag in ("baseline", "sparse"):
        with (ROOT / "data" / tag / "results/data_replications.csv").open() as f:
            count = sum(1 for _ in csv.DictReader(f))
        if count != 12500:
            failures.append(f"{tag}: expected 12500 replications, found {count}")
    warnings.append("The method schematic is supplied as a PDF; its editable source was not found.")
    warnings.append("Baseline unbiasedness p-value in paper is 0.40; archived nested batches give 0.752849. See provenance/VALIDATION.md.")
    warnings.append("Full simulation suite and LaTeX compilation need separate verification.")
    report = {"files_checked": len(manifest["files"]),
              "failures": failures, "warnings": warnings}
    write_report("input_audit.json", report)
    print(json.dumps(report, indent=2))
    if failures or (strict and warnings):
        raise SystemExit(1)


def baseline_parameters(base, sparse=False):
    """Use the archived experiment's parameters, including its explicit nudge."""
    from experiments.run_baseline import _build_parser, _derived_params
    row = next(csv.DictReader((base / "results/data_parameters.csv").open()))
    p = _build_parser().parse_args([])
    names = {"seed": "seed", "T": "T", "pi_on": "pi_on", "tau_emit": "tau_emit",
             "mu_emit": "mu_emit", "size_mu": "size_mu", "size_sigma": "size_sigma",
             "p_snap": "p_snap", "p_cont": "p_cont", "T_cont": "T_cont",
             "snap_threshold": "snap_threshold", "snap_slope": "snap_slope",
             "snap_sigma": "snap_sigma", "cont_threshold": "cont_threshold",
             "cont_slope": "cont_slope", "cont_sigma": "cont_sigma",
             "cont_fp_rate": "cont_fp_rate", "cont_fp_scale": "cont_fp_scale",
             "max_gap": "max_gap", "decision_threshold": "decision_threshold",
             "p_grid_res": "p_grid_res", "p_geom_factor": "p_geom_factor",
             "n_reps": "n_reps", "n_outer": "n_outer"}
    for attr, col in names.items():
        old = getattr(p, attr)
        setattr(p, attr, int(row[col]) if isinstance(old, int) else float(row[col]))
    p.nudge = 1.0 if sparse else 1.015
    p.p_off_nudge = p.nudge
    return _derived_params(p)


def specs_for(p):
    from src import build_baseline_specs
    return build_baseline_specs(
        aerial_threshold=p.snap_threshold, aerial_slope=p.snap_slope,
        aerial_sensor_sigma=p.snap_sigma, cont_threshold=p.cont_threshold,
        cont_slope=p.cont_slope, cont_sensor_sigma=p.cont_sigma,
        cont_fp_rate=p.cont_fp_rate, cont_fp_scale=p.cont_fp_scale)


def replay():
    """Replay representative archived simulations with the supplied estimator."""
    import numpy as np
    import pandas as pd
    from experiments.run_baseline import run_one
    mapping = {"naive": "naive", "pod": "pod_weighted", "mle_simple": "mle_simple",
               "mle": "mle", "p_on": "mle_p_on", "p_off": "mle_p_off",
               "mu_emit": "mle_mu_emit", "ms_p_on": "ms_p_on",
               "ms_p_off": "ms_p_off", "ms_mu_emit": "ms_mu_emit"}
    records = []
    for tag in ("baseline", "sparse"):
        base = ROOT / "data" / tag
        p = baseline_parameters(base, tag == "sparse")
        specs = specs_for(p)
        saved = pd.read_csv(base / "results/data_replications.csv")
        # Across batches, plus a replication where the MLE is undefined.
        missing = saved.index[saved.mle.isna()].tolist()
        reps = sorted(set([0, 499, 500, 1581, 12499] + missing[:1]))
        for rep in reps:
            fresh = run_one(specs, rep, p)
            for actual, archived in mapping.items():
                np.testing.assert_allclose(fresh[actual], saved.iloc[rep][archived],
                                           rtol=1e-10, atol=1e-10, equal_nan=True,
                                           err_msg=f"{tag} replication {rep}, {actual}")
            records.append({"experiment": tag, "replication": rep, "matched": True})
    write_report("replay_verification.json", {"tolerance": "rtol=atol=1e-10",
                                              "replications": records})
    print(f"Replayed {len(records)} archived simulations; all 10 estimator fields matched.")


def statistics():
    """Recompute reported quantities using complete archived batches."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    report = {}
    for tag in ("baseline", "sparse"):
        base = ROOT / "data" / tag
        p = baseline_parameters(base, tag == "sparse")
        df = pd.read_csv(base / "results/data_replications.csv")
        methods = {}
        batch_vars = {}
        for name, col in [("Naive", "naive"), ("POD-weighted", "pod_weighted"),
                          ("MLE-ungrouped", "mle_simple"), ("MLE", "mle")]:
            means, variances = [], []
            for o in range(p.n_outer):
                a = df[col].iloc[o*p.n_reps:(o+1)*p.n_reps].dropna().to_numpy()
                means.append(float(np.mean(a)))
                variances.append(float(np.var(a, ddof=0)))
            bias = np.array(means) - p.mu_true
            test = stats.ttest_1samp(bias, 0.0)
            batch_vars[name] = np.array(variances)
            methods[name] = {"valid_replications": int(df[col].notna().sum()),
                             "nested_mean": float(np.mean(means)),
                             "nested_bias": float(np.mean(bias)),
                             "nested_variance": float(np.mean(variances)),
                             "bias_t": float(test.statistic), "bias_p": float(test.pvalue)}
        pair = stats.ttest_rel(batch_vars["POD-weighted"], batch_vars["MLE"])
        vm, vp = methods["MLE"]["nested_variance"], methods["POD-weighted"]["nested_variance"]
        precision = {str(eps): {name: {
            "exact": 1.96**2 * methods[name]["nested_variance"] / (eps*p.mu_true)**2,
            "figure_rounded": int(round(1.96**2 * methods[name]["nested_variance"] / (eps*p.mu_true)**2)),
            "minimum_integer": int(np.ceil(1.96**2 * methods[name]["nested_variance"] / (eps*p.mu_true)**2))}
                               for name in ("MLE", "POD-weighted")}
                     for eps in (0.03, 0.05, 0.10)}
        report[tag] = {"n_outer": p.n_outer, "n_inner": p.n_reps,
                       "methods": methods, "variance_reduction_percent": 100*(1-vm/vp),
                       "paired_variance_t": float(pair.statistic),
                       "paired_variance_p": float(pair.pvalue),
                       "campaigns_for_relative_precision_at_95_percent": precision}
        print(f"{tag}: MLE variance {vm:.4f}; POD variance {vp:.4f}; "
              f"reduction {100*(1-vm/vp):.2f}%")
    write_report("statistics.json", report)


def plot_baseline(base, target, sparse=False):
    import numpy as np
    import pandas as pd
    from experiments import run_baseline as rb
    final_target = target
    if sparse:
        target = target / "sparse"
        target.mkdir(parents=True, exist_ok=True)
    p = baseline_parameters(base, sparse)
    specs = specs_for(p)
    if not sparse:
        # Match the original runner's plotting order and initial font defaults.
        rb.figure_pod_curves(specs, target / "figure_pod_curves.pdf",
                            snap_threshold=p.snap_threshold, snap_slope=p.snap_slope,
                            cont_threshold=p.cont_threshold, cont_slope=p.cont_slope)
    df = pd.read_csv(base / "results/data_replications.csv")
    outer_stats = {}
    columns = {"Naive": "naive", "POD-weighted": "pod_weighted",
               "MLE-ungrouped": "mle_simple", "MLE": "mle"}
    for name, col in columns.items():
        biases, variances = [], []
        for o in range(p.n_outer):
            a = df[col].iloc[o*p.n_reps:(o+1)*p.n_reps].to_numpy()
            a = a[np.isfinite(a)]
            biases.append(float(np.mean(a) - p.mu_true))
            variances.append(float(np.var(a, ddof=0)))
        outer_stats[name] = {"bias_arr": np.array(biases), "var_arr": np.array(variances)}
    # Preserve the original runner's arguments, including its finite-MLE filtering.
    rb.figure_violin_v2(df.naive.to_numpy(), df.pod_weighted.to_numpy(),
                       df.mle_simple.dropna().to_numpy(), df.mle.dropna().to_numpy(),
                       p.mu_true, target / "figure_baseline_violin_v2.pdf", outer_stats)
    if sparse:
        shutil.copy2(target / "figure_baseline_violin_v2_wide.pdf",
                     final_target / "figure_baseline_violin_v2_wide_sparse.pdf")
        return
    rb.figure_size_distribution(p, target / "figure_size_distribution.pdf")
    reps = sorted(int(f.stem.split("rep")[-1]) for f in
                  (base / "results").glob("data_realization_rep*.csv"))
    rb.figure_realizations(specs, p, target / "figure_realizations.pdf", reps=reps)


def figures(data_root):
    target = BUILD / "figures"
    target.mkdir(parents=True, exist_ok=True)
    base = data_root / "baseline"
    plot_baseline(base, target)
    plot_baseline(data_root / "sparse", target, sparse=True)
    run("experiments.make_figure1", "--res-dir", base, "--out", target / "figure1_composite.pdf")
    run("experiments.make_figure_precision", "--res-dir", base,
        "--out", target / "figure5_precision.pdf")
    for mode, filename in [("main3", "figure4_composite.pdf"), ("cont2", "figure_sweeps_cont.pdf")]:
        run("experiments.make_figure_sweeps_extended", "--res-dir", base,
            "--mode", mode, "--out", target / filename)
    for module in ("run_grid_robustness", "run_threshold_sweep"):
        run(f"experiments.{module}", "--plot-only", "--res-dir", base / "results",
            "--fig-dir", target)
    run("experiments.run_misspec", "--mode", "onebyone", "--plot-only",
        "--res-dir", base / "results", "--fig-dir", target)
    # The graphical methods diagram was authored outside the simulation code.
    shutil.copy2(ROOT / "paper/figures/figure0.pdf", target / "figure0.pdf")
    required = json.loads((ROOT / "provenance/source_manifest.json").read_text())["figures"]
    missing = [item["asset"] for item in required if not (target / Path(item["asset"]).name).exists()]
    if missing:
        raise RuntimeError(f"Required figures were not produced: {missing}")
    write_report("figure_verification.json", {
        "data_root": str(data_root), "reference_figures_produced": len(required),
        "simulation_figures_rebuilt": 13, "manual_schematic_copied": 1,
        "note": "Scientific inputs are unchanged; PDF metadata and fonts may differ."})
    print(f"All 14 manuscript figure assets are available in {target}")


def simulations(only):
    """Fresh simulations use a separate output tree and stop on any failure."""
    base = BUILD / "simulations/baseline"
    sparse = BUILD / "simulations/sparse"
    jobs = {
        "baseline": lambda: run("experiments.run_baseline", "--n-reps", 500,
            "--n-outer", 25, "--nudge", 1.015, "--output-dir", base),
        "sparse": lambda: run("experiments.run_baseline", "--n-reps", 500,
            "--n-outer", 25, "--p-snap", 0.0015, "--p-cont", 0.015,
            "--cont-threshold", 15, "--nudge", 1.0, "--output-dir", sparse),
    }
    for sweep in ("T", "theta_snap", "tau_emit", "p_cont", "T_cont"):
        jobs[f"sweep_{sweep}"] = lambda s=sweep: run("experiments.run_sweeps",
            "--sweep", s, "--n-inner", 500, "--n-outer", 25,
            "--res-dir", base / "results_linear", "--fig-dir", base / "figures")
    jobs["sweep_psnap"] = lambda: run("experiments.run_sweep_psnap", "--T-only", 1000,
        "--n-inner", 500, "--n-outer", 25,
        "--res-dir", base / "results", "--fig-dir", base / "figures")
    for name in ("grid_robustness", "threshold_sweep"):
        jobs[name] = lambda n=name: run(f"experiments.run_{n}",
            "--res-dir", base / "results", "--fig-dir", base / "figures")
    def misspec():
        run("experiments.run_misspec", "--mode", "onebyone", "--n-inner", 500,
            "--n-outer", 25, "--res-dir", base / "results", "--fig-dir", base / "figures")
        run("experiments.run_misspec", "--mode", "onebyone", "--n-inner", 500,
            "--n-outer", 25, "--extra-eta", 0.6, 0.7, 0.8,
            "--res-dir", base / "results", "--fig-dir", base / "figures")
    jobs["misspec"] = misspec
    if only and only not in jobs:
        raise SystemExit(f"Unknown experiment {only}; choose from {', '.join(jobs)}")
    for name, job in jobs.items():
        if only and name != only:
            continue
        print(f"\nRunning {name}", flush=True)
        job()


def manuscripts(regenerated=False):
    """Compile separate multi-file Overleaf projects with a local TeX install."""
    if not shutil.which("latexmk"):
        raise SystemExit("LaTeX compilation is unverified: latexmk is not installed. "
                         "Use Overleaf or install a TeX distribution with latexmk.")
    dest = BUILD / "manuscripts"
    for doc in ("si", "paper"):
        target = dest / doc
        shutil.copytree(ROOT / doc, target, dirs_exist_ok=True)
        if regenerated:
            for p in (target / "figures").glob("*.pdf"):
                shutil.copy2(BUILD / "figures" / p.name, p)
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "main.tex"], cwd=target, check=True)
    for doc in ("si", "paper"):
        shutil.copy2(dest / doc / "main.pdf", BUILD / f"{doc}.pdf")
    print(f"Compiled manuscripts: {BUILD / 'paper.pdf'} and {BUILD / 'si.pdf'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "tests", "replay", "statistics", "figures", "simulations", "manuscripts"])
    parser.add_argument("--strict", action="store_true", help="Fail input audit on unresolved limitations")
    parser.add_argument("--only", help="Run one simulation experiment")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--regenerated", action="store_true", help="Compile with build/figures")
    args = parser.parse_args()
    start = time.monotonic()
    os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "matplotlib"))
    BUILD.mkdir(exist_ok=True)
    dispatch = {"check": lambda: check(args.strict), "tests": lambda: run("tests.test_model"),
                "replay": replay, "statistics": statistics,
                "figures": lambda: figures(args.data_root.resolve()),
                "simulations": lambda: simulations(args.only),
                "manuscripts": lambda: manuscripts(args.regenerated)}
    dispatch[args.action]()
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib", "scipy")}
    write_report(f"{args.action}_run.json", {"action": args.action,
        "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "dependencies": versions, "elapsed_seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
