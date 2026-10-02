#!/usr/bin/env python3
"""
Master script — regenerate every figure for the JRSS-A methane MLE paper.

All outputs go into a single self-contained directory whose name encodes the
key parameters, e.g.:

    revision-psnap0125-nudge1015/
        figures/          <- all PDFs
        results/          <- cached experiment CSVs
        results_linear/   <- sweep CSVs

Usage:
    python run_all_figures.py                    # everything
    python run_all_figures.py --only baseline    # just one figure
    python run_all_figures.py --force            # re-run even if cached
    SKIP=misspec_joint python run_all_figures.py # skip one figure
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg

# Central output directory (encodes key parameters)
RUN_TAG = (f"revision-psnap{str(cfg.P_SNAP).replace('.','')}"
           f"-nudge{str(cfg.NUDGE).replace('.','')}")
RUN_DIR = PROJECT_ROOT / RUN_TAG

FIGURES_DIR = RUN_DIR / "figures"
RESULTS_DIR = RUN_DIR / "results"
RESULTS_LINEAR_DIR = RUN_DIR / "results_linear"
PAPER_FIG_DIR = PROJECT_ROOT / "mle_methane_paper" / "figures"
SI_FIG_DIR = PROJECT_ROOT / "Supplementary Methane MLE" / "figures"

for d in (FIGURES_DIR, RESULTS_DIR, RESULTS_LINEAR_DIR,
          PAPER_FIG_DIR, SI_FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"\n  Output directory: {RUN_DIR}")
print(f"  Tag: {RUN_TAG}\n")


def banner(fig_id: str, description: str):
    print(f"\n{'=' * 70}")
    print(f"  [{fig_id}] {description}")
    print(f"  Baseline: P_SNAP={cfg.P_SNAP}, NUDGE={cfg.NUDGE}, "
          f"SEED={cfg.BASE_SEED}")
    print(f"{'=' * 70}\n", flush=True)


def run_cmd(cmd: list[str], fig_id: str):
    """Run a subprocess, stream output, return elapsed seconds."""
    t0 = time.time()
    print(f"  $ {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  *** [{fig_id}] FAILED (exit {result.returncode}) "
              f"after {elapsed:.0f}s ***")
    else:
        print(f"\n  [{fig_id}] completed in {elapsed:.0f}s")
    return elapsed


def copy_to_paper(filenames: list[str], si=False):
    """Copy figure PDFs from figures/ to the paper directories."""
    for fn in filenames:
        src = FIGURES_DIR / fn
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping copy")
            continue
        dests = [PAPER_FIG_DIR / fn] if not si else [SI_FIG_DIR / fn]
        for dst in dests:
            shutil.copy2(src, dst)
            print(f"  Copied → {dst.relative_to(PROJECT_ROOT)}")


def write_log(fig_id: str, elapsed: float, **kwargs):
    """Write a small summary log next to the figure."""
    log_path = FIGURES_DIR / f"{fig_id}.log"
    lines = [
        f"figure_id: {fig_id}",
        f"baseline: P_SNAP={cfg.P_SNAP}, P_CONT={cfg.P_CONT}, "
        f"T_CONT={cfg.T_CONT}, T={cfg.T}",
        f"nudge: {cfg.NUDGE}",
        f"seed: {cfg.BASE_SEED}",
        f"elapsed_seconds: {elapsed:.1f}",
    ]
    for k, v in kwargs.items():
        lines.append(f"{k}: {v}")
    log_path.write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════
# Individual figure runners
# ══════════════════════════════════════════════════════════════════

def fig_baseline(force=False):
    """Figure 2 (violin+bars), histograms, realizations, POD curves, size dist."""
    fig_id = "baseline"
    banner(fig_id, "Baseline comparison — 25x500 nested (Figure 2 + SI)")
    csv = RESULTS_DIR / "data_replications.csv"
    need_sim = force or not csv.exists()
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_baseline",
        "--n-reps", "500", "--n-outer", "25",
        "--nudge", str(cfg.NUDGE),
        "--output-dir", str(RUN_DIR),
    ], fig_id)
    # run_baseline saves figure_baseline_violin_v2.pdf; rename to figure2.pdf
    src = FIGURES_DIR / "figure_baseline_violin_v2.pdf"
    dst = FIGURES_DIR / "figure2.pdf"
    if src.exists():
        shutil.copy2(src, dst)
    copy_to_paper(["figure2.pdf"])
    copy_to_paper([
        "figure_pod_curves.pdf",
        "figure_size_distribution.pdf",
        "figure_realizations.pdf",
    ], si=True)
    write_log(fig_id, elapsed, n_outer=25, n_inner=500)


def fig_sparse_realization(force=False):
    """Figure C — single very-sparse realization (SI)."""
    fig_id = "sparse_realization"
    banner(fig_id, "Sparse realization (p_snap=0.0015, baseline p_cont)")
    elapsed = run_cmd([
        sys.executable, str(PROJECT_ROOT / "experiments" / "_sparse_realization.py"),
        str(FIGURES_DIR),
    ], fig_id)
    copy_to_paper(["figure_sparse_realization.pdf"], si=True)
    write_log(fig_id, elapsed)


def fig_sweep_T(force=False):
    """T sweep — needed for composite figure and precision figure."""
    fig_id = "sweep_T"
    banner(fig_id, "T sweep (for composite + precision)")
    csv = RESULTS_LINEAR_DIR / "sweep_T_summary.csv"
    if not force and csv.exists():
        print(f"  Using cached: {csv}")
        return
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_sweeps",
        "--sweep", "T", "--n-inner", "500", "--n-outer", "25",
        "--fig-dir", str(FIGURES_DIR),
        "--res-dir", str(RESULTS_LINEAR_DIR),
    ], fig_id)
    write_log(fig_id, elapsed, sweep="T", n_inner=500, n_outer=25)


def fig_sweep_psnap(force=False):
    """p_snap sweep — NEW Figure A, panel of main-text Figure 4."""
    fig_id = "sweep_psnap"
    banner(fig_id, "p_snap sweep (Figure 4 panel + dual x-axis)")
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_sweep_psnap",
        "--fig-dir", str(FIGURES_DIR),
        "--res-dir", str(RESULTS_DIR),
    ], fig_id)
    copy_to_paper(["figure_sweep_psnap.pdf"])
    write_log(fig_id, elapsed)


def fig_sweep_tau_theta(force=False):
    """tau_emit and theta_snap sweeps — Figure 4 panels."""
    fig_id = "sweep_tau_theta"
    banner(fig_id, "tau_emit + theta_snap sweeps (Figure 4)")
    for sweep in ("tau_emit", "theta_snap"):
        csv = RESULTS_LINEAR_DIR / f"sweep_{sweep}_summary.csv"
        if not force and csv.exists():
            print(f"  Using cached: {csv}")
            continue
        elapsed = run_cmd([
            sys.executable, "-m", "experiments.run_sweeps",
            "--sweep", sweep, "--n-inner", "500", "--n-outer", "25",
            "--fig-dir", str(FIGURES_DIR),
            "--res-dir", str(RESULTS_LINEAR_DIR),
        ], fig_id)
    write_log(fig_id, 0)


def fig_sweep_pcont_Tcont(force=False):
    """p_cont and T_cont sweeps — SI figures."""
    fig_id = "sweep_pcont_Tcont"
    banner(fig_id, "p_cont + T_cont sweeps (SI)")
    for sweep in ("p_cont", "T_cont"):
        csv = RESULTS_LINEAR_DIR / f"sweep_{sweep}_summary.csv"
        if not force and csv.exists():
            print(f"  Using cached: {csv}")
            continue
        elapsed = run_cmd([
            sys.executable, "-m", "experiments.run_sweeps",
            "--sweep", sweep, "--n-inner", "500", "--n-outer", "25",
            "--fig-dir", str(FIGURES_DIR),
            "--res-dir", str(RESULTS_LINEAR_DIR),
        ], fig_id)
    # Rename to match tex references
    rename_map = {
        "figure_sweep_p_cont.pdf": "figure_sweep_pcont.pdf",
        "figure_sweep_T_cont.pdf": "figure_sweep_Tcont.pdf",
    }
    for old_name, new_name in rename_map.items():
        src = FIGURES_DIR / old_name
        dst = FIGURES_DIR / new_name
        if src.exists():
            shutil.copy2(src, dst)
    copy_to_paper(["figure_sweep_pcont.pdf", "figure_sweep_Tcont.pdf"], si=True)
    write_log(fig_id, 0)


def fig_grid_robustness(force=False):
    """Grid resolution robustness — SI."""
    fig_id = "grid_robustness"
    banner(fig_id, "Grid resolution robustness (SI)")
    csv = RESULTS_DIR / "grid_robustness.csv"
    if not force and csv.exists():
        print(f"  Using cached: {csv}")
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_grid_robustness",
        "--fig-dir", str(FIGURES_DIR),
        "--res-dir", str(RESULTS_DIR),
    ], fig_id)
    copy_to_paper(["figure_grid_robustness.pdf"], si=True)
    write_log(fig_id, elapsed)


def fig_threshold_sweep(force=False):
    """Plume-linking threshold sensitivity — SI."""
    fig_id = "threshold_sweep"
    banner(fig_id, "Threshold sensitivity (SI)")
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_threshold_sweep",
        "--fig-dir", str(FIGURES_DIR),
        "--res-dir", str(RESULTS_DIR),
    ], fig_id)
    copy_to_paper(["figure_threshold_sweep.pdf"], si=True)
    write_log(fig_id, elapsed)


def fig_assembly(force=False):
    """Assembly: composite Figure 1, Figure 3 (sweeps), precision Figure 5."""
    fig_id = "assembly"
    banner(fig_id, "Assembly: composite, sweeps figure, precision")

    # Figure 1 composite + histograms
    run_cmd([
        sys.executable, "-m", "experiments.make_figure1",
        "--out", str(FIGURES_DIR / "figure1_composite.pdf"),
        "--res-dir", str(RUN_DIR),
    ], fig_id)
    copy_to_paper(["figure1_histograms.pdf"])
    copy_to_paper(["figure1_composite.pdf"], si=True)

    # Figure 3 (3-panel sweeps: p_snap, tau_emit, theta_snap)
    run_cmd([
        sys.executable, "-m", "experiments.make_figure_sweeps",
        "--res-dir", str(RUN_DIR),
        "--out", str(FIGURES_DIR / "figure3.pdf"),
    ], fig_id)
    copy_to_paper(["figure3.pdf"])

    # Figure 5 (precision vs sample size)
    run_cmd([
        sys.executable, "-m", "experiments.make_figure_precision",
        "--res-dir", str(RUN_DIR),
        "--out", str(FIGURES_DIR / "figure_precision.pdf"),
    ], fig_id)
    copy_to_paper(["figure_precision.pdf"])

    write_log(fig_id, 0)


def fig_misspec_joint(force=False):
    """Figure E — joint sensor misspecification (SI)."""
    fig_id = "misspec_joint"
    banner(fig_id, "Joint sensor misspecification (Figure E)")
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_misspec",
        "--mode", "joint",
        "--fig-dir", str(FIGURES_DIR),
        "--res-dir", str(RESULTS_DIR),
    ], fig_id)
    copy_to_paper(["figure_misspec.pdf"], si=True)
    write_log(fig_id, elapsed)


def fig_misspec_onebyone(force=False):
    """Figure F — per-parameter sensor misspecification (SI)."""
    fig_id = "misspec_onebyone"
    banner(fig_id, "Per-parameter sensor misspecification (Figure F)")
    elapsed = run_cmd([
        sys.executable, "-m", "experiments.run_misspec",
        "--mode", "onebyone",
        "--fig-dir", str(FIGURES_DIR),
        "--res-dir", str(RESULTS_DIR),
    ], fig_id)
    copy_to_paper(["figure_misspec_onebyone.pdf"], si=True)
    write_log(fig_id, elapsed)


# ══════════════════════════════════════════════════════════════════
# Registry and main
# ══════════════════════════════════════════════════════════════════

FIGURES = [
    ("baseline",           fig_baseline),
    ("sparse_realization", fig_sparse_realization),
    ("sweep_T",            fig_sweep_T),
    ("sweep_psnap",        fig_sweep_psnap),
    ("sweep_tau_theta",    fig_sweep_tau_theta),
    ("sweep_pcont_Tcont",  fig_sweep_pcont_Tcont),
    ("grid_robustness",    fig_grid_robustness),
    ("threshold_sweep",    fig_threshold_sweep),
    ("assembly",           fig_assembly),
    ("misspec_joint",      fig_misspec_joint),
    ("misspec_onebyone",   fig_misspec_onebyone),
]


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all paper figures")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this figure ID")
    parser.add_argument("--force", action="store_true",
                        help="Re-run experiments even if cached")
    parser.add_argument("--list", action="store_true",
                        help="List all figure IDs and exit")
    args = parser.parse_args()

    if args.list:
        for fig_id, _ in FIGURES:
            print(f"  {fig_id}")
        return

    skip_env = os.environ.get("SKIP", "").split(",")
    skip_env = [s.strip() for s in skip_env if s.strip()]

    t0_global = time.time()
    for fig_id, func in FIGURES:
        if args.only and fig_id != args.only:
            continue
        if fig_id in skip_env:
            print(f"\n  SKIPPED [{fig_id}] (SKIP env var)")
            continue
        func(force=args.force)

    total = time.time() - t0_global
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — total elapsed: {total:.0f}s "
          f"({total / 3600:.1f}h)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
