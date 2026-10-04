"""Numerical sensitivity of conditional transition optimization.

Keeps all empirical emission-size support points; varies optimizer stopping
accuracy on the same 500 baseline datasets. The legacy output filename is
retained to avoid changing Overleaf figure references.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "methane-mpl-cache"))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from experiments import plot_style as ps
from src import (
    build_baseline_specs,
    simulate_series,
    generate_masks,
    loop_empirical,
)

N_REPS = 500

OPTIMIZER_TOLERANCES = [1e-7, 1e-9, 1e-11, 1e-13]

PARAMS = [
    ("p_on",    cfg.P_ON,    r"$p_{\mathrm{on}}$"),
    ("p_off",   cfg.P_OFF,   r"$p_{\mathrm{off}}$"),
    ("mu_emit", cfg.MU_EMIT, r"$\mu_{\mathrm{emit}}$"),
    ("mu",      cfg.TRUE_PARAMS["mu"], r"$\hat{\mu}$"),
]

COLORS = {
    "p_on": ps.PARAMETERS[0], "p_off": ps.PARAMETERS[1],
    "mu_emit": ps.PARAMETERS[2], "mu": ps.PARAMETERS[3],
}


def _run_one_task(task):
    specs,rep,ftol=task
    sm,cm=generate_masks(cfg.T,cfg.P_SNAP,cfg.P_CONT,cfg.T_CONT,np.random.default_rng(cfg.BASE_SEED+rep))
    obs,*_=simulate_series(cfg.T,cfg.P_ON,cfg.P_OFF,cfg.SIZE_MU,cfg.SIZE_SIGMA,specs,sm,cm,
                          np.random.default_rng(cfg.BASE_SEED+rep+100_000))
    res=loop_empirical(obs,specs,cfg.T, **cfg.MLE_OPTIONS,optimizer_tolerance=ftol)
    return dict(p_on=res['p_on'],p_off=res['p_off'],mu_emit=res['mu_emit'],mu=res['mean'])


def run_sweep(specs):
    from experiments.parallel import ordered_map
    records=[]
    for ftol in OPTIMIZER_TOLERANCES:
        results=list(ordered_map(_run_one_task,[(specs,rep,ftol) for rep in range(N_REPS)]))
        for key,true,_ in PARAMS:
            x=np.array([r[key] for r in results]);x=x[np.isfinite(x)]
            records.append(dict(sweep='resolution',sweep_val=-np.log10(ftol),
                optimizer_ftol=ftol,param=key,true=true,mean=x.mean(),bias=x.mean()-true,
                rel_bias_pct=100*(x.mean()-true)/true,variance=x.var(),
                cv_pct=100*x.std()/true,n_valid=len(x)))
        print('Optimizer tolerance',ftol,'complete',flush=True)
    return records


def make_figure(df, out_path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    df_res = df[df["sweep"] == "resolution"]
    sub = df_res[df_res["param"] == "mu"].sort_values("sweep_val")
    c = COLORS["mu"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.plot(sub["sweep_val"], sub["rel_bias_pct"], "o-",
            color=c, markersize=8, markeredgecolor="white", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel(r"Optimizer tolerance $-\log_{10}(\mathrm{ftol})$")
    ax.set_ylabel("Relative bias (%)")
    ax.set_title(r"(a) Bias of $\hat{\mu}$", fontweight="bold")
    ax.set_xticks(sorted(sub["sweep_val"].unique()))
    ax.grid(alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(sub["sweep_val"], sub["cv_pct"], "o-",
            color=c, markersize=8, markeredgecolor="white", lw=2)
    ax.set_xlabel(r"Optimizer tolerance $-\log_{10}(\mathrm{ftol})$")
    ax.set_ylabel("CV (%)")
    ax.set_title(r"(b) CV of $\hat{\mu}$", fontweight="bold")
    ax.set_xticks(sorted(sub["sweep_val"].unique()))
    ax.grid(alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=str, default=None)
    parser.add_argument("--res-dir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    pargs = parser.parse_args()

    fig_dir = Path(pargs.fig_dir) if pargs.fig_dir else PROJECT_ROOT / "figures"
    res_dir = Path(pargs.res_dir) if pargs.res_dir else PROJECT_ROOT / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if pargs.plot_only:
        make_figure(pd.read_csv(res_dir / "grid_robustness.csv"),
                    fig_dir / "figure_grid_robustness.pdf")
        return

    tech_specs = build_baseline_specs(
        aerial_threshold=cfg.AERIAL_THRESHOLD,
        aerial_slope=cfg.AERIAL_SLOPE,
        aerial_sensor_sigma=cfg.AERIAL_SENSOR_SIGMA,
        cont_threshold=cfg.CONT_THRESHOLD,
        cont_slope=cfg.CONT_SLOPE,
        cont_sensor_sigma=cfg.CONT_SENSOR_SIGMA,
        cont_fp_rate=cfg.CONT_FP_RATE,
        cont_fp_scale=cfg.CONT_FP_SCALE,
    )

    all_records = run_sweep(tech_specs)

    df = pd.DataFrame(all_records)
    out_csv = res_dir / "grid_robustness.csv"
    df.to_csv(out_csv, index=False, float_format="%.12g")
    print(f"\nSaved: {out_csv}")

    out_fig = fig_dir / "figure_grid_robustness.pdf"
    make_figure(df, out_fig)


if __name__ == "__main__":
    main()
