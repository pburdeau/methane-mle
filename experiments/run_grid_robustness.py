"""
Grid robustness check: resolution
============================================================
Vary the number of grid points G while keeping the geometric ratio
between consecutive points fixed.

Reports relative bias (%) and CV (%) for p_on, p_off, mu_emit, and mu.

Usage:
    python -m experiments.run_grid_robustness
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
from src import (
    build_baseline_specs,
    simulate_series,
    generate_masks,
    loop_empirical,
)

N_REPS = 500

RESOLUTION_G_VALUES = [5, 9, 15, 25, 41]

PARAMS = [
    ("p_on",    cfg.P_ON,    r"$p_{\mathrm{on}}$"),
    ("p_off",   cfg.P_OFF,   r"$p_{\mathrm{off}}$"),
    ("mu_emit", cfg.MU_EMIT, r"$\mu_{\mathrm{emit}}$"),
    ("mu",      cfg.TRUE_PARAMS["mu"], r"$\hat{\mu}$"),
]

COLORS = {
    "p_on": "#9B59B6", "p_off": "#E67E22",
    "mu_emit": "#2ECC71", "mu": "#3498DB",
}


def run_one(tech_specs, rep, p_on_grid, p_off_grid):
    mask_rng = np.random.default_rng(cfg.BASE_SEED + rep)
    emit_rng = np.random.default_rng(cfg.BASE_SEED + rep + 100_000)

    snap_mask, cont_mask = generate_masks(
        cfg.T, cfg.P_SNAP, cfg.P_CONT, cfg.T_CONT, mask_rng)

    obs, _st, _sz, _eid = simulate_series(
        T=cfg.T, p_on=cfg.P_ON, p_off=cfg.P_OFF,
        size_mu=cfg.SIZE_MU, size_sigma=cfg.SIZE_SIGMA,
        tech_specs=tech_specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=emit_rng)

    if obs.n_total_detections() < 2:
        return dict(p_on=np.nan, p_off=np.nan, mu_emit=np.nan, mu=np.nan)

    res = loop_empirical(obs, tech_specs, cfg.T,
                         max_gap=cfg.MAX_GAP,
                         decision_threshold=cfg.DECISION_THRESHOLD,
                         p_on_grid=p_on_grid, p_off_grid=p_off_grid,
                         p_off_nudge=cfg.NUDGE)

    return dict(p_on=res["p_on"], p_off=res["p_off"],
                mu_emit=res["mu_emit"], mu=res["mean"])


def run_sweep(tech_specs, sweep_name, configs):
    """Run a sweep and return a list of result dicts.

    configs: list of (sweep_val, display_label, p_on_grid, p_off_grid)
    """
    records = []
    for sweep_val, display, p_on_grid, p_off_grid in configs:
        step = p_on_grid[1] / p_on_grid[0] if len(p_on_grid) > 1 else 0

        print(f"\n{'=' * 62}")
        print(f"  {display}")
        print(f"  p_on grid:  [{p_on_grid[0]:.6f} ... {p_on_grid[-1]:.6f}]"
              f"  ({len(p_on_grid)} pts, step={step:.3f}x)")
        print(f"  p_off grid: [{p_off_grid[0]:.6f} ... {p_off_grid[-1]:.6f}]"
              f"  ({len(p_off_grid)} pts)")
        print(f"{'=' * 62}")

        results = []
        t0 = time.time()
        for rep in range(N_REPS):
            if (rep + 1) % 100 == 0:
                el = time.time() - t0
                eta = el / (rep + 1) * (N_REPS - rep - 1)
                print(f"  rep {rep+1}/{N_REPS}  elapsed={el:.0f}s  "
                      f"eta={eta:.0f}s", flush=True)
            results.append(run_one(tech_specs, rep, p_on_grid, p_off_grid))

        for key, true_val, _ in PARAMS:
            arr = np.array([r[key] for r in results])
            valid = arr[np.isfinite(arr)]
            bias = float(np.mean(valid) - true_val)
            var = float(np.var(valid, ddof=0))
            rel_bias = bias / true_val * 100
            cv = np.sqrt(var) / true_val * 100
            records.append(dict(
                sweep=sweep_name, sweep_val=sweep_val,
                param=key, true=true_val,
                mean=float(np.mean(valid)), bias=bias,
                rel_bias_pct=rel_bias,
                variance=var, cv_pct=cv,
                n_valid=len(valid),
            ))
            print(f"  {key:>8s}: mean={np.mean(valid):.4f}  "
                  f"rel_bias={rel_bias:+.2f}%  CV={cv:.1f}%")
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
    ax.set_xlabel("Number of grid points $G$")
    ax.set_ylabel("Relative bias (%)")
    ax.set_title(r"(a) Bias of $\hat{\mu}$", fontweight="bold")
    ax.set_xticks(sorted(sub["sweep_val"].unique()))
    ax.grid(alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(sub["sweep_val"], sub["cv_pct"], "o-",
            color=c, markersize=8, markeredgecolor="white", lw=2)
    ax.set_xlabel("Number of grid points $G$")
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

    all_records = []

    print("\n" + "=" * 62)
    print(f"  Resolution Sweep (fixed step r={cfg.P_GEOM_FACTOR})")
    print("=" * 62)

    configs_res = []
    for G in RESOLUTION_G_VALUES:
        p_on_grid = cfg.build_geom_grid(cfg.P_ON, cfg.P_GEOM_FACTOR, G)
        p_off_grid = cfg.build_geom_grid(cfg.P_OFF, cfg.P_GEOM_FACTOR, G)
        step = p_on_grid[1] / p_on_grid[0] if G > 1 else 1.0
        spread = cfg.P_GEOM_FACTOR ** ((G - 1) / 2)
        label = f"G={G} (step={step:.3f}x, spread={spread:.2f}x)"
        configs_res.append((G, label, p_on_grid, p_off_grid))

    all_records.extend(run_sweep(tech_specs, "resolution", configs_res))

    df = pd.DataFrame(all_records)
    out_csv = res_dir / "grid_robustness.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nSaved: {out_csv}")

    out_fig = fig_dir / "figure_grid_robustness.pdf"
    make_figure(df, out_fig)


if __name__ == "__main__":
    main()
