"""
Plume-linking threshold sensitivity sweep
==========================================
Sweeps the Bayesian posterior decision threshold used to link
consecutive detections into the same plume. Baseline = 0.5.

Usage:
    python -m experiments.run_threshold_sweep
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

N_INNER = 500
N_OUTER = 5
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _run_one_task(task):
    return run_one(*task)


def run_one(tech_specs, rep, p_on_grid, p_off_grid, threshold):
    mask_rng = np.random.default_rng(cfg.BASE_SEED + rep)
    emit_rng = np.random.default_rng(cfg.BASE_SEED + rep + 100_000)

    snap_mask, cont_mask = generate_masks(
        cfg.T, cfg.P_SNAP, cfg.P_CONT, cfg.T_CONT, mask_rng)

    obs, _st, _sz, _eid = simulate_series(
        T=cfg.T, p_on=cfg.P_ON, p_off=cfg.P_OFF,
        size_mu=cfg.SIZE_MU, size_sigma=cfg.SIZE_SIGMA,
        tech_specs=tech_specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=emit_rng)


    res = loop_empirical(obs, tech_specs, cfg.T, **cfg.MLE_OPTIONS,
                         max_gap=cfg.MAX_GAP,
                         decision_threshold=threshold,
                         p_on_grid=p_on_grid, p_off_grid=p_off_grid,
                         p_off_nudge=cfg.NUDGE)

    return dict(mu=res["mean"], p_on=res["p_on"],
                p_off=res["p_off"], mu_emit=res["mu_emit"])


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
        make_figure(pd.read_csv(res_dir / "threshold_sweep.csv"),
                    fig_dir / "figure_threshold_sweep.pdf")
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

    p_on_grid = None  # fixed-bound continuous optimization
    p_off_grid = None

    mu_true = cfg.TRUE_PARAMS["mu"]
    mu_emit_true = cfg.MU_EMIT
    records = []

    for threshold in THRESHOLDS:
        print(f"\n{'=' * 62}")
        print(f"  Decision threshold = {threshold:.2f}")
        print(f"{'=' * 62}")

        bias_outer = []
        var_outer = []
        me_bias_outer = []
        t0 = time.time()

        for outer in range(N_OUTER):
            ests_mu = []
            ests_me = []
            from experiments.parallel import ordered_map
            tasks = [(tech_specs,outer*N_INNER+inner,None,None,threshold)
                     for inner in range(N_INNER)]
            for r in ordered_map(_run_one_task,tasks):
                ests_mu.append(r["mu"])
                ests_me.append(r["mu_emit"])

            mu_arr = np.array(ests_mu)
            me_arr = np.array(ests_me)
            mu_valid = mu_arr[np.isfinite(mu_arr)]
            me_valid = me_arr[np.isfinite(me_arr)]

            bias_outer.append(float(np.mean(mu_valid) - mu_true))
            var_outer.append(float(np.var(mu_valid, ddof=0)))
            me_bias_outer.append(float(np.mean(me_valid) - mu_emit_true))

            el = time.time() - t0
            eta = el / (outer + 1) * (N_OUTER - outer - 1)
            print(f"    outer {outer+1}/{N_OUTER}  "
                  f"elapsed={el:.0f}s  eta={eta:.0f}s", flush=True)

        b_arr = np.array(bias_outer)
        v_arr = np.array(var_outer)
        me_b = np.array(me_bias_outer)
        n_o = len(b_arr)

        records.append(dict(
            threshold=threshold,
            mu_bias=float(np.mean(b_arr)),
            mu_bias_se=float(np.std(b_arr, ddof=1) / np.sqrt(n_o)),
            mu_variance=float(np.mean(v_arr)),
            mu_variance_se=float(np.std(v_arr, ddof=1) / np.sqrt(n_o)),
            mu_emit_bias=float(np.mean(me_b)),
            mu_emit_bias_se=float(np.std(me_b, ddof=1) / np.sqrt(n_o)),
        ))

        print(f"  mu:      bias={np.mean(b_arr):+.3f} ± {np.std(b_arr,ddof=1)/np.sqrt(n_o):.3f}  "
              f"var={np.mean(v_arr):.2f}")
        print(f"  mu_emit: bias={np.mean(me_b):+.3f}")

    df = pd.DataFrame(records)
    out_csv = res_dir / "threshold_sweep.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nSaved: {out_csv}")

    make_figure(df, fig_dir / "figure_threshold_sweep.pdf")


def make_figure(df, out_fig):
    """Render the original threshold figure from saved summary data."""
    mu_true = cfg.TRUE_PARAMS["mu"]
    mu_emit_true = cfg.MU_EMIT
    from scipy.stats import t
    z = float(t.ppf(.975, N_OUTER-1))
    # ── Figure ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    thrs = df["threshold"].values
    mu_true_val = mu_true

    # Relative bias (%)
    b_pct = df["mu_bias"].values / mu_true_val * 100
    bse_pct = df["mu_bias_se"].values / mu_true_val * 100
    ax1.plot(thrs, b_pct, "o-", color="#2ECC71", markersize=7,
             markeredgecolor="white", label=r"$\hat{\mu}$")
    ax1.fill_between(thrs, b_pct - z * bse_pct, b_pct + z * bse_pct,
                     alpha=0.15, color="#2ECC71")

    me_b_pct = df["mu_emit_bias"].values / mu_emit_true * 100
    me_bse_pct = df["mu_emit_bias_se"].values / mu_emit_true * 100
    ax1.plot(thrs, me_b_pct, "s-", color="#9B59B6", markersize=7,
             markeredgecolor="white", label=r"$\hat{\mu}_{\mathrm{emit}}$")
    ax1.fill_between(thrs, me_b_pct - z * me_bse_pct,
                     me_b_pct + z * me_bse_pct,
                     alpha=0.15, color="#9B59B6")

    ax1.axhline(0, color="gray", ls="--", lw=1)
    ax1.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Relative bias (%)")
    ax1.set_title("(a) Bias vs Linking Threshold", fontweight="bold")
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(alpha=0.3, ls=":")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # CV (%)
    cv_pct = np.sqrt(df["mu_variance"].values) / mu_true_val * 100
    v = df["mu_variance"].values
    vse = df["mu_variance_se"].values
    cv_lo = np.sqrt(np.maximum(v - z * vse, 0)) / mu_true_val * 100
    cv_hi = np.sqrt(v + z * vse) / mu_true_val * 100
    ax2.plot(thrs, cv_pct, "o-", color="#2ECC71", markersize=7,
             markeredgecolor="white")
    ax2.fill_between(thrs, cv_lo, cv_hi, alpha=0.15, color="#2ECC71")

    ax2.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("CV (%)")
    ax2.set_title("(b) CV vs Linking Threshold", fontweight="bold")
    ax2.grid(alpha=0.3, ls=":")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_fig, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_fig.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
