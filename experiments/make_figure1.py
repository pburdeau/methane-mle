"""
Composite Figure 1 — Main results: parameter estimates + T sweep
=================================================================
3 rows × 4 columns:
  Row 1: parameter estimate histograms (p_on, p_off, mu_emit, mu)
  Row 2: T sweep — relative bias (%) of MLE (and POD-weighted for mu)
  Row 3: T sweep — CV (%) of MLE parameters, mu variance + ratio

Reads from:
  results/data_replications.csv   (baseline run)
  results/sweep_T_summary.csv     (T sweep)

Usage:
    python -m experiments.make_figure1
    python -m experiments.make_figure1 --out figures/figure1.pdf
"""
from __future__ import annotations

import argparse
import sys
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

C_POD = "#C39BD3"
C_MLE = "#2ECC71"
C_RATIO = "#1A1A1A"
C_PARAMS = ["#B0B0B0", "#909090", "#C8C8C8", "#2ECC71"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "figures" / "figure1_composite.pdf"))
    parser.add_argument("--res-dir", type=str, default=None,
                        help="Base directory containing results/ and results_linear/")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.res_dir) if args.res_dir else PROJECT_ROOT

    reps = pd.read_csv(base_dir / "results" / "data_replications.csv")
    sweep_path = base_dir / "results_linear" / "sweep_T_summary.csv"
    if not sweep_path.exists():
        sweep_path = base_dir / "results" / "sweep_T_summary.csv"
    sweep = pd.read_csv(sweep_path)
    sweep = sweep[sweep["sweep_val"] <= 2000]

    true_vals = {
        "p_on": cfg.P_ON,
        "p_off": cfg.P_OFF,
        "mu_emit": cfg.MU_EMIT,
        "mu": cfg.TRUE_PARAMS["mu"],
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.8,
    })

    fig, axes = plt.subplots(3, 4, figsize=(7, 7))

    # ── Row 1: parameter histograms ──────────────────────────────
    hist_data = [
        (reps["mle_p_on"].dropna(), true_vals["p_on"],
         r"$\hat{p}_{\mathrm{on}}$", C_PARAMS[0], ""),
        (reps["mle_p_off"].dropna(), true_vals["p_off"],
         r"$\hat{p}_{\mathrm{off}}$", C_PARAMS[1], ""),
        (reps["mle_mu_emit"].dropna(), true_vals["mu_emit"],
         r"$\hat{\mu}_{\mathrm{emit}}$", C_PARAMS[2], " (kg/h)"),
        (reps["mle"].dropna(), true_vals["mu"],
         r"$\hat{\mu}$", C_PARAMS[3], " (kg/h)"),
    ]

    bins_per_panel = [None, None, 30, 30]
    for col, (arr, true, label, color, unit) in enumerate(hist_data):
        ax = axes[0, col]
        arr_vals = arr.values if hasattr(arr, 'values') else np.asarray(arr)
        arr_clean = arr_vals[np.isfinite(arr_vals)]
        uv = np.sort(np.unique(arr_clean))
        if bins_per_panel[col] is None and len(uv) < 50:
            midpts = 0.5 * (uv[:-1] + uv[1:])
            edges = np.concatenate([[uv[0] - (midpts[0] - uv[0])],
                                     midpts,
                                     [uv[-1] + (uv[-1] - midpts[-1])]])
            counts, _ = np.histogram(arr_clean, bins=edges)
            widths = np.diff(edges)
            densities = counts / (len(arr_clean) * widths)
            ax.bar(uv, densities, width=widths, color=color, alpha=0.6,
                   edgecolor="none", linewidth=0, zorder=3, align="center")
        else:
            hist_bins = (np.geomspace(arr_clean.min()*.98, min(arr_clean.max()*1.02,1.0),31)
                         if col < 2 else bins_per_panel[col])
            ax.hist(arr_clean, bins=hist_bins, color=color,
                    alpha=0.6, edgecolor="none", linewidth=0, density=True,
                    rwidth=1.0, zorder=3)
        lbl_true = "True" if col > 0 else "True value"
        lbl_mean = "Mean" if col > 0 else "Mean estimate"
        ax.axvline(true, color="k", lw=1.5, ls="-", label=lbl_true)
        ax.axvline(float(np.mean(arr_clean)), color="k", lw=1.2, ls="--",
                   label=lbl_mean)
        if col < 2:
            ax.set_xscale("log")
        ax.set_xlabel(label + unit)
        if col == 0:
            ax.legend(framealpha=0.9, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Density" if col == 0 else "")
        ax.grid(alpha=0.2, ls=":", lw=0.5)

    # ── Row 2: T sweep — relative bias (%) ─────────────────────
    mle_sw = sweep[sweep["method"] == "mle"].sort_values("sweep_val")
    pod_sw = sweep[sweep["method"] == "pod"].sort_values("sweep_val")
    Ts = mle_sw["sweep_val"].values

    param_tags = [
        ("mle_p_on_bias", "mle_p_on_bias_se", "p_on",
         r"Rel. bias of $p_{\mathrm{on}}$"),
        ("mle_p_off_bias", "mle_p_off_bias_se", "p_off",
         r"Rel. bias of $p_{\mathrm{off}}$"),
        ("mle_mu_emit_bias", "mle_mu_emit_bias_se", "mu_emit",
         r"Rel. bias of $\mu_{\mathrm{emit}}$"),
        ("bias", "bias_se", "mu",
         r"Rel. bias of $\hat{\mu}$"),
    ]

    for col, (bcol, secol, true_key, title) in enumerate(param_tags):
        ax = axes[1, col]
        tv = true_vals[true_key]
        col_color = C_PARAMS[col]

        if bcol in mle_sw.columns:
            b_pct = mle_sw[bcol].values / tv * 100
            se_pct = mle_sw[secol].values / tv * 100
            ax.plot(Ts, b_pct, "o-", color=col_color, markersize=5, label="MLE")
            ax.fill_between(Ts, b_pct - 1.96 * se_pct,
                            b_pct + 1.96 * se_pct,
                            alpha=0.15, color=col_color, linewidth=0)

        if col == 3:
            b_pod_pct = pod_sw["bias"].values / tv * 100
            se_pod_pct = pod_sw["bias_se"].values / tv * 100
            ax.plot(Ts, b_pod_pct, "s-", color=C_POD, markersize=5,
                    label="POD-weighted")
            ax.fill_between(Ts, b_pod_pct - 1.96 * se_pod_pct,
                            b_pod_pct + 1.96 * se_pod_pct,
                            alpha=0.15, color=C_POD, linewidth=0)

        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, ls=":", lw=0.5)
        if col == 0:
            ax.set_ylabel("Bias (%)")
        if col == 3:
            ax.legend(framealpha=0.9)

    # ── Row 3: T sweep — CV (%) + mu variance ratio ─────────────
    cv_tags = [
        ("mle_p_on_var", "mle_p_on_var_se", "p_on",
         r"CV of $p_{\mathrm{on}}$"),
        ("mle_p_off_var", "mle_p_off_var_se", "p_off",
         r"CV of $p_{\mathrm{off}}$"),
        ("mle_mu_emit_var", "mle_mu_emit_var_se", "mu_emit",
         r"CV of $\mu_{\mathrm{emit}}$"),
    ]

    for col, (vcol, secol, true_key, title) in enumerate(cv_tags):
        ax = axes[2, col]
        tv = true_vals[true_key]
        if vcol in mle_sw.columns:
            v = mle_sw[vcol].values
            se = mle_sw[secol].values
            cv_pct = np.sqrt(v) / tv * 100
            cv_lo = np.sqrt(np.maximum(v - 1.96 * se, 0)) / tv * 100
            cv_hi = np.sqrt(v + 1.96 * se) / tv * 100
            ax.plot(Ts, cv_pct, "o-", color=C_PARAMS[col], markersize=5)
            ax.fill_between(Ts, cv_lo, cv_hi,
                            alpha=0.15, color=C_PARAMS[col], linewidth=0)
        ax.set_title(title)
        ax.set_xlabel(r"$T$ (time steps)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, ls=":", lw=0.5)
        if col == 0:
            ax.set_ylabel("CV (%)")

    # Column 4, row 3: mu CV (%) for MLE and POD-weighted + ratio
    ax = axes[2, 3]
    tv_mu = true_vals["mu"]
    v_pod = pod_sw["variance"].values
    se_pod_v = pod_sw["variance_se"].values
    v_mle = mle_sw["variance"].values
    se_mle_v = mle_sw["variance_se"].values

    cv_pod = np.sqrt(v_pod) / tv_mu * 100
    cv_mle = np.sqrt(v_mle) / tv_mu * 100
    cv_pod_lo = np.sqrt(np.maximum(v_pod - 1.96 * se_pod_v, 0)) / tv_mu * 100
    cv_pod_hi = np.sqrt(v_pod + 1.96 * se_pod_v) / tv_mu * 100
    cv_mle_lo = np.sqrt(np.maximum(v_mle - 1.96 * se_mle_v, 0)) / tv_mu * 100
    cv_mle_hi = np.sqrt(v_mle + 1.96 * se_mle_v) / tv_mu * 100

    ax.plot(Ts, cv_pod, "s-", color=C_POD, markersize=5, label="POD-weighted")
    ax.fill_between(Ts, cv_pod_lo, cv_pod_hi,
                    alpha=0.15, color=C_POD, linewidth=0)
    ax.plot(Ts, cv_mle, "o-", color=C_MLE, markersize=5, label="MLE")
    ax.fill_between(Ts, cv_mle_lo, cv_mle_hi,
                    alpha=0.15, color=C_MLE, linewidth=0)

    ratio = v_pod / v_mle
    ax_r = ax.twinx()
    ax_r.plot(Ts, ratio, "D--", color=C_RATIO, markersize=5, alpha=0.7,
              label="Var. ratio")
    ax_r.set_ylabel("Variance ratio", color=C_RATIO)
    ax_r.tick_params(axis="y", labelcolor=C_RATIO)
    ax_r.axhline(1, color="gray", ls=":", lw=1, alpha=0.5)

    ax.set_title(r"CV of $\hat{\mu}$ + ratio")
    ax.set_xlabel(r"$T$ (time steps)")
    ax.set_ylabel("CV (%)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax_r.legend(loc="center right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha=0.2, ls=":", lw=0.5)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved composite: {out}")
    print(f"Saved composite: {out.with_suffix('.png')}")

    # ── Histograms-only figure (main paper Figure 1) ─────────────
    fig_h, axes_h = plt.subplots(1, 4, figsize=(7, 2))
    for idx, (arr, true, label, color, unit) in enumerate(hist_data):
        ax = axes_h[idx]
        arr_vals = arr.values if hasattr(arr, 'values') else np.asarray(arr)
        arr_clean = arr_vals[np.isfinite(arr_vals)]
        uv = np.sort(np.unique(arr_clean))
        if bins_per_panel[idx] is None and len(uv) < 50:
            midpts = 0.5 * (uv[:-1] + uv[1:])
            edges = np.concatenate([[uv[0] - (midpts[0] - uv[0])],
                                     midpts,
                                     [uv[-1] + (uv[-1] - midpts[-1])]])
            counts, _ = np.histogram(arr_clean, bins=edges)
            widths = np.diff(edges)
            densities = counts / (len(arr_clean) * widths)
            ax.bar(uv, densities, width=widths, color=color, alpha=0.6,
                   edgecolor="none", linewidth=0, zorder=3, align="center")
        else:
            hist_bins = (np.geomspace(arr_clean.min()*.98, min(arr_clean.max()*1.02,1.0),31)
                         if idx < 2 else bins_per_panel[idx])
            ax.hist(arr_clean, bins=hist_bins, color=color,
                    alpha=0.6, edgecolor="none", linewidth=0, density=True,
                    rwidth=1.0, zorder=3)
        est_mean = float(np.mean(arr_clean))
        ax.axvline(true, color="k", lw=1.5, ls="-",
                   label=f"True = {true:.4g}")
        ax.axvline(est_mean, color="k", lw=1.2, ls="--",
                   label=f"Mean = {est_mean:.4g}")
        panel_label = chr(97 + idx)  # a, b, c, d
        ax.set_title(f"({panel_label})", loc="left", fontsize=9,
                     fontweight="bold")
        if idx < 2:
            ax.set_xscale("log")
        ax.set_xlabel(label + unit)
        ax.legend(framealpha=0.9, fontsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Density")
        ax.grid(alpha=0.2, ls=":", lw=0.5)

    fig_h.tight_layout()
    hist_out = out.parent / "figure1_histograms.pdf"
    fig_h.savefig(hist_out, format="pdf", bbox_inches="tight", dpi=300)
    fig_h.savefig(hist_out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig_h)
    print(f"Saved histograms: {hist_out}")
    print(f"Saved histograms: {hist_out.with_suffix('.png')}")



if __name__ == "__main__":
    main()
