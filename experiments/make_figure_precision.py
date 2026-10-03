"""
Figure — Required sample size for target precision
===================================================
Single-panel figure showing the number of site visits needed
to estimate mu within +/- eps%, for MLE vs POD-weighted.

Three annotated examples at eps = 3%, 5%, 10%.

Reads from:
    results/data_replications.csv

Usage:
    python -m experiments.make_figure_precision
"""
from __future__ import annotations

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
Z = 1.96


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", type=str, default=None,
                        help="Directory containing results/data_replications.csv")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PDF path")
    pargs = parser.parse_args()

    base_dir = Path(pargs.res_dir) if pargs.res_dir else PROJECT_ROOT
    out_path = Path(pargs.out) if pargs.out else PROJECT_ROOT / "figures" / "figure_precision.pdf"

    reps = pd.read_csv(base_dir / "results" / "data_replications.csv")
    mu_true = cfg.TRUE_PARAMS["mu"]

    n_outer = 25
    n_inner = 500

    methods = {
        "POD-weighted": reps["pod_weighted"].values,
        "MLE": reps["mle"].values,
    }

    var_est = {}
    var_lo = {}
    var_hi = {}
    for name, arr in methods.items():
        outer_vars = []
        for o in range(n_outer):
            batch = arr[o * n_inner : (o + 1) * n_inner]
            valid = batch[np.isfinite(batch)]
            if len(valid) > 1:
                outer_vars.append(float(np.var(valid, ddof=0)))
        v_arr = np.array(outer_vars)
        var_est[name] = float(np.mean(v_arr))
        var_lo[name] = float(np.percentile(v_arr, 2.5))
        var_hi[name] = float(np.percentile(v_arr, 97.5))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "lines.linewidth": 2.0,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    eps_range = np.linspace(0.01, 0.20, 100)

    for name in ["POD-weighted", "MLE"]:
        V = var_est[name]
        V_p10 = var_lo[name]
        V_p90 = var_hi[name]

        N_req = (Z / eps_range) ** 2 * V / mu_true ** 2
        N_req_lo = (Z / eps_range) ** 2 * V_p10 / mu_true ** 2
        N_req_hi = (Z / eps_range) ** 2 * V_p90 / mu_true ** 2

        c = C_POD if name == "POD-weighted" else C_MLE
        ax.plot(eps_range * 100, N_req, "-", color=c, label=name)
        ax.fill_between(eps_range * 100, N_req_lo, N_req_hi,
                        alpha=0.18, color=c, linewidth=0)
        ax.plot(eps_range * 100, N_req_lo, "--", color=c, lw=0.7, alpha=0.5)
        ax.plot(eps_range * 100, N_req_hi, "--", color=c, lw=0.7, alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Target precision $\\varepsilon$ (%)")
    ax.set_ylabel("Independent monitoring campaigns $N$")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.2, ls=":", lw=0.5, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    reduction = (1 - var_est["MLE"] / var_est["POD-weighted"]) * 100

    examples = [(3, 0.03), (5, 0.05), (10, 0.10)]
    ex_data = []
    for pct, eps in examples:
        n_pod = (Z / eps) ** 2 * var_est["POD-weighted"] / mu_true ** 2
        n_mle = (Z / eps) ** 2 * var_est["MLE"] / mu_true ** 2
        n_pod_lo = (Z / eps) ** 2 * var_lo["POD-weighted"] / mu_true ** 2
        n_pod_hi = (Z / eps) ** 2 * var_hi["POD-weighted"] / mu_true ** 2
        n_mle_lo = (Z / eps) ** 2 * var_lo["MLE"] / mu_true ** 2
        n_mle_hi = (Z / eps) ** 2 * var_hi["MLE"] / mu_true ** 2
        ex_data.append((pct, n_mle, n_mle_lo, n_mle_hi, n_pod, n_pod_lo, n_pod_hi))

    for pct, n_mle, n_mle_lo, n_mle_hi, n_pod, n_pod_lo, n_pod_hi in ex_data:
        ax.axvline(pct, color="#BBBBBB", ls="--", lw=0.7, alpha=0.5)
        ax.plot(pct, n_pod, "s", color=C_POD, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6)
        ax.plot(pct, n_mle, "o", color=C_MLE, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6)

    box_specs = [
        (ex_data[0], (0.22, 0.82), "center"),
        (ex_data[1], (0.52, 0.72), "center"),
        (ex_data[2], (0.82, 0.62), "center"),
    ]

    for (pct, n_mle, n_mle_lo, n_mle_hi, n_pod, n_pod_lo, n_pod_hi), (bx, by), ha in box_specs:
        mid_y = np.sqrt(n_mle * n_pod)
        label = (
            f"$\\mathbf{{\\pm{pct}\\%}}$\n"
            f"MLE: {np.ceil(n_mle):.0f} ({np.ceil(n_mle_lo):.0f}\u2013{np.ceil(n_mle_hi):.0f})   "
            f"POD: {np.ceil(n_pod):.0f} ({np.ceil(n_pod_lo):.0f}\u2013{np.ceil(n_pod_hi):.0f})"
        )
        ax.annotate(
            label,
            xy=(pct, mid_y),
            fontsize=7, ha=ha, va="center",
            xytext=(bx, by), textcoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#AAAAAA", lw=0.7),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    ax.text(0.02, 0.04,
            r"$N = (1.96\,/(\varepsilon/100))^2"
            r" \times \mathrm{Var}(\hat{\mu})\,/\,\mu^2$"
            "\n"
            f"MLE variance is {reduction:.0f}% lower"
            f" = {reduction:.0f}% fewer campaigns needed",
            transform=ax.transAxes, fontsize=7,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F0F7F0",
                      ec=C_MLE, lw=1.0))

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n--- Precision summary ---")
    print(f"mu_true = {mu_true}")
    for name in ["POD-weighted", "MLE"]:
        V = var_est[name]
        print(f"{name:>15s}: Var = {V:.2f}")
        for eps_val in [0.03, 0.05, 0.10]:
            N = (Z / eps_val) ** 2 * V / mu_true ** 2
            print(f"  \u00b1{eps_val*100:.0f}%: N = {np.ceil(N):.0f}")

    print(f"\nVariance reduction: {reduction:.1f}%")
    print(f"Sample size reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
