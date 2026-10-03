"""
Baseline comparison of three estimators: Naive, POD-weighted, MLE (Loop).

Hybrid observation model:
  - Snapshot: per-time-step Bernoulli(p_snap).
  - Continuous: at each unmonitored step, Bernoulli(p_cont) starts
    a monitor that runs for T_cont contiguous steps.

Every parameter defaults to config.py but can be overridden via CLI.

Outputs (into figures/ and results/):
  figure_pod_curves.pdf          data_pod_curves.csv
  figure_baseline_violin.pdf     data_replications.csv
  figure_baseline_params.pdf     data_summary.csv
  figure_size_distribution.pdf   data_parameters.csv
  figure_realizations.pdf        data_realization_repN.csv

Usage:
    python -m experiments.run_baseline
    python -m experiments.run_baseline --n-reps 50
    python -m experiments.run_baseline --T 500 --p-snap 0.80
    python -m experiments.run_baseline --pi-on 0.3 --tau-emit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import pandas as pd

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "methane-mpl-cache"))

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
    naive_baseline,
    pod_weighted_baseline,
    mle_simple,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def _run_one_task(task):
    return run_one(*task)


def run_one(tech_specs, rep: int, P):
    """Single MC replication.  P is the parameter namespace."""
    mask_rng = np.random.default_rng(P.seed + rep)
    emit_rng = np.random.default_rng(P.seed + rep + 100_000)

    snap_mask, cont_mask = generate_masks(
        P.T, P.p_snap, P.p_cont, P.T_cont, mask_rng,
    )

    obs, states, sizes, _eids = simulate_series(
        T=P.T, p_on=P.p_on, p_off=P.p_off,
        size_mu=P.size_mu, size_sigma=P.size_sigma,
        tech_specs=tech_specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=emit_rng,
    )

    n_det = obs.n_total_detections()
    true_mean = float(np.mean(sizes))

    naive = naive_baseline(obs)
    pod = pod_weighted_baseline(obs, tech_specs)

    n_snap_obs = int(np.sum(snap_mask))
    n_cont_obs = int(np.sum(cont_mask))
    n_both_obs = int(np.sum(snap_mask & cont_mask))

    nudge = getattr(P, "p_off_nudge", 1.0)

    ms_res = mle_simple(obs, tech_specs, transition_bounds=cfg.TRANSITION_BOUNDS,
                        p_on_grid=P.p_on_grid, p_off_grid=P.p_off_grid,
                        p_off_nudge=nudge)
    ms_mean = ms_res["mean"]
    ms_p_on = ms_res["p_on"]
    ms_p_off = ms_res["p_off"]
    ms_mu_emit = ms_res["mu_emit"]

    res = loop_empirical(obs, tech_specs, P.T, **cfg.MLE_OPTIONS, max_gap=P.max_gap,
                         decision_threshold=P.decision_threshold,
                         p_on_grid=P.p_on_grid, p_off_grid=P.p_off_grid,
                         p_off_nudge=nudge)

    p_on_hat, p_off_hat = res["p_on"], res["p_off"]
    mu_hat = res["mean"]
    pi_hat = p_on_hat / (p_on_hat + p_off_hat)
    mu_emit_hat = mu_hat / pi_hat if pi_hat > 1e-10 else np.nan

    return dict(n_detections=n_det, converged=res['converged'],
                n_iterations=res['n_iterations'], optimizer_success=res['optimizer_success'],
                has_detections=res['has_detections'], boundary=res['boundary'],
                ms_optimizer_success=ms_res['optimizer_success'],
                true_mean=true_mean, naive=naive, pod=pod,
                mle_simple=ms_mean,
                ms_p_on=ms_p_on, ms_p_off=ms_p_off,
                ms_mu_emit=ms_mu_emit,
                mle=mu_hat, p_on=p_on_hat, p_off=p_off_hat,
                mu_emit=mu_emit_hat,
                n_snap_obs=n_snap_obs, n_cont_obs=n_cont_obs,
                n_both_obs=n_both_obs)


# ══════════════════════════════════════════════════════════════════
# Figure 1: Violin + Bias + Variance
# ══════════════════════════════════════════════════════════════════

def figure_pod_curves(tech_specs, out: Path,
                      snap_threshold=None, snap_slope=None,
                      cont_threshold=None, cont_slope=None):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    st = snap_threshold or cfg.AERIAL_THRESHOLD
    ss = snap_slope or cfg.AERIAL_SLOPE
    ct = cont_threshold or cfg.CONT_THRESHOLD
    cs = cont_slope or cfg.CONT_SLOPE

    e_grid = np.logspace(-1, 3, 500)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    pod_snap = tech_specs[0].pod.probability(e_grid)
    pod_cont = tech_specs[1].pod.probability(e_grid)

    ax.plot(e_grid, pod_snap, color="#2C3E50", lw=2.2,
            label=f"Snapshots ($\\theta$={st:.1f}, k={ss})")
    ax.plot(e_grid, pod_cont, color="#E74C3C", lw=2.2, ls="--",
            label=f"Continuous ($\\theta$={ct:.2f}, k={cs})")

    ax.axhline(0.9, color="gray", ls=":", lw=1, alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Emission size [kg/h]")
    ax.set_ylabel("Probability of Detection")
    ax.set_title("Detection Curves (POD)", fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0.1, 1000)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray", fontsize=9)
    ax.grid(alpha=0.2, ls=":", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Saved: {out.with_suffix('.png')}")


def figure_violin(naive_arr, pod_arr, mle_simple_arr,
                   mle_arr, mu_true, out: Path, outer_stats=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    c_naive = "#D98880"
    c_pod   = "#C39BD3"
    c_ms    = "#85C1E9"
    c_mle   = "#2ECC71"

    method_data = [
        ("Naive", naive_arr, c_naive),
        ("POD-weighted", pod_arr, c_pod),
        ("MLE-ungrouped", mle_simple_arr, c_ms),
        ("MLE", mle_arr, c_mle),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=200)

    parts = ax1.violinplot(
        [d for _, d, _ in method_data],
        positions=range(len(method_data)),
        showmeans=False, showmedians=False, showextrema=False,
    )
    for i, (body, (_, _, c)) in enumerate(zip(parts["bodies"], method_data)):
        body.set_facecolor(c)
        body.set_edgecolor("gray")
        body.set_alpha(0.7)

    for i, (_, d, _) in enumerate(method_data):
        q1, q3 = np.percentile(d, [25, 75])
        ax1.vlines(i, q1, q3, color="k", linewidth=4, zorder=5)
        ax1.scatter(i, np.mean(d), color="white", s=20, zorder=6, edgecolor="k", linewidth=0.8)

    ax1.axhline(mu_true, color="#7D3C98", ls="--", lw=1.5,
                label=f"True: {mu_true:.2f}", zorder=8)
    ax1.set_xticks(range(len(method_data)))
    ax1.set_xticklabels([n for n, _, _ in method_data])
    ax1.set_ylabel("Estimated Mean (kg/h)")
    ax1.set_title("(a) Violin plots")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.2, axis="y", ls=":", lw=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for name, d, c in method_data:
        bias_pt = np.mean(d) - mu_true
        var_pt = np.var(d, ddof=0)

        if outer_stats and name in outer_stats:
            st = outer_stats[name]
            biases = st["bias_arr"]
            vars_arr = st["var_arr"]
            n_o = len(biases)
            z = float(student_t.ppf(.975, n_o-1))
            bias_mean = np.mean(biases)
            bias_se = np.std(biases, ddof=1) / np.sqrt(n_o)
            var_mean = np.mean(vars_arr)
            var_se = np.std(vars_arr, ddof=1) / np.sqrt(n_o)
            ax2.errorbar(bias_mean, var_mean,
                         xerr=z * bias_se, yerr=z * var_se,
                         fmt="o", color=c, markersize=6, markeredgecolor="k",
                         markeredgewidth=0.8, ecolor=c, elinewidth=1.5,
                         capsize=3, capthick=1.2, zorder=5, label=name)
        else:
            ax2.scatter(bias_pt, var_pt, color=c, s=50, edgecolor="k",
                        linewidth=1.2, zorder=5, label=name)

    ax2.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("Bias (kg/h)")
    ax2.set_ylabel(r"Variance (kg/h)$^2$")
    ax2.set_title("(b) Bias vs Variance")
    ax2.legend(loc="best", framealpha=0.9, edgecolor="gray")
    ax2.grid(alpha=0.2, ls=":", lw=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Saved: {out.with_suffix('.png')}")


def figure_violin_v2(naive_arr, pod_arr, mle_simple_arr,
                     mle_arr, mu_true, out: Path, outer_stats=None):
    """Enhanced Figure 2: violins with annotations + bias-var scatter."""
    from scipy import stats as sp_stats

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
    })

    c_naive = "#D98880"
    c_pod   = "#C39BD3"
    c_ms    = "#85C1E9"
    c_mle   = "#2ECC71"

    method_data = [
        ("Naive",         naive_arr,      c_naive),
        ("POD-weighted",  pod_arr,        c_pod),
        ("MLE-ungrouped", mle_simple_arr, c_ms),
        ("MLE",           mle_arr,        c_mle),
    ]

    # ── Precompute bias / variance with CIs ──
    # (populated after batch summaries are built below; placeholders here)
    bias_means, bias_cis = [], []
    var_means, var_cis = [], []

    # ── Statistical helpers ──
    P_THRESH = 0.05

    clean_arrs = {n: d[np.isfinite(d)] for n, d, _ in method_data}
    method_names_list = [n for n, _, _ in method_data]
    n_methods = len(method_data)

    all_pairs = [(i, j)
                 for i in range(n_methods) for j in range(i + 1, n_methods)]

    # ── Build per-outer-batch summaries (25 replicates) ──
    n_outer = 25
    n_inner = None
    for name, d_raw, _ in method_data:
        d = d_raw[np.isfinite(d_raw)]
        if n_inner is None:
            n_inner = len(d_raw) // n_outer
        break

    batch_abs_bias = {}
    batch_var = {}
    for name, d_raw, _ in method_data:
        ab_list, var_list = [], []
        for o in range(n_outer):
            chunk = d_raw[o * n_inner:(o + 1) * n_inner]
            valid = chunk[np.isfinite(chunk)]
            if len(valid) > 0:
                ab_list.append(abs(np.mean(valid) - mu_true))
                var_list.append(np.var(valid, ddof=0))
            else:
                ab_list.append(np.nan)
                var_list.append(np.nan)
        batch_abs_bias[name] = np.array(ab_list)
        batch_var[name] = np.array(var_list)

    # Populate bias_means / bias_cis / var_means / var_cis from batches
    for name, _, _ in method_data:
        ab = batch_abs_bias[name]
        ab_fin = ab[np.isfinite(ab)]
        n_b = len(ab_fin)
        bias_means.append(float(np.mean(ab_fin)))
        bias_cis.append(float(student_t.ppf(.975,n_b-1)) * float(np.std(ab_fin, ddof=1)) / np.sqrt(n_b)
                        if n_b > 1 else 0)
        vb = batch_var[name]
        vb_fin = vb[np.isfinite(vb)]
        n_v = len(vb_fin)
        var_means.append(float(np.mean(vb_fin)))
        var_cis.append(float(student_t.ppf(.975,n_v-1)) * float(np.std(vb_fin, ddof=1)) / np.sqrt(n_v)
                       if n_v > 1 else 0)

    # ── Test 1: One-sample t-test for unbiasedness (per method) ──
    unbiased_results = {}
    print(f"\n  ── Test 1: Unbiasedness (one-sample t-test on {n_outer} "
          "outer batches, H\u2080: E[\u03bc\u0302]=\u03bc_true) ──")
    for name, _, _ in method_data:
        if outer_stats and name in outer_stats:
            b_arr = outer_stats[name]["bias_arr"]
            b_arr = b_arr[np.isfinite(b_arr)]
        else:
            b_arr = batch_abs_bias[name]
            b_arr = b_arr[np.isfinite(b_arr)]
        t_stat, p_val = sp_stats.ttest_1samp(b_arr, 0.0)
        unbiased_results[name] = (t_stat, p_val)
        p_s = "p<0.001" if p_val < 0.001 else f"p={p_val:.4f}"
        sig = "sig." if p_val < P_THRESH else "n.s."
        print(f"  {name:<20s}  t={t_stat:+8.2f}  {p_s:<12s}  {sig}"
              f"  (n={len(b_arr)} batches)")
    print()

    # ── Test 2: Paired t-test on batch |bias| (all 6 pairs) ──
    bias_results = {}
    print(f"  ── Test 2: |Bias| difference (paired t-test on {n_outer} "
          "outer batches, H\u2080: |bias|_A = |bias|_B) ──")
    for (i, j) in all_pairs:
        n1, n2 = method_names_list[i], method_names_list[j]
        a1 = batch_abs_bias[n1]
        a2 = batch_abs_bias[n2]
        mask = np.isfinite(a1) & np.isfinite(a2)
        t_stat, p_val = sp_stats.ttest_rel(a1[mask], a2[mask])
        bias_results[(i, j)] = (t_stat, p_val)
        p_s = "p<0.001" if p_val < 0.001 else f"p={p_val:.4f}"
        sig = "sig." if p_val < P_THRESH else "n.s."
        label = f"{n1} vs {n2}"
        print(f"  {label:<30s}  t={t_stat:+8.2f}  {p_s:<12s}  {sig}")
    print()

    # ── Test 3: Paired t-test on batch variance (all 6 pairs) ──
    var_results = {}
    print(f"  ── Test 3: Variance difference (paired t-test on {n_outer} "
          "outer batches, H\u2080: Var_A = Var_B) ──")
    for (i, j) in all_pairs:
        n1, n2 = method_names_list[i], method_names_list[j]
        v1 = batch_var[n1]
        v2 = batch_var[n2]
        mask = np.isfinite(v1) & np.isfinite(v2)
        t_stat, p_val = sp_stats.ttest_rel(v1[mask], v2[mask])
        var_results[(i, j)] = (t_stat, p_val)
        p_s = "p<0.001" if p_val < 0.001 else f"p={p_val:.4f}"
        sig = "sig." if p_val < P_THRESH else "n.s."
        label = f"{n1} vs {n2}"
        print(f"  {label:<30s}  t={t_stat:+8.2f}  {p_s:<12s}  {sig}")
    print()

    # ── Signed bias (kg/h) from 25 batches ──
    batch_signed_bias = {}
    signed_bias_means, signed_bias_cis = [], []
    for name, d_raw, _ in method_data:
        sb_list = []
        for o in range(n_outer):
            chunk = d_raw[o * n_inner:(o + 1) * n_inner]
            valid = chunk[np.isfinite(chunk)]
            if len(valid) > 0:
                sb_list.append(np.mean(valid) - mu_true)
            else:
                sb_list.append(np.nan)
        arr = np.array(sb_list)
        batch_signed_bias[name] = arr
        fin = arr[np.isfinite(arr)]
        n_sb = len(fin)
        signed_bias_means.append(float(np.mean(fin)))
        signed_bias_cis.append(float(student_t.ppf(.975,n_sb-1)) * float(np.std(fin, ddof=1))
                               / np.sqrt(n_sb) if n_sb > 1 else 0)

    # ── Shared drawing helpers ──
    from matplotlib.gridspec import GridSpec
    from matplotlib.font_manager import FontProperties
    _sym_font = FontProperties(family="DejaVu Sans", weight="bold")
    LBL_FS = 6.5

    positions = np.arange(n_methods)
    names = [n for n, _, _ in method_data]
    bar_colors = [c for _, _, c in method_data]
    sorted_pairs = sorted(all_pairs, key=lambda p: abs(p[1] - p[0]))

    def _draw_bracket(ax, x1, x2, y, h, sig_text, color="k", fontsize=7):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                lw=0.8, color=color)
        ax.text((x1 + x2) / 2, y + h * 1.1, sig_text,
                ha="center", va="bottom", fontsize=fontsize, color=color,
                fontproperties=_sym_font)

    def _style_ax(ax):
        ax.grid(alpha=0.2, axis="y", ls=":", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Panel drawers ──
    def draw_violin(ax, label="(a)"):
        clean_data = [d[np.isfinite(d)] for _, d, _ in method_data]
        parts = ax.violinplot(clean_data, positions=range(n_methods),
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for body, (_, _, c) in zip(parts["bodies"], method_data):
            body.set_facecolor(c); body.set_edgecolor("gray")
            body.set_alpha(0.7)
        label_info = []
        for i, (_, d_raw, _) in enumerate(method_data):
            d = d_raw[np.isfinite(d_raw)]
            q1, q3 = np.percentile(d, [25, 75])
            mn = np.mean(d)
            ax.vlines(i, q1, q3, color="k", linewidth=4, zorder=5)
            ax.scatter(i, mn, color="white", s=20, zorder=6,
                       edgecolor="k", linewidth=0.8)
            p97 = np.percentile(d, 97)
            label_info.append((i, p97, mn, q1, q3))
        all_q99 = [np.percentile(cd, 99) for cd in clean_data]
        y_top = max(all_q99) * 1.25
        y_range = y_top
        min_gap = y_range * 0.07
        placed_y = []
        for i, p97, mn, q1, q3 in label_info:
            y = p97 + y_range * 0.02
            for py in placed_y:
                if abs(y - py) < min_gap:
                    y = py + min_gap
            placed_y.append(y)
            ax.text(i, y,
                    f"{mn:.2f} [{q1:.1f}, {q3:.1f}]",
                    ha="center", va="bottom", fontsize=LBL_FS, color="k",
                    fontproperties=_sym_font,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))
        max_label_y = max(placed_y) + min_gap * 1.5
        ax.set_ylim(bottom=0, top=max(y_top, max_label_y))
        ax.axhline(mu_true, color="#7D3C98", ls="--", lw=1.5,
                   label=f"True $\\mu$={mu_true:.1f}", zorder=8)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Estimated $\\hat{\\mu}$ (kg/h)")
        ax.set_title(f"{label} Distribution of estimates", fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=6)
        _style_ax(ax)

    def draw_abs_bias(ax, label="(b)"):
        ax.bar(positions, bias_means, yerr=bias_cis, color=bar_colors,
               edgecolor="k", linewidth=0.6, width=0.6, capsize=3,
               error_kw=dict(lw=1, capthick=0.8, color="k"))
        bmax = max(bias_means)
        min_ly = bmax * 0.12
        for pos, val, ci in zip(positions, bias_means, bias_cis):
            ly = max(val + ci + bmax * 0.03, min_ly)
            ax.text(pos, ly, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=LBL_FS, fontproperties=_sym_font,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.85))
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("Mean |Batch Bias| (kg/h)")
        ax.set_title(f"{label} Mean |Batch Bias|", fontweight="bold")
        ax.set_ylim(0, None)
        _style_ax(ax)
        # ★ / ☆ brackets
        btop = max(v + c for v, c in zip(bias_means, bias_cis))
        btop = max(btop, bmax * 0.4)
        bstep = bmax * 0.18
        bh = bmax * 0.04
        for k, (i, j) in enumerate(sorted_pairs):
            p_b = bias_results[(i, j)][1]
            sym = "\u2605" if p_b < P_THRESH else "\u2606"
            _draw_bracket(ax, i, j, btop + bstep * (k + 0.6), bh, sym)
        ax.set_ylim(0, btop + bstep * (len(sorted_pairs) + 0.8)
                    + bmax * 0.1)

    def draw_signed_bias(ax, label="(c)"):
        ax.bar(positions, signed_bias_means, yerr=signed_bias_cis,
               color=bar_colors, edgecolor="k", linewidth=0.6, width=0.6,
               capsize=3,
               error_kw=dict(lw=1, capthick=0.8, color="k"))
        abs_sb_max = max(abs(v) for v in signed_bias_means)
        for pos, (name, _, _), val, ci in zip(positions, method_data,
                                               signed_bias_means,
                                               signed_bias_cis):
            p_u = unbiased_results[name][1]
            marker = "\u25CF" if p_u < P_THRESH else "\u25CB"
            offset = abs_sb_max * 0.08
            if val >= 0:
                ly = max(val + ci, 0) + offset
                va = "bottom"
            else:
                ly = min(val - ci, 0) - offset
                va = "top"
            ax.text(pos, ly, f"{val:+.2f} {marker}",
                    ha="center", va=va, fontsize=LBL_FS, fontweight="bold",
                    fontproperties=_sym_font,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.85))
        ax.axhline(0, color="gray", ls="-", lw=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("Bias (kg/h)")
        ax.set_title(f"{label} Bias", fontweight="bold")
        margin = abs_sb_max * 0.5
        ax.set_ylim(min(signed_bias_means) - margin,
                    max(signed_bias_means) + margin)
        _style_ax(ax)

    def draw_variance(ax, label="(d)"):
        ax.bar(positions, var_means, yerr=var_cis, color=bar_colors,
               edgecolor="k", linewidth=0.6, width=0.6, capsize=3,
               error_kw=dict(lw=1, capthick=0.8, color="k"))
        vm = max(var_means)
        for pos, val, ci in zip(positions, var_means, var_cis):
            ax.text(pos, val + ci + vm * 0.02, f"{val:.1f}",
                    ha="center", va="bottom",
                    fontsize=LBL_FS, fontproperties=_sym_font)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel(r"Variance (kg/h)$^2$")
        ax.set_title(f"{label} Variance", fontweight="bold")
        _style_ax(ax)
        # ◆ / ◇ brackets
        ymv = max(v + c + vm * 0.02 for v, c in zip(var_means, var_cis))
        bracket_gap = vm * 0.08
        bstep = vm * 0.08
        bh = vm * 0.015
        for k, (i, j) in enumerate(sorted_pairs):
            p_v = var_results[(i, j)][1]
            sym = "\u25C6" if p_v < P_THRESH else "\u25C7"
            _draw_bracket(ax, i, j, ymv + bracket_gap + bstep * k, bh, sym)
        ax.set_ylim(0, ymv + bracket_gap + bstep * (len(sorted_pairs) + 0.5)
                    + vm * 0.05)

    # ════════════════════════════════════════════════════════════════
    # Version A: side-by-side (1×4)
    # ════════════════════════════════════════════════════════════════
    fig_a = plt.figure(figsize=(13, 4.5), dpi=200)
    gs_a = GridSpec(1, 4, figure=fig_a,
                    width_ratios=[1.2, 1, 1, 1], wspace=0.38)
    draw_violin(fig_a.add_subplot(gs_a[0]), "(a)")
    draw_signed_bias(fig_a.add_subplot(gs_a[1]), "(b)")
    draw_abs_bias(fig_a.add_subplot(gs_a[2]), "(c)")
    draw_variance(fig_a.add_subplot(gs_a[3]), "(d)")
    fig_a.tight_layout()
    out_a = out.parent / (out.stem + "_wide.pdf")
    fig_a.savefig(out_a, format="pdf", bbox_inches="tight", dpi=300)
    fig_a.savefig(out_a.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig_a)
    print(f"  Saved: {out_a}")
    print(f"  Saved: {out_a.with_suffix('.png')}")

    # ════════════════════════════════════════════════════════════════
    # Version B: stacked bias (2×2 grid)
    # ════════════════════════════════════════════════════════════════
    fig_b = plt.figure(figsize=(10, 8), dpi=200)
    gs_b = GridSpec(2, 2, figure=fig_b,
                    width_ratios=[1.2, 1], hspace=0.35, wspace=0.35)
    draw_violin(fig_b.add_subplot(gs_b[0, 0]), "(a)")
    draw_signed_bias(fig_b.add_subplot(gs_b[0, 1]), "(b)")
    draw_abs_bias(fig_b.add_subplot(gs_b[1, 0]), "(c)")
    draw_variance(fig_b.add_subplot(gs_b[1, 1]), "(d)")
    fig_b.tight_layout()
    out_b = out.parent / (out.stem + "_stacked.pdf")
    fig_b.savefig(out_b, format="pdf", bbox_inches="tight", dpi=300)
    fig_b.savefig(out_b.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig_b)
    print(f"  Saved: {out_b}")
    print(f"  Saved: {out_b.with_suffix('.png')}")


# ══════════════════════════════════════════════════════════════════
# Figure 2: Parameter histograms  (p_on, p_off, mu_emit, mu)
# ══════════════════════════════════════════════════════════════════

def figure_params(p_on_v, p_off_v, me_v, mle_v, out: Path,
                   true_p_on=None, true_p_off=None,
                   true_mu_emit=None, true_mu=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 5,
    })

    C_PARAMS = ["#B0B0B0", "#909090", "#C8C8C8", "#2ECC71"]

    panels = [
        (r"$\hat{p}_{\mathrm{on}}$",  p_on_v,  true_p_on  or cfg.P_ON, C_PARAMS[0], ""),
        (r"$\hat{p}_{\mathrm{off}}$", p_off_v, true_p_off or cfg.P_OFF, C_PARAMS[1], ""),
        (r"$\hat{\mu}_{\mathrm{emit}}$", me_v, true_mu_emit or cfg.MU_EMIT, C_PARAMS[2], " (kg/h)"),
        (r"$\hat{\mu}$",              mle_v,  true_mu or cfg.TRUE_PARAMS["mu"], C_PARAMS[3], " (kg/h)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2), dpi=200)

    bins_per_panel = [None, None, 30, 30]
    for idx, (ax, (label, arr, true_val, color, unit)) in enumerate(zip(axes, panels)):
        arr_clean = arr[np.isfinite(arr)]
        est_mean = float(np.mean(arr_clean))
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
        ax.axvline(true_val, color="k", lw=1.5, ls="-", zorder=5,
                   label=f"True = {true_val:.4g}")
        ax.axvline(est_mean, color="k", lw=1.2, ls="--", zorder=5,
                   label=f"Mean = {est_mean:.4g}")
        if idx < 2:
            ax.set_xscale("log")
        ax.set_xlabel(label + unit)
        ax.legend(framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Density")
        ax.grid(alpha=0.2, ls=":", lw=0.5)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Saved: {out.with_suffix('.png')}")


# ══════════════════════════════════════════════════════════════════
# Figure 3: Example realizations
# ══════════════════════════════════════════════════════════════════

def _pick_interesting_reps(tech_specs, P, n_pick=3, n_scan=5000):
    """Select reps that showcase the model well.

    Prioritises reps with continuous monitoring windows (the typical case
    with the renewal model) and includes at most one snap-only panel for
    contrast, mirroring the true population distribution.
    """
    snap_only_cands = []
    cont_cands = []

    for rep in range(n_scan):
        rng_m = np.random.default_rng(P.seed + rep)
        rng_e = np.random.default_rng(P.seed + rep + 100_000)
        snap_mask, cont_mask = generate_masks(
            P.T, P.p_snap, P.p_cont, P.T_cont, rng_m)
        obs, states, sizes, eids = simulate_series(
            T=P.T, p_on=P.p_on, p_off=P.p_off,
            size_mu=P.size_mu, size_sigma=P.size_sigma,
            tech_specs=tech_specs,
            snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_e)

        n_events = int(eids.max() + 1) if eids.max() >= 0 else 0
        n_det = obs.n_total_detections()
        on_times = np.flatnonzero(states == 1)
        obs_on = np.sum(obs.any_observed[on_times]) if len(on_times) else 0
        det_on = np.sum(obs.any_detected[on_times]) if len(on_times) else 0
        missed = obs_on - det_on

        if n_events < 2 or n_det < 2 or missed < 1:
            continue

        has_snap_det = np.any(obs.snap_mask & obs.snap_detected)
        has_cont_det = np.any(obs.cont_mask & obs.cont_detected)
        has_cont_deployed = np.any(cont_mask)
        n_cont_windows = 0
        if has_cont_deployed:
            active = np.flatnonzero(cont_mask)
            n_cont_windows = len(np.split(
                active, np.where(np.diff(active) != 1)[0] + 1))
        variety = (n_events + int(has_snap_det) + int(has_cont_det)
                   + min(int(missed), 3) + n_cont_windows)

        if has_cont_deployed:
            cont_cands.append((variety, rep))
        else:
            snap_only_cands.append((variety, rep))

    cont_cands.sort(key=lambda x: -x[0])
    snap_only_cands.sort(key=lambda x: -x[0])

    chosen = []
    # Fill mostly with continuous-monitoring reps (representative)
    for c in cont_cands:
        if len(chosen) >= n_pick - 1:
            break
        chosen.append(c[1])
    # Add one snap-only rep for contrast (if available)
    if snap_only_cands:
        chosen.append(snap_only_cands[0][1])
    # Top up with remaining cont reps if needed
    idx = len(chosen) - (1 if snap_only_cands else 0)
    for c in cont_cands[idx:]:
        if len(chosen) >= n_pick:
            break
        chosen.append(c[1])
    if len(chosen) < n_pick:
        chosen += list(range(n_pick - len(chosen)))
    return chosen[:n_pick]


def figure_realizations(tech_specs, P, out: Path, reps=None):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    if reps is None:
        reps = _pick_interesting_reps(tech_specs, P)

    n = len(reps)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), dpi=200,
                              sharex=True)
    if n == 1:
        axes = [axes]

    c_on     = "#F9E79F"
    c_snap   = "#2C3E50"
    c_cont   = "#E74C3C"
    c_true   = "#7D3C98"

    for panel_idx, (ax, rep) in enumerate(zip(axes, reps)):
        rng_m = np.random.default_rng(P.seed + rep)
        rng_e = np.random.default_rng(P.seed + rep + 100_000)

        snap_mask, cont_mask = generate_masks(
            P.T, P.p_snap, P.p_cont, P.T_cont, rng_m)
        obs, states, sizes, eids = simulate_series(
            T=P.T, p_on=P.p_on, p_off=P.p_off,
            size_mu=P.size_mu, size_sigma=P.size_sigma,
            tech_specs=tech_specs,
            snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_e)

        t = np.arange(P.T)
        on_mask = states == 1
        for t_start in np.flatnonzero(on_mask & ~np.roll(on_mask, 1)):
            run = np.flatnonzero(~on_mask[t_start:])
            t_end = t_start + run[0] if len(run) else P.T
            ax.axvspan(t_start - 0.5, t_end - 0.5, alpha=0.25,
                       color=c_on, zorder=0)

        ax.step(t, sizes, where="mid", color=c_true, lw=1.2,
                alpha=0.5, zorder=1, label="True emission")

        for idx in range(P.T):
            for tech_id, mask, det, meas in [
                (0, obs.snap_mask, obs.snap_detected, obs.snap_measurements),
                (1, obs.cont_mask, obs.cont_detected, obs.cont_measurements),
            ]:
                if not mask[idx]:
                    continue
                c = c_snap if tech_id == 0 else c_cont
                lbl_det = "Snapshot det." if tech_id == 0 else "Continuous det."
                lbl_miss = "Snapshot (no det.)" if tech_id == 0 else "Continuous (no det.)"
                if det[idx]:
                    ax.scatter(idx, meas[idx],
                               marker="D" if tech_id == 0 else "o",
                               s=40 if tech_id == 0 else 30,
                               color=c, edgecolor="white", linewidth=0.6,
                               zorder=6, label=lbl_det)
                else:
                    ax.scatter(idx, 0.3, marker="x", s=25, color=c,
                               alpha=0.4, linewidth=1, zorder=4,
                               label=lbl_miss)

        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                unique_h.append(h)
                unique_l.append(l)
        ax.legend(unique_h, unique_l, loc="upper right",
                  fontsize=7, framealpha=0.9, edgecolor="gray", ncol=3)

        n_events = int(eids.max() + 1) if eids.max() >= 0 else 0
        n_det = obs.n_total_detections()
        ax.set_ylabel("Emission [kg/h]")
        ax.set_title(
            f"({chr(97 + panel_idx)})  Replication {rep}  —  "
            f"{n_events} events, {n_det} detections, "
            f"true mean = {np.mean(sizes):.1f} kg/h",
            fontweight="bold", fontsize=10)
        ax.set_ylim(bottom=-1)
        ax.grid(alpha=0.15, ls=":", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time period  $t$")
    fig.suptitle("Example Realisations  "
                 f"($T$={P.T}, $p_{{\\mathrm{{snap}}}}$={P.p_snap}, "
                 f"$p_{{\\mathrm{{cont}}}}$={P.p_cont}, "
                 f"$T_{{\\mathrm{{cont}}}}$={P.T_cont})",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Saved: {out.with_suffix('.png')}")
    return reps


# ══════════════════════════════════════════════════════════════════
# Figure 4: True emission-size distribution
# ══════════════════════════════════════════════════════════════════

def figure_size_distribution(P, out: Path):
    from scipy.stats import lognorm
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    e_grid = np.linspace(0.1, 250, 1000)
    s = P.size_sigma
    scale = np.exp(P.size_mu)
    pdf = lognorm.pdf(e_grid, s=s, scale=scale)
    cdf = lognorm.cdf(e_grid, s=s, scale=scale)
    e_mean = np.exp(P.size_mu + 0.5 * s**2)
    e_median = np.exp(P.size_mu)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=200)

    ax1.fill_between(e_grid, pdf, alpha=0.25, color="#2C3E50")
    ax1.plot(e_grid, pdf, color="#2C3E50", lw=2)
    ax1.axvline(e_mean, color="#7D3C98", ls="--", lw=1.5,
                label=f"Mean = {e_mean:.1f} kg/h")
    ax1.axvline(e_median, color="#E67E22", ls=":", lw=1.5,
                label=f"Median = {e_median:.1f} kg/h")
    ax1.set_xlabel("Emission size when ON  [kg/h]")
    ax1.set_ylabel("Probability density")
    ax1.set_title("(a)  Emission Size Distribution (Lognormal)",
                   fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor="gray")
    ax1.set_xlim(0, 250)
    ax1.grid(alpha=0.15, ls=":", lw=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.97, 0.95,
             f"$\\mu_{{\\log}}$ = {P.size_mu:.3f}\n"
             f"$\\sigma_{{\\log}}$ = {P.size_sigma}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                    fc="white", ec="gray", alpha=0.9))

    ax2.plot(e_grid, cdf, color="#2C3E50", lw=2, label="Size CDF")
    e_log = np.logspace(-1, np.log10(250), 500)
    from src.utils import LogisticPOD
    pod_snap = LogisticPOD(threshold=P.snap_threshold,
                            slope=P.snap_slope).probability(e_log)
    pod_cont = LogisticPOD(threshold=P.cont_threshold,
                            slope=P.cont_slope).probability(e_log)
    ax2.plot(e_log, pod_snap, color="#34495E", lw=1.8, ls="--",
             label=f"POD Snapshots ($\\theta$={P.snap_threshold:.1f})")
    ax2.plot(e_log, pod_cont, color="#E74C3C", lw=1.8, ls=":",
             label=f"POD Continuous ($\\theta$={P.cont_threshold:.1f})")
    ax2.axvline(e_mean, color="#7D3C98", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("Emission size  [kg/h]")
    ax2.set_ylabel("Probability")
    ax2.set_title("(b)  Size CDF and Detection Probability",
                   fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor="gray", loc="center right")
    ax2.set_xlim(0, 250)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(alpha=0.15, ls=":", lw=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Saved: {out.with_suffix('.png')}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def _save_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Baseline comparison — all parameters overridable",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Simulation")
    g.add_argument("--n-reps",      type=int,   default=1000)
    g.add_argument("--n-outer",     type=int,   default=1)
    g.add_argument("--T",           type=int,   default=cfg.T)
    g.add_argument("--pi-on",       type=float, default=cfg.PI_ON)
    g.add_argument("--tau-emit",    type=float, default=cfg.TAU_EMIT)
    g.add_argument("--mu-emit",     type=float, default=cfg.MU_EMIT)
    g.add_argument("--size-mu",     type=float, default=cfg.SIZE_MU)
    g.add_argument("--size-sigma",  type=float, default=cfg.SIZE_SIGMA)

    g = p.add_argument_group("Observation (hybrid model)")
    g.add_argument("--p-snap",         type=float, default=cfg.P_SNAP,
                   help="Per-time-step prob. of snapshot observation")
    g.add_argument("--p-cont",         type=float, default=cfg.P_CONT,
                   help="Per-step prob. of starting continuous window")
    g.add_argument("--T-cont",         type=int,   default=cfg.T_CONT,
                   help="Duration of continuous monitor (time steps)")

    g = p.add_argument_group("Snapshot sensor")
    g.add_argument("--snap-threshold", type=float, default=cfg.AERIAL_THRESHOLD)
    g.add_argument("--snap-slope",     type=float, default=cfg.AERIAL_SLOPE)
    g.add_argument("--snap-sigma",     type=float, default=cfg.AERIAL_SENSOR_SIGMA)

    g = p.add_argument_group("Continuous sensor")
    g.add_argument("--cont-threshold", type=float, default=cfg.CONT_THRESHOLD)
    g.add_argument("--cont-slope",     type=float, default=cfg.CONT_SLOPE)
    g.add_argument("--cont-sigma",     type=float, default=cfg.CONT_SENSOR_SIGMA)
    g.add_argument("--cont-fp-rate",   type=float, default=cfg.CONT_FP_RATE)
    g.add_argument("--cont-fp-scale",  type=float, default=cfg.CONT_FP_SCALE)

    g = p.add_argument_group("Estimator")
    g.add_argument("--max-gap",            type=int,   default=cfg.MAX_GAP)
    g.add_argument("--decision-threshold", type=float, default=cfg.DECISION_THRESHOLD)
    g.add_argument("--p-grid-res",    type=int,   default=cfg.P_GRID_RES)
    g.add_argument("--p-geom-factor", type=float, default=cfg.P_GEOM_FACTOR)
    g.add_argument("--nudge",         type=float, default=cfg.NUDGE,
                   help="Compatibility option; only 1.0 is accepted")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", type=str, default=None)
    g.add_argument("--seed",       type=int, default=cfg.BASE_SEED)

    return p


def _derived_params(args):
    args.p_off = 1.0 / args.tau_emit
    args.p_on  = args.pi_on * args.p_off / (1.0 - args.pi_on)
    args.mu_true = args.pi_on * args.mu_emit
    args.p_on_grid = None
    args.p_off_grid = None
    if args.nudge != 1.0:
        raise ValueError("The revised estimator requires nudge=1.0.")
    return args


def main():
    parser = _build_parser()
    args = parser.parse_args()
    P = _derived_params(args)

    if P.output_dir:
        fig_dir = Path(P.output_dir) / "figures"
        res_dir = Path(P.output_dir) / "results"
    else:
        fig_dir = PROJECT_ROOT / "figures"
        res_dir = PROJECT_ROOT / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    tech_specs = build_baseline_specs(
        aerial_threshold=P.snap_threshold,
        aerial_slope=P.snap_slope,
        aerial_sensor_sigma=P.snap_sigma,
        cont_threshold=P.cont_threshold,
        cont_slope=P.cont_slope,
        cont_sensor_sigma=P.cont_sigma,
        cont_fp_rate=P.cont_fp_rate,
        cont_fp_scale=P.cont_fp_scale,
    )

    P.p_off_nudge = P.nudge

    print("=" * 62)
    print("  BASELINE COMPARISON (Hybrid Observation Model)")
    print("=" * 62)
    n_total = P.n_reps * P.n_outer
    print(f"  T={P.T}, p_snap={P.p_snap}, p_cont={P.p_cont}, "
          f"T_cont={P.T_cont}")
    print(f"  p_off_nudge={P.p_off_nudge:.4f}")
    print(f"  n_reps={P.n_reps}, n_outer={P.n_outer}  (total={n_total})")
    print(f"  pi_on={P.pi_on}, tau_emit={P.tau_emit}")
    print(f"  => p_on={P.p_on:.6f}, p_off={P.p_off:.6f}")
    print(f"  mu_emit={P.mu_emit}, size_mu={P.size_mu}, size_sigma={P.size_sigma}")
    print(f"  True mu = pi_on * mu_emit = {P.mu_true:.4f}")
    print(f"  Snap: threshold={P.snap_threshold}, slope={P.snap_slope}, "
          f"sigma={P.snap_sigma}")
    print(f"  Cont: threshold={P.cont_threshold}, slope={P.cont_slope}, "
          f"sigma={P.cont_sigma}, fp={P.cont_fp_rate}")
    print()

    mu_true = P.mu_true
    all_results = []
    outer_bias = {"Naive": [], "POD-weighted": [], "MLE-ungrouped": [],
                  "MLE": []}
    outer_var  = {"Naive": [], "POD-weighted": [], "MLE-ungrouped": [],
                  "MLE": []}
    outer_mle_params = {"p_on": [], "p_off": [], "mu_emit": []}
    outer_ms_params  = {"p_on": [], "p_off": [], "mu_emit": []}

    t0_global = time.time()
    for o in range(P.n_outer):
        batch_P = argparse.Namespace(**vars(P))
        batch_P.seed = P.seed + o * P.n_reps

        if P.n_outer > 1:
            print(f"\n{'=' * 62}")
            print(f"  OUTER BATCH {o+1}/{P.n_outer}")
            print(f"{'=' * 62}")
        else:
            print(f"\n{'=' * 62}")
            print(f"  MAIN EXPERIMENT  ({P.n_reps} reps)")
            print(f"{'=' * 62}")

        from experiments.parallel import ordered_map
        batch_results = []
        t0 = time.time()
        batch_tasks = [(tech_specs,rep,batch_P) for rep in range(P.n_reps)]
        batch_iterator = ordered_map(_run_one_task,batch_tasks)
        for rep in range(P.n_reps):
            if (rep + 1) % max(P.n_reps // 10, 1) == 0:
                el = time.time() - t0
                eta = el / (rep + 1) * (P.n_reps - rep - 1)
                print(f"  rep {rep+1:>4d}/{P.n_reps}  "
                      f"elapsed={el:.1f}s  eta={eta:.1f}s", flush=True)
            batch_results.append(next(batch_iterator))

        all_results.extend(batch_results)

        naive_b = np.array([r["naive"] for r in batch_results])
        pod_b = np.array([r["pod"] for r in batch_results])
        ms_b = np.array([r["mle_simple"] for r in batch_results])
        ms_v_b = ms_b[np.isfinite(ms_b)]
        mle_b = np.array([r["mle"] for r in batch_results])
        mle_v_b = mle_b[np.isfinite(mle_b)]

        for key, arr in [("Naive", naive_b), ("POD-weighted", pod_b),
                         ("MLE-ungrouped", ms_v_b),
                         ("MLE", mle_v_b)]:
            outer_bias[key].append(float(np.mean(arr) - mu_true))
            outer_var[key].append(float(np.var(arr, ddof=0)))

        for pname, rkey, true_val in [("p_on", "p_on", P.p_on),
                                       ("p_off", "p_off", P.p_off),
                                       ("mu_emit", "mu_emit", P.mu_emit)]:
            pv = np.array([r[rkey] for r in batch_results])
            pv = pv[np.isfinite(pv)]
            if len(pv) > 0:
                outer_mle_params[pname].append(float(np.mean(pv) - true_val))
        for pname, rkey, true_val in [("p_on", "ms_p_on", P.p_on),
                                       ("p_off", "ms_p_off", P.p_off),
                                       ("mu_emit", "ms_mu_emit", P.mu_emit)]:
            pv = np.array([r[rkey] for r in batch_results])
            pv = pv[np.isfinite(pv)]
            if len(pv) > 0:
                outer_ms_params[pname].append(float(np.mean(pv) - true_val))

    elapsed = time.time() - t0_global

    outer_stats = {}
    for key in outer_bias:
        outer_stats[key] = dict(
            bias_arr=np.array(outer_bias[key]),
            var_arr=np.array(outer_var[key]),
        )

    naive_arr = np.array([r["naive"] for r in all_results])
    pod_arr = np.array([r["pod"] for r in all_results])
    ms_arr = np.array([r["mle_simple"] for r in all_results])
    ms_valid = ms_arr[np.isfinite(ms_arr)]
    mle_arr = np.array([r["mle"] for r in all_results])
    mle_valid = mle_arr[np.isfinite(mle_arr)]

    p_on_arr = np.array([r["p_on"] for r in all_results])
    p_off_arr = np.array([r["p_off"] for r in all_results])
    mu_emit_arr = np.array([r["mu_emit"] for r in all_results])
    ms_p_on_arr = np.array([r["ms_p_on"] for r in all_results])
    ms_p_off_arr = np.array([r["ms_p_off"] for r in all_results])
    ms_mu_emit_arr = np.array([r["ms_mu_emit"] for r in all_results])
    true_mean_arr = np.array([r["true_mean"] for r in all_results])
    p_on_v = p_on_arr[np.isfinite(p_on_arr)]
    p_off_v = p_off_arr[np.isfinite(p_off_arr)]
    me_v = mu_emit_arr[np.isfinite(mu_emit_arr)]
    ms_p_on_v = ms_p_on_arr[np.isfinite(ms_p_on_arr)]
    ms_p_off_v = ms_p_off_arr[np.isfinite(ms_p_off_arr)]
    ms_me_v = ms_mu_emit_arr[np.isfinite(ms_mu_emit_arr)]

    print(f"\n{'=' * 62}")
    print(f"  RESULTS  ({elapsed:.1f}s, {n_total} reps"
          f" = {P.n_outer} outer x {P.n_reps} inner)")
    print(f"{'=' * 62}")

    n_snap_arr = np.array([r["n_snap_obs"] for r in all_results])
    n_cont_arr = np.array([r["n_cont_obs"] for r in all_results])
    n_both_arr = np.array([r["n_both_obs"] for r in all_results])
    print(f"\n  Observation counts (per site, T={P.T}):")
    print(f"    Snap obs:  mean={np.mean(n_snap_arr):.1f}  "
          f"(expected {P.T*P.p_snap:.1f})")
    print(f"    Cont obs:  mean={np.mean(n_cont_arr):.1f}")
    print(f"    Both obs:  mean={np.mean(n_both_arr):.1f}")

    summary_rows = []
    for name, arr in [("Naive", naive_arr), ("POD-weighted", pod_arr),
                      ("MLE-ungrouped", ms_valid),
                      ("MLE", mle_valid)]:
        m = np.mean(arr)
        b = m - mu_true
        var = np.var(arr, ddof=0)
        rmse = np.sqrt(b**2 + var)
        ci_str = ""
        if P.n_outer > 1 and name in outer_stats:
            st = outer_stats[name]
            b_arr = st["bias_arr"]
            se = np.std(b_arr, ddof=1) / np.sqrt(len(b_arr))
            ci_str = f"  95%CI=+/-{1.96*se:.4f}"
        print(f"  {name:<16s}  mean={m:8.4f}  bias={b:+8.4f}{ci_str}"
              f"  var={var:10.4f}  rmse={rmse:8.4f}")
        summary_rows.append(dict(
            method=name, true_value=mu_true, n_valid=len(arr),
            mean=m, bias=b, variance=var, rmse=rmse,
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
        ))

    if P.n_outer > 1:
        print(f"\n  95% CI from {P.n_outer} outer batches:")
        for key in ["Naive", "POD-weighted", "MLE-ungrouped", "MLE"]:
            st = outer_stats[key]
            b_arr, v_arr = st["bias_arr"], st["var_arr"]
            n_o = len(b_arr)
            z = float(student_t.ppf(.975, n_o-1))
            b_m, b_se = np.mean(b_arr), np.std(b_arr, ddof=1) / np.sqrt(n_o)
            v_m, v_se = np.mean(v_arr), np.std(v_arr, ddof=1) / np.sqrt(n_o)
            print(f"    {key:<20s}  bias={b_m:+.3f} +/- {z*b_se:.3f}  "
                  f"var={v_m:.2f} +/- {z*v_se:.2f}")

    print(f"\n  MLE sub-parameters (n_valid={len(mle_valid)}/{n_total}):")
    for name, arr, true in [("p_on", p_on_v, P.p_on),
                             ("p_off", p_off_v, P.p_off),
                             ("mu_emit", me_v, P.mu_emit)]:
        m = np.mean(arr)
        b = m - true
        var = np.var(arr, ddof=0)
        ci_str = ""
        if P.n_outer > 1 and name in outer_mle_params and len(outer_mle_params[name]) > 1:
            ob = np.array(outer_mle_params[name])
            se = np.std(ob, ddof=1) / np.sqrt(len(ob))
            ci_str = f"  95%CI=+/-{1.96*se:.4f}"
        print(f"    {name:>10s}: mean={m:.4f}  true={true:.4f}  "
              f"bias={b:+.4f}{ci_str}  var={var:.6f}")
        summary_rows.append(dict(
            method=f"MLE_{name}", true_value=true, n_valid=len(arr),
            mean=m, bias=b, variance=var,
            rmse=float(np.sqrt(b**2 + var)),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
        ))

    print(f"\n  MLE-ungrouped sub-parameters (n_valid={len(ms_valid)}/{n_total}):")
    for name, arr, true in [("p_on", ms_p_on_v, P.p_on),
                             ("p_off", ms_p_off_v, P.p_off),
                             ("mu_emit", ms_me_v, P.mu_emit)]:
        m = np.mean(arr)
        b = m - true
        var = np.var(arr, ddof=0)
        ci_str = ""
        if P.n_outer > 1 and name in outer_ms_params and len(outer_ms_params[name]) > 1:
            ob = np.array(outer_ms_params[name])
            se = np.std(ob, ddof=1) / np.sqrt(len(ob))
            ci_str = f"  95%CI=+/-{1.96*se:.4f}"
        print(f"    {name:>10s}: mean={m:.4f}  true={true:.4f}  "
              f"bias={b:+.4f}{ci_str}  var={var:.6f}")
        summary_rows.append(dict(
            method=f"MLEsimple_{name}", true_value=true, n_valid=len(arr),
            mean=m, bias=b, variance=var,
            rmse=float(np.sqrt(b**2 + var)),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
        ))

    # ── Save CSV ─────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  SAVING DATA")
    print(f"{'=' * 62}")

    param_dict = {
        "method_version": cfg.METHOD_VERSION, "nudge": 1.0,
        "transition_lower": cfg.TRANSITION_BOUNDS[0], "transition_upper": cfg.TRANSITION_BOUNDS[1],
        "T": P.T, "pi_on": P.pi_on, "tau_emit": P.tau_emit,
        "p_on": P.p_on, "p_off": P.p_off,
        "mu_emit": P.mu_emit, "mu_true": P.mu_true,
        "size_mu": P.size_mu, "size_sigma": P.size_sigma,
        "p_snap": P.p_snap, "p_cont": P.p_cont,
        "T_cont": P.T_cont,
        "snap_threshold": P.snap_threshold, "snap_slope": P.snap_slope,
        "snap_sigma": P.snap_sigma,
        "cont_threshold": P.cont_threshold, "cont_slope": P.cont_slope,
        "cont_sigma": P.cont_sigma,
        "cont_fp_rate": P.cont_fp_rate, "cont_fp_scale": P.cont_fp_scale,
        "max_gap": P.max_gap, "decision_threshold": P.decision_threshold,
        "p_grid_res": P.p_grid_res, "p_geom_factor": P.p_geom_factor,
        "seed": P.seed, "n_reps": P.n_reps, "n_outer": P.n_outer,
    }
    _save_csv(res_dir / "data_parameters.csv", pd.DataFrame([param_dict]))

    df_reps = pd.DataFrame({
        "rep": np.arange(n_total),
        **{key: [r[key] for r in all_results] for key in
           ('n_detections','converged','n_iterations','optimizer_success',
            'has_detections','boundary','ms_optimizer_success')},
        "true_mean_empirical": true_mean_arr,
        "true_mean_theoretical": mu_true,
        "naive": naive_arr,
        "pod_weighted": pod_arr,
        "mle_simple": ms_arr,
        "ms_p_on": ms_p_on_arr,
        "ms_p_off": ms_p_off_arr,
        "ms_mu_emit": ms_mu_emit_arr,
        "mle": mle_arr,
        "mle_p_on": p_on_arr,
        "mle_p_off": p_off_arr,
        "mle_mu_emit": mu_emit_arr,
        "n_snap_obs": n_snap_arr,
        "n_cont_obs": n_cont_arr,
        "n_both_obs": n_both_arr,
    })
    _save_csv(res_dir / "data_replications.csv", df_reps)
    _save_csv(res_dir / "data_summary.csv", pd.DataFrame(summary_rows))

    e_grid = np.logspace(-1, 3, 500)
    df_pod = pd.DataFrame({
        "emission_size_kgh": e_grid,
        "pod_snapshots": tech_specs[0].pod.probability(e_grid),
        "pod_continuous": tech_specs[1].pod.probability(e_grid),
    })
    _save_csv(res_dir / "data_pod_curves.csv", df_pod)

    # ── Figures ───────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  GENERATING FIGURES")
    print(f"{'=' * 62}")

    figure_pod_curves(tech_specs, fig_dir / "figure_pod_curves.pdf",
                      snap_threshold=P.snap_threshold,
                      snap_slope=P.snap_slope,
                      cont_threshold=P.cont_threshold,
                      cont_slope=P.cont_slope)
    figure_violin(naive_arr, pod_arr, ms_valid, mle_valid, mu_true,
                  fig_dir / "figure_baseline_violin.pdf",
                  outer_stats=outer_stats if P.n_outer > 1 else None)
    figure_violin_v2(naive_arr, pod_arr, ms_valid, mle_valid, mu_true,
                     fig_dir / "figure_baseline_violin_v2.pdf",
                     outer_stats=outer_stats if P.n_outer > 1 else None)
    figure_params(p_on_v, p_off_v, me_v, mle_valid,
                  fig_dir / "figure_baseline_params.pdf",
                  true_p_on=P.p_on, true_p_off=P.p_off,
                  true_mu_emit=P.mu_emit, true_mu=P.mu_true)
    figure_size_distribution(P, fig_dir / "figure_size_distribution.pdf")

    chosen_reps = figure_realizations(
        tech_specs, P, fig_dir / "figure_realizations.pdf")

    for rep in chosen_reps:
        rng_m = np.random.default_rng(P.seed + rep)
        rng_e = np.random.default_rng(P.seed + rep + 100_000)
        snap_mask, cont_mask = generate_masks(
            P.T, P.p_snap, P.p_cont, P.T_cont, rng_m)
        obs_r, states_r, sizes_r, eids_r = simulate_series(
            T=P.T, p_on=P.p_on, p_off=P.p_off,
            size_mu=P.size_mu, size_sigma=P.size_sigma,
            tech_specs=tech_specs,
            snap_mask=snap_mask, cont_mask=cont_mask, rng=rng_e)
        df_real = pd.DataFrame({
            "t": np.arange(P.T),
            "state_on": states_r.astype(int),
            "true_emission": sizes_r,
            "event_id": eids_r,
            "snap_observed": obs_r.snap_mask.astype(int),
            "snap_detected": obs_r.snap_detected.astype(int),
            "snap_measurement": obs_r.snap_measurements,
            "cont_observed": obs_r.cont_mask.astype(int),
            "cont_detected": obs_r.cont_detected.astype(int),
            "cont_measurement": obs_r.cont_measurements,
        })
        _save_csv(res_dir / f"data_realization_rep{rep}.csv", df_real)

    # ── Auto-copy to paper directories (only for default baseline) ──
    if not P.output_dir:
        _sync_to_paper(fig_dir, res_dir)
    else:
        print(f"\n  Skipping paper sync (custom output-dir: {P.output_dir})")

    print(f"\n{'=' * 62}")
    print("  DONE")
    print(f"{'=' * 62}")


def _sync_to_paper(fig_dir: Path, res_dir: Path):
    """Copy key outputs to the paper and SI figure directories,
    then regenerate derived figures (precision) that depend on these results."""
    import shutil

    paper_fig = PROJECT_ROOT / "mle_methane_paper" / "figures"
    si_fig = PROJECT_ROOT / "Supplementary Methane MLE" / "figures"

    copies = []

    if paper_fig.exists():
        mapping = {
            "figure_baseline_violin_v2.pdf": "figure2.pdf",
            "figure_baseline_params.pdf": "figure1_histograms.pdf",
        }
        for src_name, dst_name in mapping.items():
            src = fig_dir / src_name
            if src.exists():
                dst = paper_fig / dst_name
                shutil.copy2(src, dst)
                copies.append(f"  {src_name} -> paper/{dst_name}")

    if si_fig.exists():
        si_files = [
            "figure_pod_curves.pdf",
            "figure_realizations.pdf",
            "figure_size_distribution.pdf",
        ]
        for name in si_files:
            src = fig_dir / name
            if src.exists():
                shutil.copy2(src, si_fig / name)
                copies.append(f"  {name} -> SI/{name}")

    top_results = PROJECT_ROOT / "results"
    top_results.mkdir(parents=True, exist_ok=True)
    data_src = res_dir / "data_replications.csv"
    data_dst = top_results / "data_replications.csv"
    if data_src.exists() and data_src.resolve() != data_dst.resolve():
        shutil.copy2(data_src, data_dst)
        copies.append("  data_replications.csv -> results/")

    if copies:
        print(f"\n  Auto-synced to paper directories:")
        for c in copies:
            print(c)

    # Regenerate derived figures that depend on data_replications.csv
    try:
        from experiments.make_figure_precision import main as precision_main
        print("\n  Regenerating precision figure ...")
        precision_main()
    except Exception as exc:
        print(f"  [WARN] Could not regenerate precision figure: {exc}")

    try:
        import sys as _sys
        _saved_argv = _sys.argv
        _sys.argv = [_sys.argv[0]]  # reset argv so argparse in make_figure1 works
        from experiments.make_figure1 import main as fig1_main
        print("\n  Regenerating Figure 1 (histograms + composite) ...")
        fig1_main()
        _sys.argv = _saved_argv
    except Exception as exc:
        print(f"  [WARN] Could not regenerate Figure 1: {exc}")


if __name__ == "__main__":
    main()
