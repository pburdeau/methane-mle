"""
Unified parameter sweeps
========================
Five sensitivity sweeps, each varying one parameter with all others at
baseline:

  1. T          — campaign length  (bias + variance of all MLE params)
  2. p_cont     — continuous monitoring probability
  3. T_cont     — continuous window duration
  4. theta_snap — snapshot 50% POD threshold
  5. tau_emit   — mean emission duration (1/p_off)

Sweeps 2–5 produce a two-row figure: variance of the three mu estimators
(top) and variance ratio POD/MLE (bottom).  Sweep 1 additionally shows
bias and variance of the individual MLE parameters.

Usage:
    python -m experiments.run_sweeps                    # all sweeps
    python -m experiments.run_sweeps --sweep T           # just one
    python -m experiments.run_sweeps --n-inner 100 --n-outer 5  # quick
    python -m experiments.run_sweeps --plot-only         # re-plot from CSV
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
    naive_baseline,
    pod_weighted_baseline,
    mle_simple,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results_linear"

# ══════════════════════════════════════════════════════════════════
# Sweep parameter grids
# ══════════════════════════════════════════════════════════════════

SWEEP_DEFS = {
    "T": {
        # "values": [250, 500, 1000, 2000, 4000, 8000],
        "values": [125, 250, 500, 1000, 2000],
        # "values": [1000, 2000, 4000],
        "label": r"Campaign length $T$ (hours)",
        "short": "T",
    },
    "p_cont": {
        "values": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00],
        "label": r"Continuous monitoring prob. $p_{\mathrm{cont}}$",
        "short": "p_cont",
    },
    "T_cont": {
        "values": [0, 25, 50, 75, 100, 125, 150, 175, 200],
        "label": r"Continuous window duration $T_{\mathrm{cont}}$",
        "short": "T_cont",
    },
    "theta_snap": {
        "values": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "label": r"Snapshot 50\% POD threshold $\theta_{\mathrm{snap}}$ (kg/h)",
        "short": "theta_snap",
    },
    "tau_emit": {
        "values": sorted(set(
            [round(v, 1) for v in np.geomspace(1, 100, 15).tolist()]
        )),
        "label": r"Mean emission duration $\tau_{\mathrm{emit}}$ (steps)",
        "short": "tau_emit",
    },
    "p_off": {
        "values": sorted(
            [1.0 / round(v, 1) for v in np.geomspace(1, 100, 15).tolist()
             if round(v, 1) > 0],
            reverse=True,
        ),
        "label": r"Mean emission duration $\tau_{\mathrm{emit}}$ (steps) — fixed $p_{\mathrm{on}}$",
        "short": "p_off",
        "display_transform": lambda v: 1.0 / v,
    },
}

# ══════════════════════════════════════════════════════════════════
# Core simulation runner
# ══════════════════════════════════════════════════════════════════

def _build_specs(theta_snap=None):
    return build_baseline_specs(
        aerial_threshold=theta_snap or cfg.AERIAL_THRESHOLD,
        aerial_slope=cfg.AERIAL_SLOPE,
        aerial_sensor_sigma=cfg.AERIAL_SENSOR_SIGMA,
        cont_threshold=cfg.CONT_THRESHOLD,
        cont_slope=cfg.CONT_SLOPE,
        cont_sensor_sigma=cfg.CONT_SENSOR_SIGMA,
        cont_fp_rate=cfg.CONT_FP_RATE,
        cont_fp_scale=cfg.CONT_FP_SCALE,
    )


def _run_one_task(task):
    return _run_one(*task)


def _run_one(tech_specs, rep, T, p_snap, p_cont, T_cont,
             p_on, p_off, size_mu, size_sigma,
             max_gap, decision_threshold, seed,
             skip_full_mle=False,
             p_on_grid=None, p_off_grid=None,
             p_off_nudge=None):
    mask_rng = np.random.default_rng(seed + rep)
    emit_rng = np.random.default_rng(seed + rep + 100_000)

    snap_mask, cont_mask = generate_masks(T, p_snap, p_cont, T_cont, mask_rng)
    obs, _st, _sz, _eid = simulate_series(
        T=T, p_on=p_on, p_off=p_off,
        size_mu=size_mu, size_sigma=size_sigma,
        tech_specs=tech_specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=emit_rng,
    )

    nudge = p_off_nudge if p_off_nudge is not None else cfg.NUDGE
    naive_est = naive_baseline(obs)
    pod_est = pod_weighted_baseline(obs, tech_specs)
    ms_res = mle_simple(obs, tech_specs, transition_bounds=cfg.TRANSITION_BOUNDS,
                        p_on_grid=p_on_grid, p_off_grid=p_off_grid,
                        p_off_nudge=nudge)
    n_det = obs.n_total_detections()

    if skip_full_mle:
        return dict(naive=naive_est, pod=pod_est,
                    mle_simple=ms_res["mean"],
                    ms_p_on=ms_res["p_on"], ms_p_off=ms_res["p_off"],
                    ms_mu_emit=ms_res["mu_emit"],
                    mle=np.nan,
                    mle_p_on=np.nan, mle_p_off=np.nan, mle_mu_emit=np.nan)

    res = loop_empirical(obs, tech_specs, T, **cfg.MLE_OPTIONS,
                         max_gap=max_gap,
                         decision_threshold=decision_threshold,
                         p_on_grid=p_on_grid, p_off_grid=p_off_grid,
                         p_off_nudge=nudge)
    return dict(naive=naive_est, pod=pod_est,
                mle_simple=ms_res["mean"],
                ms_p_on=ms_res["p_on"], ms_p_off=ms_res["p_off"],
                ms_mu_emit=ms_res["mu_emit"],
                mle=res["mean"],
                mle_p_on=res["p_on"], mle_p_off=res["p_off"],
                mle_mu_emit=res["mu_emit"])


# ══════════════════════════════════════════════════════════════════
# Generic sweep engine
# ══════════════════════════════════════════════════════════════════

def run_sweep(sweep_name: str, n_inner: int, n_outer: int,
              csv_path: Path | None = None, vi_offset: int = 0,
              skip_full_mle: bool = False):
    sdef = SWEEP_DEFS[sweep_name]
    values = sdef["values"]

    records = []

    for vi_raw, val in enumerate(values):
        vi = vi_raw + vi_offset
        # Override the swept parameter
        T = cfg.T
        p_snap = cfg.P_SNAP
        p_cont = cfg.P_CONT
        T_cont = cfg.T_CONT
        p_on = cfg.P_ON
        p_off = cfg.P_OFF
        theta_snap = cfg.AERIAL_THRESHOLD

        if sweep_name == "T":
            T = val
        elif sweep_name == "p_cont":
            p_cont = val
        elif sweep_name == "T_cont":
            T_cont = val
        elif sweep_name == "theta_snap":
            theta_snap = val
        elif sweep_name == "tau_emit":
            tau = val
            p_off = 1.0 / tau
            p_on = cfg.PI_ON * p_off / (1.0 - cfg.PI_ON)
        elif sweep_name == "p_off":
            p_off = val

        tech_specs = _build_specs(theta_snap)
        mu_true = p_on / (p_on + p_off) * cfg.MU_EMIT
        seed = cfg.BASE_SEED + vi * 1_000_000
        p_on_grid = None  # fixed-bound continuous optimization
        p_off_grid = None

        print(f"\n{'=' * 62}")
        print(f"  {sweep_name} = {val}")
        print(f"{'=' * 62}")

        var_outer = {"naive": [], "pod": [], "mle_simple": [], "mle": []}
        mean_outer = {"naive": [], "pod": [], "mle_simple": [], "mle": []}
        param_var_outer = {"p_on": [], "p_off": [], "mu_emit": []}
        param_mean_outer = {"p_on": [], "p_off": [], "mu_emit": []}
        ms_param_var_outer = {"p_on": [], "p_off": [], "mu_emit": []}
        ms_param_mean_outer = {"p_on": [], "p_off": [], "mu_emit": []}
        t0 = time.time()

        for outer in range(n_outer):
            ests = {"naive": [], "pod": [], "mle_simple": [], "mle": []}
            param_inner = {"p_on": [], "p_off": [], "mu_emit": []}
            ms_param_inner = {"p_on": [], "p_off": [], "mu_emit": []}
            from experiments.parallel import ordered_map
            tasks = [(tech_specs,outer*n_inner+inner,T,p_snap,p_cont,T_cont,
                      p_on,p_off,cfg.SIZE_MU,cfg.SIZE_SIGMA,cfg.MAX_GAP,
                      cfg.DECISION_THRESHOLD,seed,skip_full_mle,None,None)
                     for inner in range(n_inner)]
            for r in ordered_map(_run_one_task,tasks):
                for k in ests:
                    ests[k].append(r[k])
                param_inner["p_on"].append(r["mle_p_on"])
                param_inner["p_off"].append(r["mle_p_off"])
                param_inner["mu_emit"].append(r["mle_mu_emit"])
                ms_param_inner["p_on"].append(r["ms_p_on"])
                ms_param_inner["p_off"].append(r["ms_p_off"])
                ms_param_inner["mu_emit"].append(r["ms_mu_emit"])

            for k in var_outer:
                arr = np.array(ests[k])
                valid = arr[np.isfinite(arr)]
                var_outer[k].append(
                    float(np.var(valid, ddof=0)) if len(valid) > 1 else np.nan)
                mean_outer[k].append(
                    float(np.mean(valid)) if len(valid) else np.nan)

            for pk in param_var_outer:
                a = np.array(param_inner[pk])
                v = a[np.isfinite(a)]
                param_var_outer[pk].append(
                    float(np.var(v, ddof=0)) if len(v) > 1 else np.nan)
                param_mean_outer[pk].append(
                    float(np.mean(v)) if len(v) else np.nan)

            for pk in ms_param_var_outer:
                a = np.array(ms_param_inner[pk])
                v = a[np.isfinite(a)]
                ms_param_var_outer[pk].append(
                    float(np.var(v, ddof=0)) if len(v) > 1 else np.nan)
                ms_param_mean_outer[pk].append(
                    float(np.mean(v)) if len(v) else np.nan)

            el = time.time() - t0
            eta = el / (outer + 1) * (n_outer - outer - 1)
            print(f"    outer {outer+1:>3d}/{n_outer}  "
                  f"elapsed={el:.0f}s  eta={eta:.0f}s", flush=True)

        # Aggregate
        for method in ("naive", "pod", "mle_simple", "mle"):
            v_arr = np.array(var_outer[method])
            v_valid = v_arr[np.isfinite(v_arr)]
            mean_var = float(np.mean(v_valid)) if len(v_valid) else np.nan
            se_var = (float(np.std(v_valid, ddof=1) / np.sqrt(len(v_valid)))
                      if len(v_valid) > 1 else np.nan)
            m_arr = np.array(mean_outer[method])
            m_valid = m_arr[np.isfinite(m_arr)]
            mean_est = float(np.mean(m_valid)) if len(m_valid) else np.nan
            bias = (mean_est - mu_true) if np.isfinite(mean_est) else np.nan
            se_mean = (float(np.std(m_valid, ddof=1) / np.sqrt(len(m_valid)))
                       if len(m_valid) > 1 else np.nan)

            rec = dict(
                sweep=sweep_name, sweep_val=val, method=method,
                variance=mean_var, variance_se=se_var,
                mean_estimate=mean_est, bias=bias, bias_se=se_mean,
                mu_true=mu_true,
                n_outer=n_outer, n_inner=n_inner,
            )
            if method == "mle":
                for tag, true_val in [
                    ("p_on", p_on),
                    ("p_off", p_off),
                    ("mu_emit", cfg.MU_EMIT),
                ]:
                    m_arr = np.array(param_mean_outer[tag])
                    m_valid = m_arr[np.isfinite(m_arr)]
                    v_arr = np.array(param_var_outer[tag])
                    v_valid = v_arr[np.isfinite(v_arr)]
                    rec[f"mle_{tag}_mean"] = float(np.mean(m_valid)) if len(m_valid) else np.nan
                    rec[f"mle_{tag}_bias"] = (float(np.mean(m_valid)) - true_val) if len(m_valid) else np.nan
                    rec[f"mle_{tag}_bias_se"] = (float(np.std(m_valid, ddof=1) / np.sqrt(len(m_valid)))
                                                 if len(m_valid) > 1 else np.nan)
                    rec[f"mle_{tag}_true"] = true_val
                    rec[f"mle_{tag}_var"] = float(np.mean(v_valid)) if len(v_valid) else np.nan
                    rec[f"mle_{tag}_var_se"] = (float(np.std(v_valid, ddof=1) / np.sqrt(len(v_valid)))
                                                if len(v_valid) > 1 else np.nan)

            if method == "mle_simple":
                for tag, true_val in [
                    ("p_on", p_on),
                    ("p_off", p_off),
                    ("mu_emit", cfg.MU_EMIT),
                ]:
                    m_arr = np.array(ms_param_mean_outer[tag])
                    m_valid = m_arr[np.isfinite(m_arr)]
                    v_arr = np.array(ms_param_var_outer[tag])
                    v_valid = v_arr[np.isfinite(v_arr)]
                    rec[f"ms_{tag}_mean"] = float(np.mean(m_valid)) if len(m_valid) else np.nan
                    rec[f"ms_{tag}_bias"] = (float(np.mean(m_valid)) - true_val) if len(m_valid) else np.nan
                    rec[f"ms_{tag}_bias_se"] = (float(np.std(m_valid, ddof=1) / np.sqrt(len(m_valid)))
                                                 if len(m_valid) > 1 else np.nan)
                    rec[f"ms_{tag}_true"] = true_val
                    rec[f"ms_{tag}_var"] = float(np.mean(v_valid)) if len(v_valid) else np.nan
                    rec[f"ms_{tag}_var_se"] = (float(np.std(v_valid, ddof=1) / np.sqrt(len(v_valid)))
                                                if len(v_valid) > 1 else np.nan)

            records.append(rec)

        print(f"    Naive bias: {records[-4]['bias']:+.3f}  "
              f"MLE-ungrouped bias: {records[-2]['bias']:+.3f}  "
              f"MLE bias: {records[-1]['bias']:+.3f}")

        if csv_path is not None:
            pd.DataFrame(records).to_csv(csv_path, index=False,
                                         float_format="%.6f")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════
# Figure: variance + ratio (sweeps 2–5)
# ══════════════════════════════════════════════════════════════════

_RC = {
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "lines.linewidth": 2,
}

C_NAIVE, C_POD, C_MLE, C_RATIO = "#D98880", "#C39BD3", "#2ECC71", "#1A1A1A"


C_MS = "#85C1E9"


def figure_variance_ratio(df: pd.DataFrame, sweep_name: str, out: Path):
    plt.rcParams.update(_RC)
    sdef = SWEEP_DEFS[sweep_name]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    vals = sorted(df["sweep_val"].unique())

    naive_v, naive_se = [], []
    pod_v, pod_se = [], []
    ms_v, ms_se = [], []
    mle_v, mle_se = [], []
    ratios, ratio_se = [], []
    ratios_ms, ratio_ms_se = [], []

    for v in vals:
        for method, vlist, slist in [
            ("naive", naive_v, naive_se),
            ("pod", pod_v, pod_se),
            ("mle_simple", ms_v, ms_se),
            ("mle", mle_v, mle_se),
        ]:
            r = df[(df["sweep_val"] == v) & (df["method"] == method)]
            vlist.append(r["variance"].values[0] if len(r) else np.nan)
            slist.append(r["variance_se"].values[0] if len(r) else np.nan)

        vp, sp = pod_v[-1], pod_se[-1]
        vm, sm = mle_v[-1], mle_se[-1]
        vms, sms = ms_v[-1], ms_se[-1]
        if vm > 0 and vp > 0:
            rat = vp / vm
            rat_se_v = rat * np.sqrt((sp / vp) ** 2 + (sm / vm) ** 2)
        else:
            rat, rat_se_v = np.nan, np.nan
        ratios.append(rat)
        ratio_se.append(rat_se_v)
        if vms > 0 and vm > 0:
            rat_ms = vms / vm
            rat_ms_se_v = rat_ms * np.sqrt((sms / vms) ** 2 + (sm / vm) ** 2)
        else:
            rat_ms, rat_ms_se_v = np.nan, np.nan
        ratios_ms.append(rat_ms)
        ratio_ms_se.append(rat_ms_se_v)

    vals = np.array(vals, dtype=float)
    display_fn = sdef.get("display_transform")
    if display_fn is not None:
        vals = np.array([display_fn(v) for v in vals])
    naive_v = np.array(naive_v); naive_se = np.array(naive_se)
    pod_v = np.array(pod_v); pod_se = np.array(pod_se)
    ms_v = np.array(ms_v); ms_se = np.array(ms_se)
    mle_v = np.array(mle_v); mle_se = np.array(mle_se)
    ratios = np.array(ratios); ratio_se = np.array(ratio_se)
    ratios_ms = np.array(ratios_ms); ratio_ms_se = np.array(ratio_ms_se)

    for v, se, col, lab in [
        (naive_v, naive_se, C_NAIVE, "Naive"),
        (pod_v, pod_se, C_POD, "POD-weighted"),
        (ms_v, ms_se, C_MS, "MLE-ungrouped"),
        (mle_v, mle_se, C_MLE, "MLE"),
    ]:
        ax1.plot(vals, v, "o-", color=col, label=lab,
                 markersize=6, markeredgewidth=1, markeredgecolor="white")
        ax1.fill_between(vals, v - 1.96 * se, v + 1.96 * se,
                         alpha=0.15, color=col, linewidth=0)

    ax1.set_ylabel(r"Variance of $\hat\mu$", fontweight="bold")
    ax1.legend(loc="best", framealpha=0.9, edgecolor="gray")
    ax1.grid(alpha=0.3, ls=":", lw=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(f"Variance vs {sdef['short']}", fontweight="bold")

    ax2.plot(vals, ratios, "o-", color=C_RATIO, linewidth=2.5, markersize=7,
             markeredgewidth=1.5, markeredgecolor="white",
             label="POD / MLE")
    ax2.fill_between(vals, ratios - 1.96 * ratio_se,
                     ratios + 1.96 * ratio_se,
                     alpha=0.15, color=C_RATIO, linewidth=0)
    ax2.plot(vals, ratios_ms, "s--", color=C_MS, linewidth=2, markersize=6,
             markeredgewidth=1.5, markeredgecolor="white",
             label="MLE-ungrouped / MLE")
    ax2.fill_between(vals, ratios_ms - 1.96 * ratio_ms_se,
                     ratios_ms + 1.96 * ratio_ms_se,
                     alpha=0.15, color=C_MS, linewidth=0)
    ax2.axhline(1, color="gray", ls="--", lw=1.5, alpha=0.8)
    ax2.set_xlabel(sdef["label"], fontweight="bold")
    ax2.set_ylabel("Variance ratio", fontweight="bold")
    ax2.set_xticks(vals)
    ax2.legend(loc="best", framealpha=0.9, edgecolor="gray")
    ax2.grid(alpha=0.3, ls=":", lw=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════
# Figure: parameter accuracy vs T  (sweep 1)
# ══════════════════════════════════════════════════════════════════

def figure_param_accuracy(df: pd.DataFrame, out: Path):
    """Relative bias (%) and CV (%) of each MLE parameter as a function of T."""
    plt.rcParams.update(_RC)

    mle = df[df["method"] == "mle"].copy().sort_values("sweep_val")
    Ts = mle["sweep_val"].values

    params = [
        (r"$p_{\mathrm{on}}$", "mle_p_on"),
        (r"$p_{\mathrm{off}}$", "mle_p_off"),
        (r"$\mu_{\mathrm{emit}}$", "mle_mu_emit"),
        (r"$\hat\mu$", None),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    colors = ["#9B59B6", "#E67E22", "#2ECC71", "#3498DB"]

    for col, (label, prefix) in enumerate(params):
        ax_bias = axes[0, col]
        ax_cv = axes[1, col]

        if prefix is not None:
            bias_vals = mle[f"{prefix}_bias"].values
            bias_se = mle[f"{prefix}_bias_se"].values
            true_val = mle[f"{prefix}_true"].values
            var_vals = mle[f"{prefix}_var"].values
            var_se = mle[f"{prefix}_var_se"].values
        else:
            bias_vals = mle["bias"].values
            bias_se = mle["bias_se"].values
            true_val = mle["mu_true"].values
            var_vals = mle["variance"].values
            var_se = mle["variance_se"].values

        rel_bias = bias_vals / true_val * 100
        rel_bias_se = bias_se / true_val * 100
        cv = np.sqrt(var_vals) / true_val * 100
        cv_se = var_se / (2 * np.sqrt(var_vals) * true_val) * 100

        ax_bias.plot(Ts, rel_bias, "o-", color=colors[col], markersize=5)
        ax_bias.fill_between(Ts, rel_bias - 1.96 * rel_bias_se,
                             rel_bias + 1.96 * rel_bias_se,
                             alpha=0.15, color=colors[col])
        ax_bias.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
        ax_bias.set_title(f"{label}", fontweight="bold")
        ax_bias.grid(alpha=0.3, ls=":", lw=0.5)
        ax_bias.spines["top"].set_visible(False)
        ax_bias.spines["right"].set_visible(False)

        ax_cv.plot(Ts, cv, "s-", color=colors[col], markersize=5)
        ax_cv.fill_between(Ts, cv - 1.96 * cv_se,
                           cv + 1.96 * cv_se,
                           alpha=0.15, color=colors[col])
        ax_cv.set_xlabel(r"$T$ (hours)")
        ax_cv.grid(alpha=0.3, ls=":", lw=0.5)
        ax_cv.spines["top"].set_visible(False)
        ax_cv.spines["right"].set_visible(False)

    axes[0, 0].set_ylabel("Relative bias (%)", fontweight="bold")
    axes[1, 0].set_ylabel("CV (%)", fontweight="bold")

    fig.suptitle("Parameter Recovery vs Campaign Length",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

ALL_SWEEPS = list(SWEEP_DEFS.keys())


def main():
    parser = argparse.ArgumentParser(description="Unified parameter sweeps")
    parser.add_argument("--sweep", type=str, nargs="+", default=ALL_SWEEPS,
                        choices=ALL_SWEEPS,
                        help="Which sweep(s) to run")
    parser.add_argument("--n-inner", type=int, default=500,
                        help="Inner replications per outer batch")
    parser.add_argument("--n-outer", type=int, default=5,
                        help="Outer batches (for SE estimation)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from saved CSV without re-running")
    parser.add_argument("--extra-values", type=float, nargs="+", default=None,
                        help="Additional sweep values to run and append to "
                             "existing CSV (e.g. --extra-values 4000 8000)")
    parser.add_argument("--skip-full-mle", action="store_true",
                        help="Skip full MLE (only compute naive, pod, mle_simple)")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help="Override figure output directory")
    parser.add_argument("--res-dir", type=str, default=None,
                        help="Override results output directory")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    res_dir = Path(args.res_dir) if args.res_dir else RES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    for sweep_name in args.sweep:
        csv_path = res_dir / f"sweep_{sweep_name}_summary.csv"
        fig_path = fig_dir / f"figure_sweep_{sweep_name}.pdf"

        if args.extra_values is not None:
            extra = [int(v) if v == int(v) else v for v in args.extra_values]
            sdef_copy = dict(SWEEP_DEFS[sweep_name])
            sdef_copy["values"] = extra
            orig_values = SWEEP_DEFS[sweep_name]["values"]
            SWEEP_DEFS[sweep_name]["values"] = extra

            print("\n" + "=" * 62)
            print(f"  SWEEP (extra): {sweep_name}")
            print(f"  Extra values: {extra}")
            print(f"  n_inner={args.n_inner}, n_outer={args.n_outer}")
            print("=" * 62)

            df_extra = run_sweep(sweep_name, args.n_inner, args.n_outer,
                                 csv_path=None, vi_offset=100,
                                 skip_full_mle=args.skip_full_mle)
            SWEEP_DEFS[sweep_name]["values"] = orig_values

            if csv_path.exists():
                df_existing = pd.read_csv(csv_path)
                df_existing = df_existing[
                    ~df_existing["sweep_val"].isin(extra)]
                df = pd.concat([df_existing, df_extra], ignore_index=True)
            else:
                df = df_extra
            df = df.sort_values("sweep_val").reset_index(drop=True)
            df.to_csv(csv_path, index=False, float_format="%.6f")
            print(f"\n  Saved (merged): {csv_path}")

        elif not args.plot_only:
            print("\n" + "=" * 62)
            print(f"  SWEEP: {sweep_name}")
            print(f"  Values: {SWEEP_DEFS[sweep_name]['values']}")
            print(f"  n_inner={args.n_inner}, n_outer={args.n_outer}")
            if args.skip_full_mle:
                print(f"  MODE: mle_simple only (skipping full MLE)")
            print("=" * 62)

            df_new = run_sweep(sweep_name, args.n_inner, args.n_outer,
                               csv_path=None,
                               skip_full_mle=args.skip_full_mle)

            if args.skip_full_mle and csv_path.exists():
                df_existing = pd.read_csv(csv_path)
                df_existing = df_existing[
                    df_existing["method"] != "mle_simple"]
                df_ms = df_new[df_new["method"] == "mle_simple"]
                df = pd.concat([df_existing, df_ms], ignore_index=True)
                df = df.sort_values(["sweep_val", "method"]).reset_index(
                    drop=True)
            else:
                df = df_new

            df.to_csv(csv_path, index=False, float_format="%.6f")
            print(f"\n  Saved: {csv_path}")
        else:
            df = pd.read_csv(csv_path)

        if sweep_name == "T":
            figure_param_accuracy(df, fig_dir / "figure_sweep_T_params.pdf")
        figure_variance_ratio(df, sweep_name, fig_path)

    print("\n" + "=" * 62)
    print("  ALL SWEEPS DONE")
    print("=" * 62)

    if sweep_name == "T" and (res_dir.parent / "results/data_replications.csv").exists():
        print("\n  Generating composite Figure 1...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "experiments.make_figure1",
             "--res-dir", str(res_dir.parent),
             "--out", str(fig_dir / "figure1_composite.pdf")],
            cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
