"""
Sweep over realistic snapshot observation frequencies (p_snap).
===============================================================

Cases:
  Quarterly aerial:  p_snap = 0.0005
  Monthly aerial:    p_snap = 0.0015
  Weekly aerial:     p_snap = 0.006
  Daily satellite:   p_snap = 0.04
  Baseline:          p_snap = 0.30

Runs at T=1000 for all, plus T=5000 and T=10000 for the two lowest
frequencies (quarterly & monthly) where T=1000 may yield too few
detections.

Usage:
    python -m experiments.run_sweep_psnap
    python -m experiments.run_sweep_psnap --plot-only
    python -m experiments.run_sweep_psnap --n-inner 100 --n-outer 5  # quick
"""

from __future__ import annotations

import argparse
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
RES_DIR = PROJECT_ROOT / "results"

PSNAP_VALUES = [0.0014, 0.003, 0.007, 0.015, 0.03, 0.06, 0.125, 0.25, 0.5]
PSNAP_LABELS = {
    0.0014: "Monthly",
    0.003:  "Bi-weekly",
    0.007:  "Weekly",
    0.015:  "",
    0.03:   "",
    0.06:   "",
    0.125:  "Baseline",
    0.25:   "",
    0.5:    "Sub-hourly",
}

T_VALUES = [1000]
EXTENDED_T = []
EXTENDED_PSNAP = []


def _build_specs():
    return build_baseline_specs(
        aerial_threshold=cfg.AERIAL_THRESHOLD,
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
             p_on_grid=None, p_off_grid=None):
    mask_rng = np.random.default_rng(seed + rep)
    emit_rng = np.random.default_rng(seed + rep + 100_000)

    snap_mask, cont_mask = generate_masks(T, p_snap, p_cont, T_cont, mask_rng)
    obs, _st, _sz, _eid = simulate_series(
        T=T, p_on=p_on, p_off=p_off,
        size_mu=size_mu, size_sigma=size_sigma,
        tech_specs=tech_specs,
        snap_mask=snap_mask, cont_mask=cont_mask, rng=emit_rng,
    )

    naive_est = naive_baseline(obs)
    pod_est = pod_weighted_baseline(obs, tech_specs)
    ms_res = mle_simple(obs, tech_specs, transition_bounds=cfg.TRANSITION_BOUNDS,
                        p_on_grid=p_on_grid, p_off_grid=p_off_grid,
                        p_off_nudge=cfg.NUDGE)
    n_det = obs.n_total_detections()
    n_snap_obs = int(np.sum(snap_mask))
    n_cont_obs = int(np.sum(cont_mask))

    res = loop_empirical(obs, tech_specs, T, **cfg.MLE_OPTIONS,
                         max_gap=max_gap,
                         decision_threshold=decision_threshold,
                         p_on_grid=p_on_grid, p_off_grid=p_off_grid,
                         p_off_nudge=cfg.NUDGE)
    return dict(naive=naive_est, pod=pod_est,
                mle_simple=ms_res["mean"], mle=res["mean"],
                n_det=n_det, n_snap_obs=n_snap_obs,
                n_cont_obs=n_cont_obs)


def run_sweep(T_vals, psnap_vals, n_inner, n_outer, csv_path=None):
    tech_specs = _build_specs()
    mu_true = cfg.P_ON / (cfg.P_ON + cfg.P_OFF) * cfg.MU_EMIT
    p_on_grid = None  # fixed-bound continuous optimization
    p_off_grid = None

    records = []
    case_idx = 0

    for T in T_vals:
        for p_snap in psnap_vals:
            if T != T_vals[0] and p_snap not in EXTENDED_PSNAP:
                continue

            seed = cfg.BASE_SEED + case_idx * 1_000_000
            case_idx += 1

            print(f"\n{'=' * 62}")
            print(f"  T={T}, p_snap={p_snap} ({PSNAP_LABELS.get(p_snap, '')})")
            print(f"{'=' * 62}")

            var_outer = {"naive": [], "pod": [], "mle_simple": [], "mle": []}
            mean_outer = {"naive": [], "pod": [], "mle_simple": [], "mle": []}
            det_counts = []
            snap_obs_counts = []
            t0 = time.time()

            for outer in range(n_outer):
                ests = {"naive": [], "pod": [], "mle_simple": [], "mle": []}
                batch_dets = []
                batch_snap_obs = []

                from experiments.parallel import ordered_map
                tasks = [(tech_specs,outer*n_inner+inner,T,p_snap,cfg.P_CONT,
                          cfg.T_CONT,cfg.P_ON,cfg.P_OFF,cfg.SIZE_MU,cfg.SIZE_SIGMA,
                          cfg.MAX_GAP,cfg.DECISION_THRESHOLD,seed,None,None)
                         for inner in range(n_inner)]
                for r in ordered_map(_run_one_task,tasks):
                    for k in ests:
                        ests[k].append(r[k])
                    batch_dets.append(r["n_det"])
                    batch_snap_obs.append(r["n_snap_obs"])

                for k in var_outer:
                    arr = np.array(ests[k])
                    valid = arr[np.isfinite(arr)]
                    var_outer[k].append(
                        float(np.var(valid, ddof=0)) if len(valid) > 1
                        else np.nan)
                    mean_outer[k].append(
                        float(np.mean(valid)) if len(valid) else np.nan)

                det_counts.extend(batch_dets)
                snap_obs_counts.extend(batch_snap_obs)

                el = time.time() - t0
                eta = el / (outer + 1) * (n_outer - outer - 1)
                print(f"    outer {outer+1:>3d}/{n_outer}  "
                      f"elapsed={el:.0f}s  eta={eta:.0f}s", flush=True)

            mean_n_det = float(np.mean(det_counts))
            mean_n_snap_obs = float(np.mean(snap_obs_counts))

            for method in ("naive", "pod", "mle_simple", "mle"):
                v_arr = np.array(var_outer[method])
                v_valid = v_arr[np.isfinite(v_arr)]
                mean_var = (float(np.mean(v_valid)) if len(v_valid)
                            else np.nan)
                se_var = (float(np.std(v_valid, ddof=1)
                                / np.sqrt(len(v_valid)))
                          if len(v_valid) > 1 else np.nan)
                m_arr = np.array(mean_outer[method])
                m_valid = m_arr[np.isfinite(m_arr)]
                mean_est = (float(np.mean(m_valid)) if len(m_valid)
                            else np.nan)
                bias = ((mean_est - mu_true) if np.isfinite(mean_est)
                        else np.nan)
                se_mean = (float(np.std(m_valid, ddof=1)
                                 / np.sqrt(len(m_valid)))
                           if len(m_valid) > 1 else np.nan)
                n_valid_frac = (np.sum(np.isfinite(
                    np.array(var_outer[method]))) / n_outer)

                records.append(dict(
                    T=T, p_snap=p_snap,
                    label=PSNAP_LABELS.get(p_snap, ""),
                    method=method,
                    variance=mean_var, variance_se=se_var,
                    mean_estimate=mean_est, bias=bias, bias_se=se_mean,
                    mu_true=mu_true,
                    mean_n_det=mean_n_det,
                    mean_n_snap_obs=mean_n_snap_obs,
                    valid_outer_frac=n_valid_frac,
                    n_outer=n_outer, n_inner=n_inner,
                ))

            recs_last4 = records[-4:]
            print(f"    Mean detections: {mean_n_det:.1f}  "
                  f"(snap obs: {mean_n_snap_obs:.1f})")
            for r in recs_last4:
                print(f"    {r['method']:>12s}: "
                      f"bias={r['bias']:+.3f}  var={r['variance']:.2f}")

            if csv_path is not None:
                pd.DataFrame(records).to_csv(
                    csv_path, index=False, float_format="%.6f")

    return pd.DataFrame(records)


def figure_psnap(df, out):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.linewidth": 2,
    })

    C_NAIVE = "#D98880"
    C_POD   = "#C39BD3"
    C_MS    = "#85C1E9"
    C_MLE   = "#2ECC71"
    C_RATIO = "#888888"

    T_groups = sorted(df["T"].unique())

    fig, axes = plt.subplots(2, len(T_groups), figsize=(6 * len(T_groups), 7),
                             squeeze=False, sharex=False)

    for col_idx, T in enumerate(T_groups):
        dft = df[df["T"] == T]
        psnap_vals = sorted(dft["p_snap"].unique())

        pod_v, pod_se_v = [], []
        mle_v, mle_se_v = [], []
        ms_v, ms_se_v = [], []
        naive_v, naive_se_v = [], []
        ratios, ratio_se_v = [], []

        for ps in psnap_vals:
            for method, vlist, slist in [
                ("naive", naive_v, naive_se_v),
                ("pod", pod_v, pod_se_v),
                ("mle_simple", ms_v, ms_se_v),
                ("mle", mle_v, mle_se_v),
            ]:
                r = dft[(dft["p_snap"] == ps) & (dft["method"] == method)]
                vlist.append(r["variance"].values[0] if len(r) else np.nan)
                slist.append(r["variance_se"].values[0] if len(r) else np.nan)

            vp, sp = pod_v[-1], pod_se_v[-1]
            vm, sm = mle_v[-1], mle_se_v[-1]
            if vm > 0 and vp > 0:
                rat = vp / vm
                rat_se = rat * np.sqrt((sp / vp) ** 2 + (sm / vm) ** 2)
            else:
                rat, rat_se = np.nan, np.nan
            ratios.append(rat)
            ratio_se_v.append(rat_se)

        x = np.array(psnap_vals)
        ratios = np.array(ratios)
        ratio_se_v = np.array(ratio_se_v)

        ax_var = axes[0, col_idx]
        for v, se, col, lab in [
            (naive_v, naive_se_v, C_NAIVE, "Naive"),
            (pod_v, pod_se_v, C_POD, "POD-weighted"),
            (ms_v, ms_se_v, C_MS, "MLE-ungrouped"),
            (mle_v, mle_se_v, C_MLE, "MLE"),
        ]:
            v = np.array(v); se = np.array(se)
            ax_var.plot(x, v, "o-", color=col, label=lab,
                        markersize=6, markeredgewidth=1,
                        markeredgecolor="white")
            ax_var.fill_between(x, v - 1.96 * se, v + 1.96 * se,
                                alpha=0.15, color=col, linewidth=0)

        ax_var.set_xscale("log")
        ax_var.set_ylabel(r"Variance of $\hat\mu$", fontweight="bold")
        ax_var.set_title(f"T = {T}", fontweight="bold")
        if col_idx == 0:
            ax_var.legend(loc="best", framealpha=0.9, edgecolor="gray")
        ax_var.grid(alpha=0.3, ls=":", lw=0.5)
        ax_var.spines["top"].set_visible(False)
        ax_var.spines["right"].set_visible(False)

        # Dual top x-axis: average hours between snapshots
        ax_top = ax_var.secondary_xaxis("top",
            functions=(lambda ps: 1.0 / ps, lambda h: 1.0 / h))
        ax_top.set_xlabel("Avg. hours between snapshots", fontsize=9)
        ax_top.set_xscale("log")

        for ps in psnap_vals:
            lbl = PSNAP_LABELS.get(ps, "")
            if lbl:
                ax_var.annotate(lbl, xy=(ps, 0), xycoords=("data", "axes fraction"),
                                fontsize=6, rotation=45, ha="right", va="top",
                                alpha=0.6)

        ax_rat = axes[1, col_idx]
        ax_rat.plot(x, ratios, "o-", color=C_RATIO, linewidth=2.5,
                    markersize=7, markeredgewidth=1.5,
                    markeredgecolor="white", label="POD / MLE")
        ax_rat.fill_between(x, ratios - 1.96 * ratio_se_v,
                            ratios + 1.96 * ratio_se_v,
                            alpha=0.15, color=C_RATIO, linewidth=0)
        ax_rat.axhline(1, color="gray", ls="--", lw=1.5, alpha=0.8)
        ax_rat.set_xscale("log")
        ax_rat.set_xlabel(r"$p_{\mathrm{snap}}$ (per hour)", fontweight="bold")
        ax_rat.set_ylabel("Variance ratio (POD / MLE)", fontweight="bold")
        ax_rat.legend(loc="best", framealpha=0.9, edgecolor="gray")
        ax_rat.grid(alpha=0.3, ls=":", lw=0.5)
        ax_rat.spines["top"].set_visible(False)
        ax_rat.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_summary_table(df):
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE: p_snap sweep")
    print("=" * 100)
    print(f"{'T':>6s}  {'p_snap':>8s}  {'Label':>12s}  "
          f"{'Method':>12s}  {'Bias':>8s}  {'Var':>8s}  "
          f"{'VarRatio':>8s}  {'MeanDet':>8s}")
    print("-" * 100)

    for T in sorted(df["T"].unique()):
        for ps in sorted(df[df["T"] == T]["p_snap"].unique()):
            dft = df[(df["T"] == T) & (df["p_snap"] == ps)]
            pod_var = dft[dft["method"] == "pod"]["variance"].values
            mle_var = dft[dft["method"] == "mle"]["variance"].values
            ratio = (pod_var[0] / mle_var[0]
                     if len(pod_var) and len(mle_var) and mle_var[0] > 0
                     else np.nan)
            for _, row in dft.iterrows():
                print(f"{T:>6d}  {ps:>8.4f}  {row['label']:>12s}  "
                      f"{row['method']:>12s}  {row['bias']:>+8.3f}  "
                      f"{row['variance']:>8.2f}  "
                      f"{ratio:>8.2f}  {row['mean_n_det']:>8.1f}")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-inner", type=int, default=500)
    parser.add_argument("--n-outer", type=int, default=25)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--T-only", type=int, default=None,
                        help="Run only this T value")
    parser.add_argument("--fig-dir", type=str, default=None)
    parser.add_argument("--res-dir", type=str, default=None)
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    res_dir = Path(args.res_dir) if args.res_dir else RES_DIR
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = res_dir / "sweep_psnap.csv"

    if args.plot_only:
        df = pd.read_csv(csv_path)
    else:
        all_T = T_VALUES + EXTENDED_T
        if args.T_only is not None:
            all_T = [args.T_only]

        df = run_sweep(all_T, PSNAP_VALUES,
                       args.n_inner, args.n_outer, csv_path)

    print_summary_table(df)
    figure_psnap(df, fig_dir / "figure_sweep_psnap.pdf")

    print("\n" + "=" * 62)
    print("  p_snap SWEEP DONE")
    print("=" * 62)


if __name__ == "__main__":
    main()
