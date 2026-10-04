"""
Extended sweep composite figures.

Matches the style of `make_figure_sweeps.py` (only MLE and POD-weighted,
95% CI bands), but supports:

  --mode main3   3 sweeps  (p_snap, theta_snap, tau_emit)            -> figure4
  --mode all5    5 sweeps  (p_snap, theta_snap, tau_emit, p_cont,    -> figure4_all
                            T_cont) in a 4-row x 3-col block layout
  --mode cont2   2 sweeps  (p_cont, T_cont) in 2x2 layout            -> figure4_cont
"""
from __future__ import annotations

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "methane-mpl-cache"))

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import plot_style as ps


SWEEP_META = {
    "p_snap": {
        "file": "sweep_psnap.csv",
        "dir": "results",
        "value_col": "p_snap",
        "filter_T1000": True,
        "label": r"$p_{\mathrm{snap}}$",
        "title": "Snapshot frequency",
        "keep_vals": None,
        "xscale": "log",
        "dual_axis": True,
        "dual_axis_label": "Avg. hours between snapshots",
    },
    "theta_snap": {
        "file": "sweep_theta_snap_summary.csv",
        "dir": "results_linear",
        "value_col": "sweep_val",
        "label": r"$\theta_{\mathrm{snap}}$ (kg/h)",
        "title": "Snapshot POD threshold",
        "keep_vals": [50, 100, 200, 300, 500],
        "xscale": "linear",
        "dual_axis": False,
    },
    "tau_emit": {
        "file": "sweep_tau_emit_summary.csv",
        "dir": "results_linear",
        "value_col": "sweep_val",
        "label": r"$\tau_{\mathrm{emit}}$ (hours)",
        "title": "Emission duration",
        "keep_vals": [1.0, 2.7, 7.2, 20.0, 53.6, 100.0],
        "xscale": "linear",
        "dual_axis": False,
    },
    "p_cont": {
        "file": "sweep_p_cont_summary.csv",
        "dir": "results_linear",
        "value_col": "sweep_val",
        "label": r"$p_{\mathrm{cont}}$",
        "title": "Continuous monitoring frequency",
        "keep_vals": None,
        "xscale": "linear",
        "dual_axis": False,
    },
    "T_cont": {
        "file": "sweep_T_cont_summary.csv",
        "dir": "results_linear",
        "value_col": "sweep_val",
        "label": r"$T_{\mathrm{cont}}$ (hours)",
        "title": "Continuous window duration",
        "keep_vals": None,
        "xscale": "linear",
        "dual_axis": False,
    },
}

C_POD = ps.POD
C_MLE = ps.MLE
C_RATIO = ps.RATIO

PANEL_LETTERS = "abcdefghijklmnop"


def load_sweep(base_dir: Path, name: str):
    """Load var/ratio data for one sweep, return (vals, pod_v, pod_se,
    mle_v, mle_se, ratios, ratio_se)."""
    meta = SWEEP_META[name]
    csv_path = base_dir / meta["dir"] / meta["file"]
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if meta.get("filter_T1000"):
        df = df[df["T"] == 1000].copy()

    val_col = meta["value_col"]
    all_vals = sorted(df[val_col].unique())
    keep = meta.get("keep_vals")
    if keep is not None:
        keep_set = set(keep)
        vals = [v for v in all_vals
                if any(abs(v - k) < 0.05 * max(abs(k), 0.01)
                       for k in keep_set)]
    else:
        vals = all_vals

    pod_v, pod_se = [], []
    mle_v, mle_se = [], []
    ratios, ratio_se = [], []
    for v in vals:
        for method, vlist, slist in [("pod", pod_v, pod_se),
                                     ("mle", mle_v, mle_se)]:
            r = df[(df[val_col] == v) & (df["method"] == method)]
            vlist.append(r["variance"].values[0] if len(r) else np.nan)
            slist.append(r["variance_se"].values[0] if len(r) else np.nan)
        vp, sp = pod_v[-1], pod_se[-1]
        vm, sm = mle_v[-1], mle_se[-1]
        if vm > 0 and vp > 0:
            rat = vp / vm
            rat_se_v = rat * np.sqrt((sp / vp) ** 2 + (sm / vm) ** 2)
        else:
            rat, rat_se_v = np.nan, np.nan
        ratios.append(rat)
        ratio_se.append(rat_se_v)

    return (np.asarray(vals, dtype=float),
            np.asarray(pod_v), np.asarray(pod_se),
            np.asarray(mle_v), np.asarray(mle_se),
            np.asarray(ratios), np.asarray(ratio_se))


def plot_sweep_pair(ax_var, ax_rat, name, data, panel_idx,
                    show_ylabel: bool, show_legend: bool):
    """Render one sweep into the given var/ratio axes pair."""
    meta = SWEEP_META[name]
    vals, pod_v, pod_se, mle_v, mle_se, ratios, ratio_se = data

    for v, se, col, lab in [
        (pod_v, pod_se, C_POD, "POD-weighted"),
        (mle_v, mle_se, C_MLE, "MLE"),
    ]:
        ax_var.plot(vals, v, "o-", color=col, label=lab,
                    markersize=5, markeredgewidth=1, markeredgecolor="white")
        ax_var.fill_between(vals, v - 1.96 * se, v + 1.96 * se,
                            alpha=0.15, color=col, linewidth=0)

    title = f"({PANEL_LETTERS[panel_idx]}) {meta['title']}"
    ax_var.set_title(title)
    ax_var.grid(alpha=0.3, ls=":", lw=0.5)
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    if meta.get("xscale") == "log":
        ax_var.set_xscale("log")
    if meta.get("dual_axis"):
        ax_top = ax_var.secondary_xaxis(
            "top", functions=(lambda ps: 1.0 / ps, lambda h: 1.0 / h))
        ax_top.set_xlabel(meta.get("dual_axis_label",
                                   "Avg. hours between obs."), fontsize=7)
        if meta.get("xscale") == "log":
            ax_top.set_xscale("log")
        ax_var.spines["top"].set_visible(True)
    if show_ylabel:
        ax_var.set_ylabel(r"Variance of $\hat\mu$")
    if show_legend:
        ax_var.legend(loc="best", framealpha=0.9, edgecolor="gray")

    ax_rat.plot(vals, ratios, "o-", color=C_RATIO, linewidth=2.5,
                markersize=6, markeredgewidth=1.5, markeredgecolor="white",
                label="POD / MLE")
    ax_rat.fill_between(vals, ratios - 1.96 * ratio_se,
                        ratios + 1.96 * ratio_se,
                        alpha=0.15, color=C_RATIO, linewidth=0)
    ax_rat.axhline(1, color="gray", ls="--", lw=1.5, alpha=0.8)
    ax_rat.set_xlabel(meta["label"])
    ax_rat.grid(alpha=0.3, ls=":", lw=0.5)
    ax_rat.spines["top"].set_visible(False)
    ax_rat.spines["right"].set_visible(False)
    if meta.get("xscale") == "log":
        ax_rat.set_xscale("log")
    if show_ylabel:
        ax_rat.set_ylabel("Variance ratio\n(POD / MLE)")


def build_main3(base_dir: Path, out_path: Path):
    sweeps = ["p_snap", "theta_snap", "tau_emit"]
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    panel = 0
    for col, name in enumerate(sweeps):
        data = load_sweep(base_dir, name)
        if data is None:
            print(f"  WARNING: missing data for {name}")
            continue
        plot_sweep_pair(axes[0, col], axes[1, col], name, data,
                        panel_idx=panel,
                        show_ylabel=(col == 0),
                        show_legend=(col == 0))
        panel += 1
    plt.tight_layout()
    _save(fig, out_path)


def build_all5(base_dir: Path, out_path: Path):
    """4 rows x 3 cols.

    Rows 0/1: variance/ratio for (p_snap, theta_snap, tau_emit)
    Rows 2/3: variance/ratio for (p_cont, T_cont, empty)
    """
    fig = plt.figure(figsize=(7, 9.5))
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1, 1, 1, 1],
                           hspace=0.55, wspace=0.40)
    block_top = ["p_snap", "theta_snap", "tau_emit"]
    block_bot = ["p_cont", "T_cont"]
    panel = 0
    for col, name in enumerate(block_top):
        data = load_sweep(base_dir, name)
        if data is None:
            print(f"  WARNING: missing data for {name}")
            continue
        ax_v = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        plot_sweep_pair(ax_v, ax_r, name, data,
                        panel_idx=panel,
                        show_ylabel=(col == 0),
                        show_legend=(col == 0))
        panel += 1
    for col, name in enumerate(block_bot):
        data = load_sweep(base_dir, name)
        if data is None:
            print(f"  WARNING: missing data for {name}")
            continue
        ax_v = fig.add_subplot(gs[2, col])
        ax_r = fig.add_subplot(gs[3, col])
        plot_sweep_pair(ax_v, ax_r, name, data,
                        panel_idx=panel,
                        show_ylabel=(col == 0),
                        show_legend=(col == 0))
        panel += 1
    _save(fig, out_path)


def build_cont2(base_dir: Path, out_path: Path):
    sweeps = ["p_cont", "T_cont"]
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 5))
    panel = 0
    for col, name in enumerate(sweeps):
        data = load_sweep(base_dir, name)
        if data is None:
            print(f"  WARNING: missing data for {name}")
            continue
        plot_sweep_pair(axes[0, col], axes[1, col], name, data,
                        panel_idx=panel,
                        show_ylabel=(col == 0),
                        show_legend=(col == 0))
        panel += 1
    plt.tight_layout()
    _save(fig, out_path)


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", required=True,
                        help="Run root containing results/ and results_linear/")
    parser.add_argument("--out", required=True, help="Output PDF path")
    parser.add_argument("--mode", required=True,
                        choices=["main3", "all5", "cont2"])
    args = parser.parse_args()

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
        "lines.linewidth": 2,
    })

    base_dir = Path(args.res_dir)
    out_path = Path(args.out)
    if args.mode == "main3":
        build_main3(base_dir, out_path)
    elif args.mode == "all5":
        build_all5(base_dir, out_path)
    elif args.mode == "cont2":
        build_cont2(base_dir, out_path)


if __name__ == "__main__":
    main()
