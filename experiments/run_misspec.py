"""
Sensor-parameter misspecification experiments (Figures E and F).
================================================================

Figure E (--mode joint):
  Perturb all six sensor parameters jointly with log-normal noise at
  varying eta, measure total variance decomposition of mu_hat_MLE.

Figure F (--mode onebyone):
  Perturb one sensor parameter at a time, same eta grid.

Usage:
    python -m experiments.run_misspec --mode joint
    python -m experiments.run_misspec --mode onebyone
    python -m experiments.run_misspec --mode joint --plot-only
    python -m experiments.run_misspec --n-inner 100 --n-outer 5  # quick
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
    pod_weighted_baseline,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

FIG_DIR = PROJECT_ROOT / "figures"
RES_DIR = PROJECT_ROOT / "results"

ETA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

SENSOR_PARAMS = [
    ("theta_s", cfg.AERIAL_THRESHOLD),
    ("k_s",     cfg.AERIAL_SLOPE),
    ("sigma_s", cfg.AERIAL_SENSOR_SIGMA),
    ("theta_c", cfg.CONT_THRESHOLD),
    ("k_c",     cfg.CONT_SLOPE),
    ("sigma_c", cfg.CONT_SENSOR_SIGMA),
]
PARAM_NAMES = [p[0] for p in SENSOR_PARAMS]
PARAM_TRUE  = {p[0]: p[1] for p in SENSOR_PARAMS}

PARAM_LABELS = {
    "theta_s": r"$\theta_s$",
    "k_s":     r"$k_s$",
    "sigma_s": r"$\sigma_s$",
    "theta_c": r"$\theta_c$",
    "k_c":     r"$k_c$",
    "sigma_c": r"$\sigma_c$",
}


def _build_specs_from_dict(params: dict):
    return build_baseline_specs(
        aerial_threshold=params["theta_s"],
        aerial_slope=params["k_s"],
        aerial_sensor_sigma=params["sigma_s"],
        cont_threshold=params["theta_c"],
        cont_slope=params["k_c"],
        cont_sensor_sigma=params["sigma_c"],
        cont_fp_rate=cfg.CONT_FP_RATE,
        cont_fp_scale=cfg.CONT_FP_SCALE,
    )


def _run_inner(obs, tech_specs_misspec, p_on_grid, p_off_grid):
    """Run MLE with misspecified sensor parameters on pre-simulated data."""
    if obs.n_total_detections() < 2:
        return np.nan
    res = loop_empirical(
        obs, tech_specs_misspec, cfg.T,
        max_gap=cfg.MAX_GAP,
        decision_threshold=cfg.DECISION_THRESHOLD,
        p_on_grid=p_on_grid, p_off_grid=p_off_grid,
        p_off_nudge=cfg.NUDGE)
    return res["mean"]


def run_misspec(mode: str, n_inner: int, n_outer: int, csv_path: Path,
                eta_values=None, seed_offset: int = 0):
    """Run the misspecification experiment.

    mode='joint': perturb all 6 params together
    mode='onebyone': perturb one at a time (6 separate runs per eta)

    eta_values: list of eta values to iterate. Defaults to ETA_VALUES.
    seed_offset: additive offset (in units of 1e6) for case seeds, to keep
                 fresh seeds when extending the grid (avoids collision with
                 already-saved runs).
    """
    if eta_values is None:
        eta_values = ETA_VALUES

    mu_true = cfg.P_ON / (cfg.P_ON + cfg.P_OFF) * cfg.MU_EMIT
    p_on_grid = cfg.build_geom_grid(cfg.P_ON, cfg.P_GEOM_FACTOR, cfg.P_GRID_RES)
    p_off_grid = cfg.build_geom_grid(cfg.P_OFF, cfg.P_GEOM_FACTOR, cfg.P_GRID_RES)

    tech_specs_true = _build_specs_from_dict(PARAM_TRUE)

    if mode == "joint":
        perturb_sets = [("all",)]
    else:
        perturb_sets = [(pname,) for pname in PARAM_NAMES]

    records = []
    case_idx = 0

    for perturb_group in perturb_sets:
        perturb_label = perturb_group[0]

        for eta in eta_values:
            seed_base = cfg.BASE_SEED + (seed_offset + case_idx) * 1_000_000
            case_idx += 1

            print(f"\n{'=' * 62}")
            print(f"  perturb={perturb_label}, eta={eta}")
            print(f"{'=' * 62}")

            within_vars = []
            between_means = []
            pod_vars = []
            t0 = time.time()

            for outer in range(n_outer):
                rng_psi = np.random.default_rng(seed_base + outer * 100_000)

                misspec_params = dict(PARAM_TRUE)
                if eta > 0:
                    if perturb_label == "all":
                        for pname in PARAM_NAMES:
                            log_true = np.log(PARAM_TRUE[pname])
                            log_perturbed = rng_psi.normal(log_true, eta)
                            misspec_params[pname] = np.exp(log_perturbed)
                    else:
                        pname = perturb_label
                        log_true = np.log(PARAM_TRUE[pname])
                        log_perturbed = rng_psi.normal(log_true, eta)
                        misspec_params[pname] = np.exp(log_perturbed)

                tech_specs_misspec = _build_specs_from_dict(misspec_params)

                mu_mle_inner = []
                mu_pod_inner = []

                for inner in range(n_inner):
                    rep = outer * n_inner + inner
                    mask_rng = np.random.default_rng(seed_base + rep)
                    emit_rng = np.random.default_rng(seed_base + rep + 500_000)

                    snap_mask, cont_mask = generate_masks(
                        cfg.T, cfg.P_SNAP, cfg.P_CONT, cfg.T_CONT, mask_rng)
                    obs, _st, _sz, _eid = simulate_series(
                        T=cfg.T, p_on=cfg.P_ON, p_off=cfg.P_OFF,
                        size_mu=cfg.SIZE_MU, size_sigma=cfg.SIZE_SIGMA,
                        tech_specs=tech_specs_true,
                        snap_mask=snap_mask, cont_mask=cont_mask,
                        rng=emit_rng)

                    mu_hat = _run_inner(obs, tech_specs_misspec,
                                       p_on_grid, p_off_grid)
                    mu_mle_inner.append(mu_hat)

                    pod_hat = pod_weighted_baseline(obs, tech_specs_true)
                    mu_pod_inner.append(pod_hat)

                arr = np.array(mu_mle_inner)
                valid = arr[np.isfinite(arr)]
                if len(valid) > 1:
                    within_vars.append(float(np.var(valid, ddof=0)))
                    between_means.append(float(np.mean(valid)))
                else:
                    within_vars.append(np.nan)
                    between_means.append(np.nan)

                pod_arr = np.array(mu_pod_inner)
                pod_valid = pod_arr[np.isfinite(pod_arr)]
                pod_vars.append(float(np.var(pod_valid, ddof=0))
                                if len(pod_valid) > 1 else np.nan)

                el = time.time() - t0
                eta_str = f"eta={eta:.2f}"
                print(f"    outer {outer+1:>3d}/{n_outer}  "
                      f"elapsed={el:.0f}s  "
                      f"mean_mu={between_means[-1]:.3f}  "
                      f"within_var={within_vars[-1]:.2f}", flush=True)

            wv = np.array(within_vars)
            bm = np.array(between_means)
            pv = np.array(pod_vars)
            wv_valid = wv[np.isfinite(wv)]
            bm_valid = bm[np.isfinite(bm)]
            pv_valid = pv[np.isfinite(pv)]

            E_within = float(np.mean(wv_valid)) if len(wv_valid) else np.nan
            Var_between = (float(np.var(bm_valid, ddof=0))
                           if len(bm_valid) > 1 else np.nan)
            total_var = E_within + Var_between if (
                np.isfinite(E_within) and np.isfinite(Var_between)) else np.nan
            bias = (float(np.mean(bm_valid)) - mu_true
                    if len(bm_valid) else np.nan)
            pod_var_mean = float(np.mean(pv_valid)) if len(pv_valid) else np.nan

            n_valid = len(wv_valid)
            within_se = (float(np.std(wv_valid, ddof=1) / np.sqrt(n_valid))
                         if n_valid > 1 else np.nan)
            pod_var_se = (float(np.std(pv_valid, ddof=1) / np.sqrt(len(pv_valid)))
                          if len(pv_valid) > 1 else np.nan)
            # SE of total via jackknife over outer batches
            if n_valid > 2:
                jack_totals = []
                for j in range(n_valid):
                    wv_j = np.delete(wv_valid, j)
                    bm_j = np.delete(bm_valid, j)
                    jack_totals.append(float(np.mean(wv_j) + np.var(bm_j, ddof=0)))
                jack_arr = np.array(jack_totals)
                total_se = float(np.sqrt((n_valid - 1) / n_valid
                                         * np.sum((jack_arr - np.mean(jack_arr))**2)))
            else:
                total_se = np.nan

            records.append(dict(
                perturb=perturb_label,
                eta=eta,
                within_var=E_within,
                within_var_se=within_se,
                between_var=Var_between,
                total_var=total_var,
                total_var_se=total_se,
                bias=bias,
                pod_var=pod_var_mean,
                pod_var_se=pod_var_se,
                mu_true=mu_true,
                n_outer=n_outer,
                n_inner=n_inner,
            ))

            print(f"    => within={E_within:.3f}  between={Var_between:.3f}  "
                  f"total={total_var:.3f}  bias={bias:+.3f}  "
                  f"pod_var={pod_var_mean:.3f}")

    df = pd.DataFrame(records)
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        merged = pd.concat([prev, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["perturb", "eta"], keep="last")
        merged = merged.sort_values(["perturb", "eta"]).reset_index(drop=True)
        merged.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\n  Appended {len(df)} new row(s) to: {csv_path}")
        df = merged
    else:
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\n  Results saved to: {csv_path}")
    return df


def figure_misspec_joint(df, out):
    """Figure E: single panel — variance decomposition vs eta."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.linewidth": 2,
    })

    dfa = df[df["perturb"] == "all"]
    eta = dfa["eta"].values

    fig, ax = plt.subplots(figsize=(6, 4.5))

    z = 1.96
    ax.plot(eta, dfa["within_var"], "o-", color="#2ECC71",
            label="Within (sampling)")
    if "within_var_se" in dfa.columns:
        se = dfa["within_var_se"].values
        ax.fill_between(eta, dfa["within_var"] - z * se,
                        dfa["within_var"] + z * se,
                        alpha=0.15, color="#2ECC71")
    ax.plot(eta, dfa["between_var"], "s--", color="#E67E22",
            label="Between (misspec)")
    ax.plot(eta, dfa["total_var"], "D-", color="#2C3E50",
            linewidth=2.5, label="Total")
    if "total_var_se" in dfa.columns:
        se = dfa["total_var_se"].values
        ax.fill_between(eta, dfa["total_var"] - z * se,
                        dfa["total_var"] + z * se,
                        alpha=0.15, color="#2C3E50")
    pod_ref = dfa["pod_var"].values[0]
    pod_se = (dfa["pod_var_se"].values[0]
              if "pod_var_se" in dfa.columns else 0.0)
    ax.axhline(pod_ref, color="#C39BD3", ls="--", lw=1.5,
               label="POD-weighted (correct spec.)")
    if pod_se > 0:
        ax.axhspan(pod_ref - z * pod_se, pod_ref + z * pod_se,
                   alpha=0.10, color="#C39BD3")

    crossover_eta = np.nan
    for i in range(len(eta) - 1):
        t_i = dfa["total_var"].values[i]
        t_j = dfa["total_var"].values[i + 1]
        if t_i <= pod_ref <= t_j:
            frac = (pod_ref - t_i) / (t_j - t_i)
            crossover_eta = eta[i] + frac * (eta[i + 1] - eta[i])
            break

    if np.isfinite(crossover_eta):
        ax.axvline(crossover_eta, color="gray", ls=":", lw=1)
        ax.annotate(f"$\\eta^* = {crossover_eta:.2f}$",
                    xy=(crossover_eta, pod_ref),
                    xytext=(crossover_eta + 0.03, pod_ref * 1.1),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(r"Misspecification level $\eta$")
    ax.set_ylabel(r"Variance of $\hat\mu$ (kg$^2$/h$^2$)")
    ax.set_title("Variance decomposition under sensor misspecification")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, ls=":", lw=0.5)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    eta_star_path = out.with_name("misspec_eta_star.txt")
    eta_star_path.write_text(
        f"eta_star = {crossover_eta:.4f}\n"
        f"pod_var_correct = {pod_ref:.4f}\n")
    print(f"  eta* = {crossover_eta:.4f}  "
          f"(POD-weighted var = {pod_ref:.2f})")
    return crossover_eta


def _bias_se(row, n_outer):
    """Derive SE(bias) from the stored between-variance.

    bias = mean(per-outer means) - mu_true, so SE(bias) is the SE of the
    mean of n_outer per-outer means. Using ddof=0 between_var, the
    unbiased SE is sqrt(between_var / (n_outer - 1)).
    """
    bv = row.get("between_var", np.nan)
    if not np.isfinite(bv) or n_outer < 2:
        return np.nan
    return float(np.sqrt(bv / (n_outer - 1)))


def _mse_with_se(df):
    """Append MSE and MSE_SE columns to a copy of df.

    MSE = bias^2 + total_var
    SE(MSE) ~ sqrt( (2|bias| SE(bias))^2 + SE(total_var)^2 )
    """
    out = df.copy()
    out["bias_sq"] = out["bias"] ** 2
    out["mse"] = out["bias_sq"] + out["total_var"]

    n_outer = out["n_outer"].iloc[0] if "n_outer" in out.columns else 25
    bias_se = out.apply(lambda r: _bias_se(r, n_outer), axis=1)
    out["bias_se"] = bias_se
    out["bias_sq_se"] = 2.0 * out["bias"].abs() * bias_se
    tot_se = out["total_var_se"] if "total_var_se" in out.columns else 0.0
    out["mse_se"] = np.sqrt(out["bias_sq_se"] ** 2 + tot_se ** 2)
    return out


def figure_misspec_joint_mse(df, out):
    """Figure E (MSE version): MSE decomposition vs eta.

    Plots:  bias^2, within (sampling), between (misspec), total MSE
            and POD-weighted MSE reference (~= POD variance, since
            POD-weighted is approximately unbiased at baseline).
    """
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.linewidth": 2,
    })

    dfa = _mse_with_se(df[df["perturb"] == "all"]).reset_index(drop=True)
    eta = dfa["eta"].values
    z = 1.96

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(eta, dfa["within_var"], "o-", color="#2ECC71",
            label="Within (sampling)")
    if "within_var_se" in dfa.columns:
        se = dfa["within_var_se"].values
        ax.fill_between(eta, dfa["within_var"] - z * se,
                        dfa["within_var"] + z * se,
                        alpha=0.15, color="#2ECC71")
    ax.plot(eta, dfa["between_var"], "s--", color="#E67E22",
            label="Between (misspec)")
    ax.plot(eta, dfa["bias_sq"], "^:", color="#8E44AD",
            label=r"Bias$^2$")
    if "bias_sq_se" in dfa.columns:
        se = dfa["bias_sq_se"].values
        ax.fill_between(eta, np.maximum(0, dfa["bias_sq"] - z * se),
                        dfa["bias_sq"] + z * se,
                        alpha=0.15, color="#8E44AD")
    ax.plot(eta, dfa["mse"], "D-", color="#2C3E50",
            linewidth=2.5, label="Total MSE")
    if "mse_se" in dfa.columns:
        se = dfa["mse_se"].values
        ax.fill_between(eta, dfa["mse"] - z * se,
                        dfa["mse"] + z * se,
                        alpha=0.15, color="#2C3E50")

    pod_ref = dfa["pod_var"].values[0]
    pod_se = (dfa["pod_var_se"].values[0]
              if "pod_var_se" in dfa.columns else 0.0)
    ax.axhline(pod_ref, color="#C39BD3", ls="--", lw=1.5,
               label="POD-weighted (correct spec.)")
    if pod_se > 0:
        ax.axhspan(pod_ref - z * pod_se, pod_ref + z * pod_se,
                   alpha=0.10, color="#C39BD3")

    crossover_eta = np.nan
    for i in range(len(eta) - 1):
        m_i = dfa["mse"].values[i]
        m_j = dfa["mse"].values[i + 1]
        if m_i <= pod_ref <= m_j:
            frac = (pod_ref - m_i) / (m_j - m_i)
            crossover_eta = eta[i] + frac * (eta[i + 1] - eta[i])
            break
    if np.isfinite(crossover_eta):
        ax.axvline(crossover_eta, color="gray", ls=":", lw=1)
        ax.annotate(f"$\\eta^* = {crossover_eta:.2f}$",
                    xy=(crossover_eta, pod_ref),
                    xytext=(crossover_eta + 0.03, pod_ref * 1.1),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(r"Misspecification level $\eta$")
    ax.set_ylabel(r"MSE of $\hat\mu$ (kg$^2$/h$^2$)")
    ax.set_title("MSE decomposition under sensor misspecification")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, ls=":", lw=0.5)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return crossover_eta


def figure_misspec_onebyone_mse(df, out):
    """Figure F (MSE version): MSE per parameter."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "lines.linewidth": 1.5,
    })

    params = [p for p in PARAM_NAMES]
    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 3.5),
                             squeeze=False)

    for col, pname in enumerate(params):
        dfp = _mse_with_se(df[df["perturb"] == pname]).reset_index(drop=True)
        eta = dfp["eta"].values
        pod_ref = dfp["pod_var"].values[0]
        z = 1.96

        ax = axes[0, col]
        ax.plot(eta, dfp["mse"], "o-", color="#2C3E50",
                linewidth=2, label="MLE MSE")
        if "mse_se" in dfp.columns:
            se = dfp["mse_se"].values
            ax.fill_between(eta, dfp["mse"] - z * se,
                            dfp["mse"] + z * se,
                            alpha=0.15, color="#2C3E50")
        ax.axhline(pod_ref, color="#C39BD3", ls="--", lw=1.5,
                   label="POD (correct)")
        if "pod_var_se" in dfp.columns:
            pod_se = dfp["pod_var_se"].values[0]
            if pod_se > 0:
                ax.axhspan(pod_ref - z * pod_se, pod_ref + z * pod_se,
                           alpha=0.10, color="#C39BD3")
        ax.set_title(f"({chr(97 + col)}) {PARAM_LABELS[pname]}")
        ax.set_xlabel(r"$\eta$")
        if col == 0:
            ax.set_ylabel(r"MSE of $\hat\mu$")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, ls=":", lw=0.5)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def figure_misspec_joint_bias(df, out):
    """Joint case — bias of mu_hat vs eta with 95% CI."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.linewidth": 2,
    })
    dfa = _mse_with_se(df[df["perturb"] == "all"]).reset_index(drop=True)
    eta = dfa["eta"].values
    z = 1.96

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.plot(eta, dfa["bias"], "o-", color="#8E44AD",
            linewidth=2, label=r"Bias of $\hat\mu_{\mathrm{MLE}}$")
    if "bias_se" in dfa.columns:
        se = dfa["bias_se"].values
        ax.fill_between(eta, dfa["bias"] - z * se, dfa["bias"] + z * se,
                        alpha=0.20, color="#8E44AD")
    ax.set_xlabel(r"Misspecification level $\eta$")
    ax.set_ylabel(r"Bias of $\hat\mu$ (kg/h)")
    ax.set_title("MLE bias under joint sensor misspecification")
    ax.grid(alpha=0.3, ls=":", lw=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def figure_misspec_onebyone_bias(df, out):
    """Per-parameter case — bias of mu_hat vs eta, one panel per param."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "lines.linewidth": 1.5,
    })

    params = [p for p in PARAM_NAMES]
    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 3.0),
                             squeeze=False, sharey=True)

    for col, pname in enumerate(params):
        dfp = _mse_with_se(df[df["perturb"] == pname]).reset_index(drop=True)
        eta = dfp["eta"].values
        z = 1.96

        ax = axes[0, col]
        ax.axhline(0.0, color="gray", ls="--", lw=1)
        ax.plot(eta, dfp["bias"], "o-", color="#8E44AD",
                linewidth=2, label="Bias")
        if "bias_se" in dfp.columns:
            se = dfp["bias_se"].values
            ax.fill_between(eta, dfp["bias"] - z * se,
                            dfp["bias"] + z * se,
                            alpha=0.20, color="#8E44AD")
        ax.set_title(f"({chr(97 + col)}) {PARAM_LABELS[pname]}")
        ax.set_xlabel(r"$\eta$")
        if col == 0:
            ax.set_ylabel(r"Bias of $\hat\mu$ (kg/h)")
        ax.grid(alpha=0.3, ls=":", lw=0.5)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def figure_misspec_single_param_mse(df, out, pname: str = "theta_s"):
    """Single-panel MSE figure for one perturbed sensor parameter."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.linewidth": 2,
    })

    dfp = _mse_with_se(df[df["perturb"] == pname]).reset_index(drop=True)
    if len(dfp) == 0:
        raise ValueError(f"No rows for perturb='{pname}' in dataframe")
    eta = dfp["eta"].values
    z = 1.96

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(eta, dfp["mse"], "o-", color="#2C3E50",
            linewidth=2.5, label=r"MLE (misspecified $\theta_s$)")
    if "mse_se" in dfp.columns:
        se = dfp["mse_se"].values
        ax.fill_between(eta, dfp["mse"] - z * se,
                        dfp["mse"] + z * se,
                        alpha=0.18, color="#2C3E50")

    pod_ref = dfp["pod_var"].values[0]
    pod_se = (dfp["pod_var_se"].values[0]
              if "pod_var_se" in dfp.columns else 0.0)
    ax.axhline(pod_ref, color="#C39BD3", ls="--", lw=1.8,
               label="POD-weighted baseline")
    if pod_se > 0:
        ax.axhspan(pod_ref - z * pod_se, pod_ref + z * pod_se,
                   alpha=0.12, color="#C39BD3")

    ax.set_xlabel(r"Misspecification level $\eta$ (log-normal SD)")
    ax.set_ylabel(r"MSE of $\hat\mu$ (kg$^2$/h$^2$)")
    ax.set_title(
        r"MSE of $\hat\mu_{\mathrm{MLE}}$ vs misspecification of $\theta_s$"
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, ls=":", lw=0.5)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="gray")

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def figure_misspec_onebyone(df, out):
    """Figure F: single row, 6 columns — total variance per parameter."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "lines.linewidth": 1.5,
    })

    params = [p for p in PARAM_NAMES]
    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 3.5),
                             squeeze=False)

    for col, pname in enumerate(params):
        dfp = df[df["perturb"] == pname]
        eta = dfp["eta"].values
        pod_ref = dfp["pod_var"].values[0]

        ax = axes[0, col]
        ax.plot(eta, dfp["total_var"], "o-", color="#2C3E50",
                linewidth=2, label="MLE total")
        if "total_var_se" in dfp.columns:
            z = 1.96
            se = dfp["total_var_se"].values
            ax.fill_between(eta, dfp["total_var"] - z * se,
                            dfp["total_var"] + z * se,
                            alpha=0.15, color="#2C3E50")
        ax.axhline(pod_ref, color="#C39BD3", ls="--", lw=1.5,
                   label="POD (correct)")
        if "pod_var_se" in dfp.columns:
            pod_se = dfp["pod_var_se"].values[0]
            if pod_se > 0:
                ax.axhspan(pod_ref - 1.96 * pod_se, pod_ref + 1.96 * pod_se,
                           alpha=0.10, color="#C39BD3")
        ax.set_title(f"({chr(97 + col)}) {PARAM_LABELS[pname]}")
        ax.set_xlabel(r"$\eta$")
        if col == 0:
            ax.set_ylabel(r"Variance of $\hat\mu$")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, ls=":", lw=0.5)
        if col == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["joint", "onebyone"], required=True)
    parser.add_argument("--n-inner", type=int, default=500)
    parser.add_argument("--n-outer", type=int, default=25)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--fig-dir", type=str, default=None)
    parser.add_argument("--res-dir", type=str, default=None)
    parser.add_argument("--extra-eta", type=float, nargs="+", default=None,
                        help="Additional eta values to run; results are "
                             "appended to the existing CSV without "
                             "overwriting prior eta values.")
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    res_dir = Path(args.res_dir) if args.res_dir else RES_DIR
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = res_dir / f"misspec_{args.mode}.csv"

    if args.plot_only:
        df = pd.read_csv(csv_path)
    elif args.extra_eta is not None:
        # Skip etas already in the saved CSV
        new_eta = list(args.extra_eta)
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            present = set(round(float(e), 6) for e in existing["eta"].unique())
            new_eta = [e for e in new_eta
                       if round(float(e), 6) not in present]
        if not new_eta:
            print("  All requested extra etas already present; nothing to do.")
            df = pd.read_csv(csv_path)
        else:
            print(f"  Extending eta grid with: {new_eta}")
            # seed_offset = 1000 ensures seeds don't collide with existing runs
            df = run_misspec(args.mode, args.n_inner, args.n_outer, csv_path,
                             eta_values=new_eta, seed_offset=1000)
    else:
        df = run_misspec(args.mode, args.n_inner, args.n_outer, csv_path)

    if args.mode == "joint":
        figure_misspec_joint(df, fig_dir / "figure_misspec.pdf")
        figure_misspec_joint_mse(df, fig_dir / "figure_misspec_mse.pdf")
        figure_misspec_joint_bias(df, fig_dir / "figure_misspec_bias.pdf")
    else:
        figure_misspec_onebyone(df, fig_dir / "figure_misspec_onebyone.pdf")
        figure_misspec_onebyone_mse(
            df, fig_dir / "figure_misspec_onebyone_mse.pdf")
        figure_misspec_onebyone_bias(
            df, fig_dir / "figure_misspec_onebyone_bias.pdf")
        figure_misspec_single_param_mse(
            df, fig_dir / "figure_misspec_thetas_mse.pdf",
            pname="theta_s")

    print(f"\n  MISSPEC ({args.mode}) DONE")


if __name__ == "__main__":
    main()
