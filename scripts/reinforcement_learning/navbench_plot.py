# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# # navbench_plots.py  ───────────────────────────────────────────
# import os, glob, json
# from pathlib import Path
# from typing import List, Sequence, Dict

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # -----------------------------------------------------------
# # utilities
# # -----------------------------------------------------------
# def _collect_csv(combo_pattern: str, results_dir: str = "results") -> pd.DataFrame:
#     """Load every run-CSV matching pattern <results>/<combo_pattern>_run-*.csv"""
#     files = glob.glob(os.path.join(results_dir, f"{combo_pattern}_run-*.csv"))
#     if not files:
#         raise FileNotFoundError(f"No CSVs for pattern {combo_pattern}")

#     return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# # def _agg_mean_std(df: pd.DataFrame, metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
# #     """Return {'metric': {'mean': x, 'std': y}, ... } across runs"""
# #     out = {}
# #     for m in metrics:
# #         out[m] = {
# #             "mean": df[f"{m}_mean"].mean(),
# #             "std":  df[f"{m}_std"].mean() if f"{m}_std" in df else 0.0,
# #         }
#     return out

# def _agg_mean_std(df: pd.DataFrame,
#                   metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
#     """
#     Return {'metric': {'mean': x, 'std': y}, ... }.
#     Accept both  <metric>           and
#                  <metric>_mean/_std columns.
#     """
#     out = {}
#     for m in metrics:
#         col_mean = f"{m}_mean" if f"{m}_mean" in df.columns else m
#         col_std  = f"{m}_std"  if f"{m}_std"  in df.columns else None
#         out[m] = {
#             "mean": df[col_mean].mean(),
#             "std":  df[col_std].mean() if col_std else 0.0,
#         }
#     return out

# def _needs_positive(metric):
#     # metrics that are "smaller is better"
#     return metric in {"final_distance_error",
#                       "heading_error",
#                       "avg_time_to_target",
#                       "control_variation",
#                       "tracking_error"}

# # -----------------------------------------------------------
# # plotting helpers
# # -----------------------------------------------------------
# def plot_radar(combo_patterns: Sequence[str],
#                metrics: Sequence[str],
#                labels: Sequence[str] = None,
#                title: str = "",
#                results_dir: str = "results",
#                save_path: str = None):
#     """
#     combo_patterns e.g. ["FloatingPlatform_GoToPosition_skrl",
#                          "FloatingPlatform_GoToPosition_rl_games"]
#     metrics        e.g. ["success_rate", "final_distance_error", ...]
#     """
#     if labels is None:
#         labels = combo_patterns

#     # gather data (mean across runs)
#     data = []
#     for pattern in combo_patterns:
#         df = _collect_csv(pattern, results_dir)
#         d  = [_agg_mean_std(df, metrics)[m]["mean"] for m in metrics]
#         data.append(d)
#     arr = np.asarray(data)

#     # shape (L, M)
#     for j, m in enumerate(metrics):
#         if _needs_positive(m):
#             arr[:, j] = -arr[:, j]   # invert so higher = better

#     # normalise each metric to [0,1] for nice radar
#     mn, mx = arr.min(axis=0), arr.max(axis=0)
#     norm   = (arr - mn) / (mx - mn + 1e-9)

#     # radar
#     N      = len(metrics)
#     angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
#     fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6,6))

#     for vals, lab in zip(norm, labels):
#         vals = vals.tolist() + [vals[0]]
#         ax.plot(angles, vals, label=lab)
#         ax.fill(angles, vals, alpha=.25)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(metrics, fontsize=10)
#     ax.set_title(title, pad=20)
#     ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1))
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#     plt.show()


# def plot_grouped_bars(combo_patterns: Sequence[str],
#                       metrics: Sequence[str],
#                       labels: Sequence[str] = None,
#                       title: str = "",
#                       results_dir: str = "results",
#                       save_path: str = None):
#     """
#     Draw grouped bar chart with mean ± std error-bars
#     """
#     if labels is None:
#         labels = combo_patterns
#     L, M = len(combo_patterns), len(metrics)
#     means, stds = np.zeros((L, M)), np.zeros((L, M))

#     for i, patt in enumerate(combo_patterns):
#         df = _collect_csv(patt, results_dir)
#         agg = _agg_mean_std(df, metrics)
#         means[i] = [agg[m]["mean"] for m in metrics]
#         stds[i]  = [agg[m]["std"]  for m in metrics]

#     x = np.arange(M)
#     w = 0.8 / L
#     fig, ax = plt.subplots(figsize=(1.8+1.8*M, 4))
#     for i in range(L):
#         ax.bar(x + i*w, means[i], w,
#                yerr=stds[i], capsize=3, label=labels[i])
#     ax.set_xticks(x + w*(L-1)/2)
#     ax.set_xticklabels(metrics, rotation=20, ha="right")
#     ax.set_title(title)
#     ax.legend()
#     plt.tight_layout()
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#     plt.show()


# def plot_violin_goals(combo_patterns: Sequence[str],
#                       results_dir: str = "results",
#                       title: str = "",
#                       save_path: str = None):
#     """
#     Violin plot for # goals reached (GoThroughPositions).
#     """
#     import seaborn as sns
#     data = []
#     for patt in combo_patterns:
#         df = _collect_csv(patt, results_dir)
#         # g  = df["num_goals_reached_mean"].values
#         # col = "num_goals_reached_mean" if "num_goals_reached_mean" in df.columns else "num_goals_reached"
#         col = next((c for c in df.columns if "num_goals_reached" in c), None)
#         if col is None:
#             print(f"Warning: 'num_goals_reached' not in {patt}")
#             continue

#         g = df[col].values

#         lab= patt.split("_")[-1]        # library part
#         data.extend([{"lib": lab, "goals": v} for v in g])

#     df_plot = pd.DataFrame(data)
#     fig, ax = plt.subplots(figsize=(4,4))
#     sns.violinplot(data=df_plot, x="lib", y="goals", ax=ax)
#     ax.set_title(title)
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#     plt.show()


# def plot_ribbon_tracking(combo_patterns: Sequence[str],
#                          results_dir: str = "results",
#                          title: str = "",
#                          save_path: str = None):
#     """
#     Ribbon plot for time-series velocity-tracking error.
#     Requires each run-CSV to include column 'tracking_error_series'
#     saved as JSON list (optional – only if you logged per-timestep error).
#     """
#     fig, ax = plt.subplots(figsize=(6,4))
#     for patt in combo_patterns:
#         dfs = _collect_csv(patt, results_dir)
#         series_list = []
#         for _, row in dfs.iterrows():
#             if "tracking_error_series" in row:
#                 series_list.append(json.loads(row["tracking_error_series"]))
#         if not series_list:
#             continue
#         arr = np.array(series_list)         # [runs, T]
#         mean, std = arr.mean(0), arr.std(0)
#         t = np.arange(len(mean))
#         ax.plot(t, mean, label=patt.split("_")[-1])
#         ax.fill_between(t, mean-std, mean+std, alpha=.3)
#     ax.set_xlabel("timestep"); ax.set_ylabel("vel error")
#     ax.set_title(title)
#     ax.legend()
#     if save_path:
#         Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#     plt.show()
# # ───────────────────────────────────────────────────────────────────────────


import matplotlib.pyplot as plt
import os
from math import pi

import pandas as pd
import seaborn as sns


def _agg_mean_std(df, metrics):
    out = {}
    for m in metrics:
        try:
            out[m] = {
                "mean": df[f"{m}_mean"].mean(),
                "std": df[f"{m}_std"].mean(),
                "ci95": df[f"{m}_ci95"].mean(),
            }
        except KeyError:
            print(f"⚠️ Missing metric: {m} in DataFrame columns.")
    return out


def plot_radar(combo, metrics, libs, title, results_dir="results", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    for lib in libs:
        filename = os.path.join(results_dir, f"{combo}_{lib}_run-0.csv")
        if not os.path.exists(filename):
            print(f"Missing file: {filename}")
            continue
        df = pd.read_csv(filename)
        metric_values = _agg_mean_std(df, metrics)
        d = [metric_values.get(m, {}).get("mean", 0.0) for m in metrics]
        d += d[:1]
        ax.plot(angles, d, label=lib)
        ax.fill(angles, d, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_violin_goals(combo, title, results_dir="results", save_path=None):
    dfs = []
    for lib in ["skrl", "rlgames"]:
        path = os.path.join(results_dir, f"{combo}_{lib}_run-0.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "num_goals_reached_mean" in df.columns:
            df["lib"] = lib
            dfs.append(df)

    if dfs:
        full_df = pd.concat(dfs)
        plt.figure(figsize=(5, 4))
        sns.violinplot(data=full_df, x="lib", y="num_goals_reached_mean")
        plt.title(title)
        plt.ylabel("# Goals Reached")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def plot_ribbon_tracking(combo, title, results_dir="results", save_path=None):
    dfs = []
    for lib in ["skrl", "rlgames"]:
        path = os.path.join(results_dir, f"{combo}_{lib}_run-0.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "tracking_error_mean" in df.columns:
            df["lib"] = lib
            dfs.append(df)

    if dfs:
        full_df = pd.concat(dfs)
        plt.figure(figsize=(5, 4))
        sns.barplot(data=full_df, x="lib", y="tracking_error_mean", ci="sd")
        plt.ylabel("Velocity Tracking Error")
        plt.title(title)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
