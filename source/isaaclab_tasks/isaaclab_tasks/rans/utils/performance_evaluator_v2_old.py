# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# performance_evaluator_v2.py
import numpy as np
import os
from pathlib import Path
from typing import Any

import pandas as pd

# ---- helper ---------------------------------------------------------------


def _control_variation(actions: np.ndarray) -> float:
    # actions: [T, N, A]
    return np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=-1))


def _heading_err(cos_val: np.ndarray, sin_val: np.ndarray) -> np.ndarray:
    # returns |angle| in rad
    return np.abs(np.arctan2(sin_val, cos_val))


# ---- main evaluator -------------------------------------------------------


class PerformanceEvaluatorV2:
    """
    Compute reviewer-requested metrics + write a run-level CSV that later
    feeds the master “Evaluation Metrics CSV Template”.
    """

    THRESH_DIST = 0.20  # 20 cm → goal considered reached
    RESULTS_DIR = "results"  # where run-level CSVs are stored

    def __init__(
        self,
        task_name: str,
        robot_name: str,
        rl_lib: str,
        episode_data: dict[str, np.ndarray],
        max_horizon: int,
        combo_id: str,
        seed: int = 0,
    ):
        self.task = task_name
        self.robot = robot_name
        self.lib = rl_lib
        self.data = episode_data
        self.T = max_horizon
        self.combo = combo_id  # e.g. FloatingPlatform_GoToPose_skrl
        self.seed = seed
        self.res: dict[str, Any] = {}

    # ---------------------------------------------------------------------

    def _basic(self):
        acts = self.data["act"]  # [T, N, A]
        obs = self.data["obs"]  # [T, N, D]

        # control variation
        self.res["control_variation"] = _control_variation(acts)

        # success mask by distance
        final_d = obs[-1, :, 0]
        self.res["success_rate"] = np.mean(final_d < self.THRESH_DIST)

        # final distance error (m)
        self.res["final_distance_error"] = np.mean(final_d)

        # avg time-to-target
        reach_mask = obs[:, :, 0] < self.THRESH_DIST
        first_hit = reach_mask.argmax(axis=0)  # [N]
        hit_any = reach_mask.any(axis=0)
        times = np.where(hit_any, first_hit, self.T)
        self.res["avg_time_to_target"] = np.mean(times)

    # ---------------------------------------------------------------------

    def _goto_pose(self, obs):
        # heading error uses last step
        cos_d, sin_d = obs[-1, :, 1], obs[-1, :, 2]
        self.res["heading_error"] = np.mean(_heading_err(cos_d, sin_d))

    def _go_through_positions(self, obs):
        # number of way-points hit
        goals_hit = np.sum(obs[:, :, 0] < self.THRESH_DIST, axis=0)
        self.res["num_goals_reached"] = np.mean(goals_hit)

    def _track_velocities(self, obs):
        # velocity error stored in first 3 dims
        vel_err = np.linalg.norm(obs[:, :, :3], axis=-1)  # [T,N]
        self.res["tracking_error"] = np.mean(vel_err)

    # ---------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        obs = self.data["obs"]
        self._basic()

        if self.task == "GoToPose":
            self._goto_pose(obs)
        elif self.task == "GoThroughPositions":
            self._go_through_positions(obs)
        elif self.task == "TrackVelocities":
            self._track_velocities(obs)

        return self.res

    # ---------------------------------------------------------------------

    def export_timeseries_metrics(self, path="timeseries"):
        """
        Export per-timestep, per-env values to long-format CSV
        Columns: timestep, env_id, metric, value, robot, task, rl_lib, seed
        """
        Path(path).mkdir(exist_ok=True)
        obs = self.data["obs"]  # [T, N, D]
        T, N, _ = obs.shape

        rows = []

        for t in range(T):
            for n in range(N):
                dist = obs[t, n, 0]
                rows.append([t, n, "distance_error", dist])

                if self.task == "GoToPose":
                    cos_d, sin_d = obs[t, n, 1], obs[t, n, 2]
                    heading = np.abs(np.arctan2(sin_d, cos_d))
                    rows.append([t, n, "heading_error", heading])

                if self.task == "TrackVelocities":
                    v = np.linalg.norm(obs[t, n, :3])
                    rows.append([t, n, "linear_velocity_error", v])

                if self.task == "GoThroughPositions":
                    hit = float(obs[t, n, 0] < self.THRESH_DIST)
                    rows.append([t, n, "cumulative_goals", hit])

        # Fix cumulative sum and avoid chained assignment
        df = pd.DataFrame(rows, columns=["timestep", "env_id", "metric", "value"])
        if self.task == "GoThroughPositions":
            df = df.copy()
            mask = df["metric"] == "cumulative_goals"
            df.loc[mask, "value"] = df[mask].groupby("env_id")["value"].cumsum()

        # Add context columns and write CSV
        df["robot"] = self.robot
        df["task"] = self.task
        df["rl_lib"] = self.lib
        df["seed"] = self.seed
        out_path = Path(path) / f"{self.combo}_run-{self.seed}.csv"
        df.to_csv(out_path, index=False)
        print(f"[Timeseries] saved {out_path.name}")

    # ---------------------------------------------------------------------

    def save_csv(self):
        """
        Save one CSV per run:
        <results>/<combo>_run-<seed>.csv
        Columns: metric, mean, std
        """
        Path(self.RESULTS_DIR).mkdir(exist_ok=True)
        obs = self.data["obs"]
        acts = self.data["act"]

        # --- compute per-env values for std ---
        # copy mean results, then add std_* counterparts
        stds = {}

        # helper lambdas
        def get_mask(o):
            return o < self.THRESH_DIST

        if "control_variation" in self.res:
            stds["control_variation_std"] = np.std(np.linalg.norm(np.diff(acts, axis=0), axis=-1))

        if "final_distance_error" in self.res:
            stds["final_distance_error_std"] = np.std(obs[-1, :, 0])

        if "avg_time_to_target" in self.res:
            reach_mask = get_mask(obs[:, :, 0])
            first_hit = reach_mask.argmax(axis=0)
            hit_any = reach_mask.any(axis=0)
            times = np.where(hit_any, first_hit, self.T)
            stds["avg_time_to_target_std"] = np.std(times)

        if self.task == "GoToPose":
            stds["heading_error_std"] = np.std(_heading_err(obs[-1, :, 1], obs[-1, :, 2]))

        if self.task == "GoThroughPositions":
            stds["num_goals_reached_std"] = np.std(np.sum(get_mask(obs[:, :, 0]), axis=0))

        if self.task == "TrackVelocities":
            stds["tracking_error_std"] = np.std(np.linalg.norm(obs[:, :, :3], axis=-1))

        # success rate std (binomial)
        stds["success_rate_std"] = np.sqrt(self.res["success_rate"] * (1 - self.res["success_rate"]) / obs.shape[1])

        # merge means + stds
        merged = {
            **self.res,
            **stds,
            "robot": self.robot,
            "task": self.task,
            "rl_lib": self.lib,
            "combo": self.combo,
            "seed": self.seed,
        }

        df = pd.DataFrame([merged])
        out_path = Path(self.RESULTS_DIR) / f"{self.combo}_run-{self.seed}.csv"
        df.to_csv(out_path, index=False)
        print(f"[Evaluator] saved {out_path}")


# -------------------------------------------------------------------------
# Helper to aggregate all run-CSVs into the big comparison table
def aggregate_runs(results_dir="results", template_csv="Evaluation_Metrics CSV Template.csv"):
    """After all runs are done, call this once to fill the master table."""
    import glob

    run_files = glob.glob(os.path.join(results_dir, "*_run-*.csv"))
    if not run_files:
        print("No run-level CSVs found.")
        return

    big = pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)
    # Compute mean across seeds (group by combo)
    agg = big.groupby("combo").mean().reset_index()

    # load the template and fill matching rows
    template = pd.read_csv(template_csv)
    for _, row in agg.iterrows():
        idx = template["combo"] == row["combo"]
        if idx.any():
            for col in row.index:
                if col.endswith("_std") or col.endswith("_mean"):
                    template.loc[idx, col] = row[col]

    template.to_csv("Evaluation_Metrics_Filled.csv", index=False)
    print("[Aggregator] saved Evaluation_Metrics_Filled.csv")
