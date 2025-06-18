# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
from pathlib import Path
from typing import Any

import pandas as pd


class PerformanceEvaluatorV2:
    THRESH_DIST = 0.20
    THRESH_DIST_GTP = 10  # For GoToPose task (10 degrees)
    THRESH_LIN_VEL = 0.2  # Threshold for linear velocity tracking (m/s)
    THRESH_ANG_VEL = 10  # Threshold for angular velocity tracking (degrees/s)

    RESULTS_DIR = "results"

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

    def _control_variation(self, actions: np.ndarray) -> float:
        # Sum of control signal variations over time
        variations = np.abs(np.diff(actions, axis=0))
        return np.mean(np.sum(variations, axis=-1))  # axis -1 sums over the last dimension (action space)

    def _heading_error(self, cos_val, sin_val):
        return np.abs(np.arctan2(sin_val, cos_val))

    def _basic_goto(self, obs):
        # GoToPosition: index 0 = dist
        dist = obs[:, :, 0]
        final_d = dist[-1]
        self.res["success_rate"] = np.mean(final_d < self.THRESH_DIST)
        self.res["final_distance_error"] = np.mean(final_d)

        reach_mask = dist < self.THRESH_DIST
        first_hit = reach_mask.argmax(axis=0)
        hit_any = reach_mask.any(axis=0)
        times = np.where(hit_any, first_hit, self.T)
        self.res["avg_time_to_target"] = np.mean(times)

    def _goto_pose(self, obs):
        self._basic_goto(obs)
        cos_h, sin_h = obs[-1, :, 3], obs[-1, :, 4]
        # Modulate success rate based on heading error
        heading_err = self._heading_error(cos_h, sin_h)
        heading_mask = np.rad2deg(heading_err) < self.THRESH_DIST_GTP  # 10 degrees
        self.res["heading_error"] = np.mean(np.rad2deg(heading_err))
        self.res["success_rate"] *= np.mean(heading_mask)

    def _go_through_positions(self, obs: np.ndarray):
        """
        Evaluate GoThroughPositions task with ordered goal tracking.
        """

        T, N, D = obs.shape
        num_goals = (D - 6) // 3  # Number of waypoints encoded in obs
        goal_idx = np.zeros(N, dtype=int)  # Tracks current goal index per env
        goals_reached = np.zeros(N, dtype=int)

        for t in range(T):
            for n in range(N):
                g = goal_idx[n]
                if g >= num_goals:
                    continue  # All goals already reached

                dist_idx = 6 + g * 3
                dist_to_goal = obs[t, n, dist_idx]

                if dist_to_goal < self.THRESH_DIST:
                    goal_idx[n] += 1  # Move to next
                    goals_reached[n] += 1

        self.res["num_goals_reached"] = np.mean(goals_reached)

        # Final distance to current (or last) goal
        final_d = np.array([obs[-1, n, 6 + min(goal_idx[n], num_goals - 1) * 3] for n in range(N)])
        self.res["final_distance_error"] = np.mean(final_d)
        self.res["success_rate"] = np.mean(final_d < self.THRESH_DIST)

        # First time current goal was reached
        reach_mask = np.zeros((T, N), dtype=bool)
        for t in range(T):
            for n in range(N):
                g = goal_idx[n]
                if g < num_goals:
                    dist_idx = 6 + g * 3
                    reach_mask[t, n] = obs[t, n, dist_idx] < self.THRESH_DIST

        first_hit = reach_mask.argmax(axis=0)
        hit_any = reach_mask.any(axis=0)
        times = np.where(hit_any, first_hit, self.T)

        self.res["avg_time_to_target"] = np.mean(times)

    def _track_velocities(self, obs):
        # : tracking linear/angular velocity within ϵv and ϵw during the episode
        lin_err = np.linalg.norm(obs[:, :, 0:2], axis=-1)  # vx, vy
        ang_err = np.abs(obs[:, :, 2])  # omega
        self.res["linear_velocity_error"] = np.mean(lin_err)
        self.res["angular_velocity_error"] = np.mean(ang_err)
        # compute score based on thresholds
        lin_mask = lin_err < self.THRESH_LIN_VEL
        ang_mask = ang_err < np.deg2rad(self.THRESH_ANG_VEL)  # convert to radians
        self.res["success_rate"] = np.mean(lin_mask & ang_mask)

    def evaluate(self) -> dict[str, Any]:
        obs = self.data["obs"]
        act = self.data["act"]
        self.res["control_variation"] = self._control_variation(act)

        if self.task == "GoToPosition":
            self._basic_goto(obs)
        elif self.task == "GoToPose":
            self._goto_pose(obs)
        elif self.task == "GoThroughPositions":
            self._go_through_positions(obs)
        elif self.task == "TrackVelocities":
            self._track_velocities(obs)

        return self.res

    def save_csv(self):
        """
        Save one CSV per run:
        <results>/<combo>_run-<seed>.csv
        Columns: metric, mean, std, and metadata like robot/task/lib/seed
        """
        Path(self.RESULTS_DIR).mkdir(exist_ok=True)
        obs = self.data["obs"]
        acts = self.data["act"]
        T, N, _ = obs.shape

        stds = {}

        # --- control variation std ---
        if "control_variation" in self.res:
            stds["control_variation_std"] = np.std(np.sum(np.abs(np.diff(acts, axis=0)), axis=-1))

        # --- final distance ---
        if "final_distance_error" in self.res:
            goal_idx = np.zeros(N, dtype=int)
            T, N, D = obs.shape
            num_goals = (D - 6) // 3
            if self.task == "GoThroughPositions":
                final_d = np.array([obs[-1, n, 6 + min(goal_idx[n], num_goals - 1) * 3] for n in range(N)])
                stds["final_distance_error_std"] = np.std(final_d)

            else:
                stds["final_distance_error_std"] = np.std(obs[-1, :, 0])

        # --- time to target ---
        if "avg_time_to_target" in self.res:
            reach_mask = (obs[:, :, 3] if self.task == "GoThroughPositions" else obs[:, :, 0]) < self.THRESH_DIST
            first_hit = reach_mask.argmax(axis=0)
            hit_any = reach_mask.any(axis=0)
            times = np.where(hit_any, first_hit, self.T)
            stds["avg_time_to_target_std"] = np.std(times)

        # --- heading error ---
        if self.task == "GoToPose" and "heading_error" in self.res:
            cos_h, sin_h = obs[-1, :, 3], obs[-1, :, 4]
            rad = np.abs(np.arctan2(sin_h, cos_h))
            stds["heading_error_std"] = np.std(np.rad2deg(rad))

        # --- velocity tracking error ---
        if self.task == "TrackVelocities" and "linear_velocity_error" in self.res:
            lin_err = np.linalg.norm(obs[:, :, 0:2], axis=-1)  # vx, vy
            ang_err = np.abs(obs[:, :, 2])  # omega
            stds["linear_velocity_error_std"] = np.std(lin_err)
            stds["angular_velocity_error_std"] = np.std(ang_err)

        # --- goals reached ---
        if self.task == "GoThroughPositions" and "num_goals_reached" in self.res:
            T, N, D = obs.shape
            num_goals = (D - 6) // 3
            goal_idx = np.zeros(N, dtype=int)
            for t in range(T):
                for n in range(N):
                    g = goal_idx[n]
                    if g >= num_goals:
                        continue
                    dist_idx = 6 + g * 3
                    dist = obs[t, n, dist_idx]
                    if dist < self.THRESH_DIST:
                        goal_idx[n] += 1
            final_d = np.array([obs[-1, n, 6 + min(goal_idx[n], num_goals - 1) * 3] for n in range(N)])
            stds["final_distance_error_std"] = np.std(final_d)

        # --- success rate ---
        if "success_rate" in self.res:
            stds["success_rate_std"] = np.sqrt(self.res["success_rate"] * (1 - self.res["success_rate"]) / N)

        # --- compile everything ---
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

    # ---------------------------------------------------------------------

    def export_timeseries_metrics(self, path="timeseries"):
        """
        Export per-timestep, per-env values to long-format CSV
        Columns: timestep, env_id, metric, value, robot, task, rl_lib, seed
        """
        Path(path).mkdir(exist_ok=True)
        obs = self.data["obs"]  # [T, N, D] - where T = timesteps, N = envs, D = dimensions
        T, N, _ = obs.shape

        rows = []

        for t in range(T):
            for n in range(N):
                if self.task in ["GoToPosition", "GoToPose"]:
                    dist = obs[t, n, 0]
                    rows.append([t, n, "distance_error", dist])

                if self.task == "GoToPose":
                    cos_h, sin_h = obs[t, n, 3], obs[t, n, 4]
                    heading = np.abs(np.arctan2(sin_h, cos_h))
                    heading_deg = np.rad2deg(heading)
                    rows.append([t, n, "heading_error", heading_deg])

                if self.task == "TrackVelocities":
                    vx_err = obs[t, n, 0]
                    vy_err = obs[t, n, 1]
                    w_err = obs[t, n, 2]
                    rows.append([t, n, "vx_error", vx_err])
                    rows.append([t, n, "vy_error", vy_err])
                    rows.append([t, n, "omega_error", w_err])
                    v_err = np.linalg.norm([vx_err, vy_err, w_err])
                    rows.append([t, n, "velocity_error", v_err])

                if self.task == "GoThroughPositions":
                    dist = obs[t, n, 3]
                    rows.append([t, n, "distance_error", dist])
                    hit = float(dist < self.THRESH_DIST)
                    rows.append([t, n, "cumulative_goals", hit])

        # cumulative goals sum per env
        df = pd.DataFrame(rows, columns=["timestep", "env_id", "metric", "value"])
        if self.task == "GoThroughPositions":
            mask = df["metric"] == "cumulative_goals"
            df.loc[mask, "value"] = df[mask].groupby("env_id")["value"].cumsum()

        # Add context columns and write
        df["robot"] = self.robot
        df["task"] = self.task
        df["rl_lib"] = self.lib
        df["seed"] = self.seed
        df.to_csv(Path(path) / f"{self.combo}_run-{self.seed}.csv", index=False)
        print(f"[Timeseries] saved {self.combo}_run-{self.seed}.csv")

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
        agg = big.groupby("combo").mean(numeric_only=True).reset_index()

        template = pd.read_csv(template_csv)
        for _, row in agg.iterrows():
            idx = template["combo"] == row["combo"]
            if idx.any():
                for col in row.index:
                    if col.endswith("_std") or col.endswith("_mean") or col in ["robot", "task", "combo"]:
                        if col in template.columns:
                            template.loc[idx, col] = row[col]
                        else:
                            print(f"[WARN] Column '{col}' missing in template.")

        template.to_csv("Evaluation_Metrics_Filled.csv", index=False)
        print("[Aggregator] saved Evaluation_Metrics_Filled.csv")
