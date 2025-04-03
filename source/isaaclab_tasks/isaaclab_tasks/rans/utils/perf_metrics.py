# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import seaborn as sns


class PerformanceMetrics:
    """
    Computes performance metrics for RL-trained agents in navigation tasks.
    """

    def __init__(self, task_name, robot_name, episode_data, max_horizon, plot_metrics=False, save_path=None):
        """
        Args:
            task_name (str): The name of the task being evaluated.
            robot_name (str): The robot executing the task.
            episode_data (dict): Contains episode information (obs, rewards, actions, etc.).
            max_horizon (int): Maximum number of steps in each episode.
            plot_metrics (bool): If True, generates plots for relevant tasks.
        """
        self.task_name = task_name
        self.robot_name = robot_name
        self.episode_data = episode_data
        self.max_horizon = max_horizon
        self.plot_metrics = plot_metrics
        self.save_path = save_path
        self.results = {}
        self.results_timseries = {}

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # define plot colors
        self.colors = sns.color_palette("Paired")
        # {blu,gree,red,orange,purple,brown} - {1,3,5,7,9,11}

        # Define success thresholds
        self.thresholds = {
            "GoToPosition": {"epsilon_p": 0.1},
            "GoToPose": {"epsilon_p": 0.1, "epsilon_theta": np.deg2rad(10)},
            "GoThroughPositions": {"epsilon_p": 0.2},
            "GoThroughPoses": {"epsilon_p": 0.2, "epsilon_theta": np.deg2rad(10)},
            "TrackVelocities": {"tau_v": 0.2, "tau_w": np.deg2rad(5)},
            "PushBlock": {"epsilon_p": 0.1},
            "RaceWaypoints": {"time_threshold": None},  # Set externally if needed
        }

    def compute_basic_metrics(self):
        """Compute task-agnostic RL evaluation metrics."""
        rewards = np.array(self.episode_data.get("rews", []))  # (steps, num_envs)
        actions = np.array(self.episode_data.get("act", []))  # (steps, num_envs, action_dim)

        # General performance metrics
        self.results["total_reward"] = np.sum(rewards)
        self.results["mean_reward"] = np.mean(rewards)
        self.results["action_magnitude"] = np.mean(np.linalg.norm(actions, axis=-1))
        self.results["action_variability"] = np.std(actions)

    def compute_navigation_metrics(self):
        """Compute task-specific performance metrics using paper-defined thresholds."""
        observations = np.array(self.episode_data.get("obs", []))  # (steps, num_envs, data_dim)
        num_steps, num_envs, _ = observations.shape

        success_mask = np.zeros(num_envs, dtype=bool)

        if self.task_name == "GoToPosition":
            final_distances = observations[-1, :, 0]
            success_mask = final_distances < self.thresholds["GoToPosition"]["epsilon_p"]
            self.results["final_distance_to_goal"] = np.mean(final_distances)
            self.results_timseries["distance_to_goal"] = observations[:, :, 0]

            if self.plot_metrics:
                self.plot_go_to_position()

        elif self.task_name == "GoToPose":
            final_distances = observations[-1, :, 0]
            final_heading_errors = np.abs(np.arctan2(observations[-1, :, 4], observations[-1, :, 3]))
            success_mask = (final_distances < self.thresholds["GoToPose"]["epsilon_p"]) & (
                final_heading_errors < self.thresholds["GoToPose"]["epsilon_theta"]
            )
            self.results["final_distance_to_goal"] = np.mean(final_distances)
            self.results["heading_alignment_error"] = np.mean(final_heading_errors)
            self.results_timseries["distance_to_goal"] = observations[:, :, 0]
            self.results_timseries["heading_errors"] = np.abs(np.arctan2(observations[:, :, 4], observations[:, :, 3]))
            if self.plot_metrics:
                self.plot_go_to_pose()

        elif self.task_name == "GoThroughPositions":
            cumulative_goals = self.compute_goals_reached(
                observations, threshold=self.thresholds["GoThroughPositions"]["epsilon_p"]
            )

            # Final statistics
            final_goals_per_env = cumulative_goals[-1]
            mean_goals = np.mean(final_goals_per_env)
            std_goals = np.std(final_goals_per_env)

            print(f"Final goals reached per env: Mean = {mean_goals:.2f}, Std = {std_goals:.2f}")

            # Success: define as at least X% of goals reached (or N goals)
            min_goals_required = 3  # ⚠️ Adjust based on your task's difficulty
            success_mask = final_goals_per_env >= min_goals_required
            self.results["mean_goals_reached"] = mean_goals
            self.results["std_goals_reached"] = std_goals
            self.results["success_rate"] = np.mean(success_mask)  # Percentage of envs meeting min goals

            # Store full time series for later analysis
            self.results_timseries["cumulative_goals"] = cumulative_goals
            self.results_timseries["active_goal_distances"] = observations[:, :, 0]

            if self.plot_metrics:
                self.plot_goals_histogram(final_goals_per_env)
                self.plot_cumulative_goals(cumulative_goals)

        elif self.task_name == "TrackVelocities":
            linear_velocity_errors = np.abs(observations[:, :, 0])
            angular_velocity_errors = np.abs(observations[:, :, 2])

            linear_tracking_success = linear_velocity_errors < self.thresholds["TrackVelocities"]["tau_v"]
            angular_tracking_success = angular_velocity_errors < self.thresholds["TrackVelocities"]["tau_w"]

            success_mask = np.mean(linear_tracking_success * angular_tracking_success, axis=0) > 0.7
            self.results["linear_velocity_tracking_error"] = np.mean(linear_velocity_errors)
            self.results["angular_velocity_tracking_error"] = np.mean(angular_velocity_errors)
            self.results_timseries["linear_velocity_errors"] = linear_velocity_errors
            self.results_timseries["angular_velocity_errors"] = angular_velocity_errors

            if self.plot_metrics:
                self.plot_velocity_tracking()

        self.results["success_rate"] = np.mean(success_mask)

    def compute_goals_reached(self, observations, threshold=0.2):
        """
        Compute cumulative goals reached per env over time, tracking transitions.

        Args:
            observations (np.ndarray): (timesteps, num_envs, obs_dim)
            threshold (float): Distance threshold to consider a goal reached.

        Returns:
            np.ndarray: (timesteps, num_envs) cumulative goals reached.
        """
        obs = observations  # (timesteps, num_envs, obs_dim)
        timesteps, num_envs, _ = obs.shape

        cumulative_goals = np.zeros((timesteps, num_envs))
        goal_reached_flag = np.zeros((num_envs,), dtype=bool)

        for t in range(timesteps):
            active_goal_distances = obs[t, :, 0]
            newly_reached = (active_goal_distances < threshold) & (~goal_reached_flag)

            # Update cumulative goals where a new goal was just reached
            if t > 0:
                cumulative_goals[t] = cumulative_goals[t - 1] + newly_reached
            else:
                cumulative_goals[t] = newly_reached

            # Update flag to avoid re-counting the same goal
            goal_reached_flag = active_goal_distances < threshold

            # Reset flag if the distance increases again (goal changed)
            goal_reached_flag = goal_reached_flag & (active_goal_distances < threshold)

        return cumulative_goals

    def plot_cumulative_goals(self, cumulative_goals):
        """
        Plot cumulative goals over time (mean ± std).
        """
        sns.set_theme(style="whitegrid")
        timesteps = np.arange(cumulative_goals.shape[0])
        mean_goals = np.mean(cumulative_goals, axis=1)
        std_goals = np.std(cumulative_goals, axis=1)

        plt.figure(figsize=(7, 5))
        window_size = 100
        smooth_mean_goals = moving_average(mean_goals, window_size)
        smooth_std_goals = moving_average(std_goals, window_size)
        smooth_timesteps = timesteps[: len(smooth_mean_goals)]

        plt.plot(
            smooth_timesteps,
            smooth_mean_goals,
            label="Smoothed Mean Goals Reached",
            color=sns.color_palette("Paired")[2],
            linewidth=2,
        )
        plt.fill_between(
            smooth_timesteps,
            smooth_mean_goals - smooth_std_goals,
            smooth_mean_goals + smooth_std_goals,
            alpha=0.2,
            color=sns.color_palette("Paired")[3],
        )

        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Cumulative Goals Reached", fontsize=12)
        plt.title("GoThroughPositions: Cumulative Goals Over Time", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        plt.legend(loc="upper left", fontsize=10, frameon=True)

        if self.save_path:
            plot_filename = os.path.join(self.save_path, "GoThroughPositions_cumulative_goals.png")
            plt.savefig(plot_filename, dpi=200)
        plt.show()

    def plot_goals_histogram(self, final_goals_per_env):
        """
        Plot an improved histogram of total goals reached per environment using
        percentage and KDE distribution line, for a single robot.
        """
        import matplotlib.pyplot as plt
        import os

        import seaborn as sns

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 5))

        # Plot histogram with KDE and percentage scaling
        sns.histplot(
            final_goals_per_env,
            bins=range(int(final_goals_per_env.min()), int(final_goals_per_env.max()) + 1),
            kde=True,
            kde_kws={"bw_adjust": 2},
            stat="percent",
            color=self.colors[3],
            alpha=0.6,
            edgecolor="black",
            linewidth=0.6,
        )

        # Format Y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))

        # Plot styling
        plt.xlabel("Total Goals Reached", fontsize=13)
        plt.ylabel("Percentage of Environments", fontsize=13)
        plt.title("Distribution of Goals Reached per Environment", fontsize=15, fontweight="bold")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        # Save or show
        if self.save_path:
            plot_filename = os.path.join(self.save_path, "GoThroughPositions_goals_histogram_improved.png")
            plt.savefig(plot_filename, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.tight_layout()
        plt.show()

    def plot_velocity_tracking(self):
        """Plots the velocity tracking error over time."""
        obs = np.array(self.episode_data["obs"])
        linear_velocity_errors = np.abs(obs[:, :, 0])
        angular_velocity_errors = np.abs(obs[:, :, 2])
        mean_error = np.mean(linear_velocity_errors, axis=1)
        std_error = np.std(linear_velocity_errors, axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(mean_error, label="Mean Linear Velocity Error", color="blue")
        plt.fill_between(
            np.arange(len(mean_error)), mean_error - std_error, mean_error + std_error, alpha=0.2, color=self.colors[1]
        )
        plt.axhline(
            self.thresholds["TrackVelocities"]["tau_v"],
            linestyle="--",
            color=self.colors[3],
            label="Threshold (0.2m/s)",
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Velocity Error (m/s)")
        plt.title("TrackVelocities: Linear Velocity Tracking Error")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        if self.save_path:
            plt.savefig(self.save_path + "_linear", dpi=200)
        plt.show()
        angular_velocity_errors = np.abs(obs[:, :, 2])
        mean_error = np.mean(angular_velocity_errors, axis=1)
        std_error = np.std(angular_velocity_errors, axis=1)
        plt.figure(figsize=(6, 4))
        plt.plot(mean_error, label="Mean Angular Velocity Error", color="blue")
        plt.fill_between(
            np.arange(len(mean_error)), mean_error - std_error, mean_error + std_error, alpha=0.2, color=self.colors[1]
        )
        plt.axhline(
            self.thresholds["TrackVelocities"]["tau_w"],
            linestyle="--",
            color=self.colors[3],
            label=r"Threshold (10$^\circ$/s)",
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Velocity Error (rad/s)")
        plt.title("TrackVelocities: Angular Velocity Tracking Error")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        # Save or show the figure
        if self.save_path:
            plt.savefig(self.save_path + "_angular", dpi=200)
        plt.show()
        plt.close("all")

    def plot_go_to_position(self):
        """
        Plots the mean and standard deviation of distance to goal over time for the GoToPosition task.
        """
        sns.set_theme(style="whitegrid")

        obs = np.array(self.episode_data["obs"])  # (steps, num_envs, data_dim)
        distances = obs[:, :, 0]  # Distance to goal

        mean_distance = np.mean(distances, axis=1)
        std_distance = np.std(distances, axis=1)
        timesteps = np.arange(distances.shape[0])
        # Create figure
        plt.figure(figsize=(7, 5))

        # Plot mean distance and variance
        plt.plot(timesteps, mean_distance, label="Mean Distance", color=self.colors[1], linewidth=2)
        plt.fill_between(
            timesteps, mean_distance - std_distance, mean_distance + std_distance, alpha=0.2, color=self.colors[1]
        )

        # Plot threshold
        plt.axhline(
            self.thresholds["GoToPosition"]["epsilon_p"],
            linestyle="--",
            color=self.colors[5],
            label="Threshold (0.1m)",
            linewidth=1.5,
        )

        # Labels and aesthetics
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Distance to Goal (m)", fontsize=12)
        plt.title("GoToPosition: Distance Convergence", fontsize=14, fontweight="bold")
        plt.ylim(bottom=0)  # Ensure zero alignment

        # Legend and grid
        plt.legend(loc="upper right", fontsize=10, frameon=True)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        # Optimize layout
        plt.tight_layout()

        # Save or show the figure
        if self.save_path:
            plt.savefig(self.save_path, dpi=200)
        plt.show()

    def plot_go_to_pose(self):
        """
        Plots the convergence of position and orientation errors for the GoToPose task.
        """
        # sns.set_theme(style="whitegrid")

        obs = np.array(self.episode_data["obs"])  # (steps, num_envs, data_dim)
        distances = obs[:, :, 0]  # Distance to goal

        # Compute heading errors in degrees
        all_cos_sin_phi_headings = obs[:, :, 3:5]  # Target headings
        all_phi_heading_distances = np.abs(
            np.arctan2(all_cos_sin_phi_headings[:, :, 1], all_cos_sin_phi_headings[:, :, 0])
        )
        all_phi_heading_distances = np.rad2deg(all_phi_heading_distances)  # Convert to degrees

        # Compute mean and std
        mean_distance = np.mean(distances, axis=1)
        std_distance = np.std(distances, axis=1)

        mean_heading = np.mean(all_phi_heading_distances, axis=1).squeeze()
        std_heading = np.std(all_phi_heading_distances, axis=1).squeeze()

        timesteps = np.arange(distances.shape[0])

        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(7, 5))

        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Distance to Goal (m)", color=self.colors[1])
        ax1.plot(timesteps, mean_distance, label="Mean Distance", color=self.colors[1], linewidth=2)
        ax1.fill_between(
            timesteps, mean_distance - std_distance, mean_distance + std_distance, alpha=0.2, color=self.colors[1]
        )
        ax1.axhline(
            self.thresholds["GoToPose"]["epsilon_p"],
            linestyle="--",
            color=self.colors[5],
            label="Threshold (0.1m)",
            linewidth=1.5,
        )
        ax1.tick_params(axis="y", labelcolor=self.colors[1])

        # Secondary Y-axis for Orientation Error
        ax2 = ax1.twinx()

        ax2.set_ylabel("Heading Error (°)", color=self.colors[7])
        ax2.plot(timesteps, mean_heading, label="Mean Heading Error", color=self.colors[7], linewidth=2)
        ax2.fill_between(
            timesteps, mean_heading - std_heading, mean_heading + std_heading, alpha=0.2, color=self.colors[7]
        )
        ax2.axhline(
            np.rad2deg(self.thresholds["GoToPose"]["epsilon_theta"]),
            linestyle="--",
            color=self.colors[9],
            label="Threshold (10°)",
            linewidth=1.5,
        )
        ax2.tick_params(axis="y", labelcolor=self.colors[7])

        # Ensure both axes start from zero
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        # ax2.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)

        # Title and layout adjustments
        fig.suptitle("GoToPose: Position & Orientation Convergence", fontsize=14, fontweight="bold")
        fig.tight_layout()

        # Legends
        ax1.legend(loc="upper right", fontsize=10, frameon=True)
        ax2.legend(loc="upper left", fontsize=10, frameon=True)

        # Save or show the figure
        if self.save_path:
            plt.savefig(self.save_path, dpi=200)
        plt.show()

    def export_results(self, output_format="csv", file_name="performance_results"):
        """Export computed results to JSON or CSV format in a flattened, readable structure."""
        if self.save_path:
            file_name = os.path.join(self.save_path, file_name)

        if output_format == "json":
            with open(f"{file_name}.json", "w") as f:
                json.dump({k: v.tolist() for k, v in self.results_timseries.items()}, f, indent=4)

        elif output_format == "csv":
            records = []
            num_steps, num_envs = next(iter(self.results_timseries.values())).shape

            for key, data in self.results_timseries.items():
                for step in range(num_steps):
                    for env in range(num_envs):
                        records.append({"timestep": step, "env_id": env, "metric": key, "value": data[step, env]})

            df = pd.DataFrame(records)
            df.to_csv(f"{file_name}.csv", index=False)

    def evaluate(self):
        """Run all performance evaluations and return results."""
        self.compute_basic_metrics()
        self.compute_navigation_metrics()
        self.export_results()
        return self.results


def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
