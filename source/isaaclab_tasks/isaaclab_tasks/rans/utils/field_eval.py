# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

import pandas as pd
import seaborn as sns


class FieldEvaluation:
    """
    Evaluates field test logs for GoToPosition, GoToPose, and GoThroughPositions tasks.
    Supports single-test and multi-test batch evaluations.
    """

    def __init__(
        self,
        file_path=None,
        folder_path=None,
        task_name=None,
        threshold_pos=0.1,
        threshold_yaw=np.deg2rad(10),
        plot_results=True,
        save_dir=None,
    ):
        """
        Args:
            file_path (str): Path to a single CSV file for evaluation.
            folder_path (str): Path to a folder containing multiple CSVs.
            task_name (str): Task name ('GoToPosition', 'GoToPose', 'GoThroughPositions').
            threshold_pos (float): Distance threshold for success (default: 0.1m).
            threshold_yaw (float): Orientation threshold for success in GoToPose (default: 10° in radians).
            plot_results (bool): Whether to generate plots.
            save_dir (str): Directory to save plots and results.
        """
        self.file_path = file_path
        self.folder_path = folder_path
        self.task_name = task_name
        self.threshold_pos = threshold_pos
        self.threshold_yaw = threshold_yaw
        self.plot_results = plot_results
        self.save_dir = folder_path + "/results" if folder_path else file_path.replace(".csv", "/results")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.results = {}
        self.colors = sns.color_palette("Paired")  # Default dark theme matching the slide

        if file_path:
            self.evaluate_single_test(file_path)
        elif folder_path:
            self.evaluate_multiple_tests(folder_path)
        else:
            raise ValueError("Please provide either a file or a folder path for evaluation.")

    def load_data(self, file_path):
        """Loads CSV log data and ensures required columns exist."""
        df = pd.read_csv(file_path)
        required_columns = ["elapsed_time.s", "distance_error.m"]
        if self.task_name == "GoToPose":
            required_columns.append("target_heading_error.rad")
        elif self.task_name == "GoThroughPositions":
            required_columns.append("num_goals_reached.u")

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {file_path}")
        # transform heading error to absolute value
        if "heading_error.rad" in df.columns:
            df["heading_error.rad"] = np.abs(df["heading_error.rad"])
        return df

    def evaluate_single_test(self, file_path):
        """Evaluates a single test log."""
        self.data = self.load_data(file_path)
        self.data["time"] = self.data["elapsed_time.s"]  # Use elapsed time as primary axis
        self.compute_metrics()
        if self.plot_results:
            self.plot_metrics(file_path)
        self.export_results(file_path)

    def evaluate_multiple_tests(self, folder_path):
        """Evaluates multiple test logs in a folder and computes aggregated statistics."""
        all_files = glob(os.path.join(folder_path, "*.csv"))
        all_results = []
        all_distances = []
        all_times = []
        all_heading_errors = []

        for file in all_files:
            print(f"Evaluating: {file}")
            self.data = self.load_data(file)
            self.data["time"] = self.data["elapsed_time.s"]
            self.compute_metrics()
            all_results.append(self.results.copy())
            # [TODO: fix in csv wrong mapping of info for error distance in GoThroughPositions]
            dist = "task_data.lin_vel_body.x.m/s" if self.task_name == "GoThroughPositions" else "distance_error.m"
            # Store data for aggregated plots
            all_distances.append(self.data[dist].values)
            all_times.append(self.data["time"].values)
            if self.task_name == "GoToPose":
                all_heading_errors.append(np.rad2deg(self.data["heading_error.rad"].values))

        # Compute statistics
        df_results = pd.DataFrame(all_results)
        mean_results = df_results.mean().to_dict()
        std_results = df_results.std().to_dict()

        # Print summary
        print(f"\n🔹 Evaluated {len(all_files)} tests in '{folder_path}'")
        print("📊 Mean Performance:", mean_results)
        print("📊 Std Deviation:", std_results)

        # Ensure the directory exists before saving summary results
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save statistics
        summary_path = os.path.join(self.save_dir, "summary_results.json")
        with open(summary_path, "w") as f:
            json.dump({"mean": mean_results, "std": std_results}, f, indent=4)

        # Generate aggregated plots
        if self.plot_results:
            self.plot_aggregated_metrics(folder_path, all_times, all_distances, all_heading_errors)

    def compute_metrics(self):
        """Computes key performance metrics."""
        final_distance = self.data["distance_error.m"].iloc[-1]
        success = final_distance < self.threshold_pos

        self.results["final_distance_to_goal"] = final_distance
        self.results["success"] = int(success)

        if self.task_name == "GoToPose":
            final_heading_error = self.data["heading_error.rad"].iloc[-1]
            success_orientation = final_heading_error < self.threshold_yaw
            self.results["final_heading_error"] = final_heading_error
            self.results["success"] = int(success and success_orientation)

        elif self.task_name == "GoThroughPositions":
            total_goals = self.data["num_goals_reached.u"].max()
            success = total_goals > 0
            self.results["waypoints_reached"] = total_goals
            self.results["success"] = int(success)

    def plot_metrics(self, file_path):
        """Generates individual test plots for performance analysis."""
        sns.set_theme(style="whitegrid")
        file_name = os.path.basename(file_path).replace(".csv", "")

        if self.task_name == "GoToPosition" or self.task_name == "GoToPose":
            # Plot distance convergence
            plt.figure(figsize=(6, 4))
            plt.plot(self.data["time"], self.data["distance_error.m"], label="Distance to Goal", color="blue")
            plt.axhline(self.threshold_pos, linestyle="--", color="red", label=f"Threshold ({self.threshold_pos}m)")
            plt.xlabel("Time (s)")
            plt.ylabel("Distance to Goal (m)")
            plt.title(f"{self.task_name}: Distance Convergence")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            plt.savefig(f"{self.save_dir}/{file_name}_distance_convergence.png")
        # Plot heading error for GoToPose
        if self.task_name == "GoToPose":
            plt.figure(figsize=(6, 4))
            plt.plot(
                self.data["time"], np.rad2deg(self.data["heading_error.rad"]), label="Heading Error", color="green"
            )
            plt.axhline(np.rad2deg(self.threshold_yaw), linestyle="--", color="orange", label="Threshold (10°)")
            plt.xlabel("Time (s)")
            plt.ylabel("Heading Error (°)")
            plt.title("GoToPose: Heading Convergence")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            plt.savefig(f"{self.save_dir}/{file_name}_heading_convergence.png")
            plt.close()
        elif self.task_name == "GoThroughPositions":
            plt.figure(figsize=(6, 4))
            plt.plot(self.data["time"], self.data["num_goals_reached.u"], label="Waypoints Reached", color="purple")
            plt.xlabel("Time (s)")
            plt.ylabel("Waypoints Reached")
            plt.title("GoThroughPositions: Waypoints Reached")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            plt.savefig(f"{self.save_dir}/{file_name}_waypoints_reached.png")
            plt.close

    def plot_aggregated_metrics(self, folder_path, all_times, all_distances, all_heading_errors):
        """Generates aggregated plots for multi-test evaluations."""
        sns.set_theme(style="whitegrid")

        #  reshaping variable-length sequences into uniform matrices for computing aggregated statistics (mean & std).
        #  to be able to get valid stats with trajectories of different lengths
        max_length = max(len(arr) for arr in all_distances)
        times_matrix = np.full((len(all_times), max_length), np.nan)
        distances_matrix = np.full((len(all_distances), max_length), np.nan)

        for i, (time_series, dist_series) in enumerate(zip(all_times, all_distances)):
            times_matrix[i, : len(time_series)] = time_series
            distances_matrix[i, : len(dist_series)] = dist_series

        mean_distance = np.nanmean(distances_matrix, axis=0)
        std_distance = np.nanstd(distances_matrix, axis=0)
        mean_time = np.nanmean(times_matrix, axis=0)

        # Plot aggregated distance convergence
        plt.figure(figsize=(6, 4))
        plt.plot(mean_time, mean_distance, label="Mean Distance to Goal", color=self.colors[1])
        plt.fill_between(
            mean_time, mean_distance - std_distance, mean_distance + std_distance, alpha=0.2, color=self.colors[1]
        )
        plt.axhline(
            self.threshold_pos, linestyle="--", color=self.colors[9], label=f"Threshold ({self.threshold_pos}m)"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Distance to Goal (m)")
        plt.title(f"{self.task_name}: Aggregated Distance Convergence")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        plt.savefig(f"{self.save_dir}/{self.task_name}_aggregated_distance_convergence.png")
        plt.close()
        # Plot heading error convergence for GoToPose
        if self.task_name == "GoToPose":
            max_heading_length = max(len(arr) for arr in all_heading_errors)
            heading_matrix = np.full((len(all_heading_errors), max_heading_length), np.nan)

            for i, heading_series in enumerate(all_heading_errors):
                heading_matrix[i, : len(heading_series)] = heading_series

            mean_heading = np.nanmean(heading_matrix, axis=0)
            std_heading = np.nanstd(heading_matrix, axis=0)
            mean_time = np.nanmean(times_matrix, axis=0)

            # Plot aggregated heading error
            plt.figure(figsize=(6, 4))
            plt.plot(mean_time, mean_heading, label="Mean Heading Error", color=self.colors[7])
            plt.fill_between(
                mean_time, mean_heading - std_heading, mean_heading + std_heading, alpha=0.2, color=self.colors[7]
            )
            plt.axhline(np.rad2deg(self.threshold_yaw), linestyle="--", color=self.colors[5], label="Threshold (10°)")
            plt.xlabel("Time (s)")
            plt.ylabel("Heading Error (°)")
            plt.title("Aggregated Heading Convergence")
            plt.legend()
            plt.savefig(f"{self.save_dir}/GoToPose_heading_aggregated.png")
            plt.close()

    def export_results(self, file_path):
        """Exports evaluation results to JSON."""
        file_name = os.path.basename(file_path).replace(".csv", "_results.json")
        with open(os.path.join(self.save_dir, file_name), "w") as f:
            json.dump(self.results, f, indent=4)


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate field test logs.")
    parser.add_argument("--file", type=str, help="Path to a single CSV file.")
    parser.add_argument("--folder", type=str, help="Path to a folder containing multiple test CSVs.")
    parser.add_argument(
        "--task", type=str, required=True, help="Task name (GoToPosition, GoToPose, GoThroughPositions)."
    )
    args = parser.parse_args()

    FieldEvaluation(file_path=args.file, folder_path=args.folder, task_name=args.task)
