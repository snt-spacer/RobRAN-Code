# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import numpy as np

import pandas as pd


class PerformanceEvaluator:
    """
    Modular RL evaluation framework for IsaacLab navigation tasks.
    Computes task-specific and robot-specific metrics, normalizes performance,
    and enables systematic comparison across tasks and robots.
    """

    def __init__(self, task_name, robot_name, episode_data, max_horizon, config=None):
        """
        Initializes the evaluator.

        Args:
            task_name (str): Name of the task being evaluated.
            robot_name (str): Name of the robot executing the task.
            episode_data (dict): Contains observations, actions, rewards, etc.
            max_horizon (int): Maximum episode length.
            config (dict, optional): Custom configuration for normalization.
        """
        self.task_name = task_name
        self.robot_name = robot_name
        self.episode_data = episode_data
        self.max_horizon = max_horizon
        self.config = config if config else {}
        self.results = {}

    def compute_basic_metrics(self):
        """Computes general performance metrics applicable to all tasks."""
        rewards = np.array(self.episode_data.get("rews", []))
        actions = np.array(self.episode_data.get("act", []))

        self.results["total_reward"] = np.sum(rewards)
        self.results["mean_reward"] = np.mean(rewards)
        self.results["action_magnitude"] = np.mean(np.linalg.norm(actions, axis=-1))
        self.results["action_variability"] = np.std(actions)
        # TODO: Add time under thresholds from target, etc. (10, 5, 2, 1 cm)
        self.results["time_completion"] = len(rewards) / self.max_horizon

    def compute_navigation_metrics(self):
        """Computes task-specific navigation performance metrics."""
        observations = np.array(self.episode_data.get("obs", []))

        if observations.ndim < 3:
            return  # Avoid indexing errors on empty or improperly shaped data

        if self.task_name in ["GoToPosition"]:
            distance_to_target = observations[:, :, 0]  # Distance is first in obs
            self.results["final_distance_to_goal"] = np.mean(distance_to_target[:, -1])

        elif self.task_name in ["GoToPose"]:
            distance_to_target = observations[:, :, 0]  # First element is distance
            heading_error = np.abs(np.arctan2(observations[:, :, 2], observations[:, :, 1]))
            self.results["final_distance_to_goal"] = np.mean(distance_to_target[:, -1])
            self.results["heading_alignment"] = np.mean(heading_error)

        elif self.task_name in ["GoThroughPositions", "GoThroughPoses", "RaceWaypoints", "RaceWayposes"]:
            waypoints_distance = observations[:, :, 0]  # First element is distance to next waypoint
            path_efficiency = np.mean(waypoints_distance[:, -1] / np.sum(np.diff(waypoints_distance, axis=1), axis=-1))
            self.results["final_distance_to_goal"] = np.mean(waypoints_distance[:, -1])
            self.results["path_efficiency"] = path_efficiency

        elif self.task_name == "PushBlock":
            block_distance = observations[:, :, 0]  # Distance between robot and block
            self.results["final_block_distance"] = np.mean(block_distance[:, -1])

        elif self.task_name == "TrackVelocities":
            velocity_error = np.linalg.norm(observations[:, :, 0], axis=-1)
            self.results["velocity_tracking_error"] = np.mean(velocity_error)

    def compute_robot_specific_metrics(self):
        """Applies normalization based on robot dynamics and constraints."""
        if self.robot_name in ["Jetbot", "Leatherback"]:
            self.results["turning_efficiency"] = self.results.get("action_magnitude", 0) / (
                1 + self.results.get("action_variability", 1)
            )

        if self.robot_name == "Floating Platform":
            self.results["thruster_usage"] = np.mean(np.abs(np.array(self.episode_data.get("acts", []))))

    def normalize_metrics(self):
        """Normalizes metrics across different tasks and robots for fair comparison."""
        complexity_factors = {
            "GoToPose": 1.2,
            "PushBlock": 1.5,
            "RaceWaypoints": 1.1,
            "TrackVelocities": 1.0,
        }
        factor = complexity_factors.get(self.task_name, 1.0)

        for key in self.results:
            self.results[key] /= factor

    def export_results(self, output_format="json", file_name="evaluation_results"):
        """Exports evaluation results to a structured format (JSON or CSV)."""
        if output_format == "json":
            with open(f"{file_name}.json", "w") as f:
                json.dump(self.results, f, indent=4)
        elif output_format == "csv":
            df = pd.DataFrame([self.results])
            df.to_csv(f"{file_name}.csv", index=False)

    def evaluate(self):
        """Runs the full evaluation pipeline."""
        self.compute_basic_metrics()
        self.compute_navigation_metrics()
        self.compute_robot_specific_metrics()
        self.normalize_metrics()
        return self.results
