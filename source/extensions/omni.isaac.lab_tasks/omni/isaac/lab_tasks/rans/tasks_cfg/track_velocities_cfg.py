# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .task_core_cfg import TaskCoreCfg


@configclass
class TrackVelocitiesCfg(TaskCoreCfg):
    """Configuration for the TrackVelocityTask task."""

    # Initial conditions
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    enable_linear_velocity: bool = True
    """Enable linear velocity goal. Defaults to True."""
    goal_min_lin_vel: float = 0.0
    """Minimal linear velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_lin_vel: float = 2.0
    """Maximal linear velocity goal in m/s. Defaults to 2.0 m/s. (a random sign is added)"""
    enable_lateral_velocity: bool = False
    """Enable lateral velocity goal. Defaults to False."""
    goal_min_lat_vel: float = 0.0
    """Minimal lateral velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_lat_vel: float = 0.0
    """Maximal lateral velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    enable_angular_velocity: bool = True
    """Enable angular velocity goal. Defaults to True."""
    goal_min_ang_vel: float = 0.0
    """Minimal angular velocity goal in rad/s. Defaults to 0.0 rad/s. (a random sign is added)"""
    goal_max_ang_vel: float = 0.4
    """Maximal angular velocity goal in rad/s. Defaults to 0.4 rad/s. (a random sign is added)"""

    # Settings
    resample_at_regular_interval: bool = True
    interval: tuple[int, int] = (60, 80)
    smoothing_factor: tuple[float, float] = (0.0, 0.9)

    # Tolerance
    linear_velocity_tolerance: float = 0.01
    lateral_velocity_tolerance: float = 0.01
    angular_velocity_tolerance: float = 0.05
    maximum_robot_distance: float = 1000.0  # should be plenty enough not to reset
    resample_after_steps_in_tolerance: int = 50

    # Reward Would be good to have a config for each reward type
    lin_vel_exponential_reward_coeff: float = 0.25
    lat_vel_exponential_reward_coeff: float = 0.25
    ang_vel_exponential_reward_coeff: float = 0.25
    linear_velocity_weight: float = 0.5
    lateral_velocity_weight: float = 0.0
    angular_velocity_weight: float = 0.5

    # Visualization
    visualization_linear_velocity_scale: float = 1.0
    visualization_angular_velocity_scale: float = 2.5
