# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class TrackVelocities3DCfg(TaskCoreCfg):
    """Configuration for the TrackVelocityTask task."""

    # Initial conditions
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 2.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 1.5
    """Maximal angular velocity at spawn in rad/s. Defaults to 1.5 rad/s."""
    spawn_initial_height: float = 1.0
    """Initial height of the robot in meters. Defaults to 1.0 m."""

    # Goal spawn
    enable_linear_velocity: bool = True
    """Enable linear velocity goal. Defaults to True."""
    enable_lateral_velocity: bool = True
    """Enable lateral velocity goal. Defaults to True."""
    enable_vertical_velocity: bool = True
    """Enable vertical velocity goal. Defaults to True."""
    goal_min_lin_vel: float = 0.0
    """Minimal linear velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_lin_vel: float = 1.0
    """Maximal linear velocity goal in m/s. Defaults to 1.0 m/s. (a random sign is added)"""
    goal_min_lat_vel: float = 0.0
    """Minimal lateral velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_lat_vel: float = 1.0
    """Maximal lateral velocity goal in m/s. Defaults to 1.0 m/s. (a random sign is added)"""
    goal_min_ver_vel: float = 0.0
    """Minimal vertical velocity goal in m/s. Defaults to 0.0 m/s. (a random sign is added)"""
    goal_max_ver_vel: float = 1.0
    """Maximal vertical velocity goal in m/s. Defaults to 1.0 m/s. (a random sign is added)"""
    enable_yaw_velocity: bool = True
    """Enable angular velocity goal. Defaults to True."""
    enable_pitch_velocity: bool = True
    """Enable angular velocity goal. Defaults to True."""
    enable_roll_velocity: bool = True
    """Enable angular velocity goal. Defaults to True."""
    goal_min_yaw_vel: float = 0.0
    """Minimal angular velocity goal in rad/s. Defaults to 0.0 rad/s. (a random sign is added)"""
    goal_max_yaw_vel: float = 0.4
    """Maximal angular velocity goal in rad/s. Defaults to 0.4 rad/s. (a random sign is added)"""
    goal_min_roll_vel: float = 0.0
    """Minimal angular velocity goal in rad/s. Defaults to 0.0 rad/s. (a random sign is added)"""
    goal_max_roll_vel: float = 0.4
    """Maximal angular velocity goal in rad/s. Defaults to 0.4 rad/s. (a random sign is added)"""
    goal_min_pitch_vel: float = 0.0
    """Minimal angular velocity goal in rad/s. Defaults to 0.0 rad/s. (a random sign is added)"""
    goal_max_pitch_vel: float = 0.4
    """Maximal angular velocity goal in rad/s. Defaults to 0.4 rad/s. (a random sign is added)"""

    # Settings
    resample_at_regular_interval: bool = True
    interval: tuple[int, int] = (60, 80)
    smoothing_factor: tuple[float, float] = (0.0, 0.9)

    # Tolerance
    linear_velocity_tolerance: float = 0.01
    lateral_velocity_tolerance: float = 0.01
    vertical_velocity_tolerance: float = 0.01
    yaw_velocity_tolerance: float = 0.05
    pitch_velocity_tolerance: float = 0.05
    roll_velocity_tolerance: float = 0.05
    maximum_robot_distance: float = 1000.0  # To make it plenty enough not to reset
    resample_after_steps_in_tolerance: int = 50

    # Reward
    lin_vel_exponential_reward_coeff: float = 0.1
    lat_vel_exponential_reward_coeff: float = 0.1
    ver_vel_exponential_reward_coeff: float = 0.1
    yaw_vel_exponential_reward_coeff: float = 0.1
    pitch_vel_exponential_reward_coeff: float = 0.1
    roll_vel_exponential_reward_coeff: float = 0.1
    linear_velocity_weight: float = 0.2
    lateral_velocity_weight: float = 0.2
    vertical_velocity_weight: float = 0.2
    yaw_velocity_weight: float = 0.2
    pitch_velocity_weight: float = 0.2
    roll_velocity_weight: float = 0.2

    # Visualization
    visualization_linear_velocity_scale: float = 1.0
    visualization_angular_velocity_scale: float = 2.0

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 6), (6, 12)],
        max_delta=[0.01, 0.03],
    )

    # Spaces
    observation_space: int = 12  # linear velocity errors + angular velocity errors + linear velocity + angular velocity
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 10
