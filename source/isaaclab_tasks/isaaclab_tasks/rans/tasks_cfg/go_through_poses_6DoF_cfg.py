# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class GoThroughPoses3DCfg(TaskCoreCfg):
    """Configuration for the GoThroughPosition task in 3D space."""

    # Initial conditions
    spawn_min_dist: float = 0.5
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 5.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 5.0 m."""
    spawn_min_heading_dist: float = 0.0
    """Minimal yaw offset between the spawn orientation and the target orientation in rad. Defaults to 0.0 rad."""
    spawn_max_heading_dist: float = math.pi
    """Maximal yaw offset between the spawn orientation and the target orientation in rad. Defaults to pi rad."""
    spawn_min_pitch_dist: float = 0.0
    """Minimal pitch offset between the spawn orientation and the target orientation in rad. Defaults to 0.0 rad."""
    spawn_max_pitch_dist: float = math.pi / 2
    """Maximal pitch offset between the spawn orientation and the target orientation in rad. Defaults to pi/2 rad."""
    spawn_min_roll_dist: float = 0.0
    """Minimal roll offset between the spawn orientation and the target orientation in rad. Defaults to 0.0 rad."""
    spawn_max_roll_dist: float = math.pi / 2
    """Maximal roll offset between the spawn orientation and the target orientation in rad. Defaults to pi/2 rad."""
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    goal_max_dist_from_origin: float = 2.0
    """Maximal distance from the origin of the environment. Defaults to 0.0."""
    goal_min_dist: float = 1.0
    """Minimal distance between the goals. Defaults to 1.0 m."""
    goal_max_dist: float = 5.0
    """Maximal distance between the goals. Defaults to 5.0 m."""
    goal_min_polar_angle: float = math.pi / 6
    """Minimal polar angle of the goal position. Defaults to pi/6 rad."""
    goal_max_polar_angle: float = 5 * math.pi / 6
    """Maximal polar angle of the goal position. Defaults to 5*pi/6 rad."""
    goal_min_azimuthal_angle: float = -math.pi
    """Minimal azimuthal angle of the goal position. Defaults to -pi rad."""
    goal_max_azimuthal_angle: float = math.pi
    """Maximal azimuthal angle of the goal position. Defaults to pi rad."""
    goal_min_yaw_offset: float = 0.0
    """Minimal yaw offset from the target orientation in rad. Defaults to 0.0 rad."""
    goal_max_yaw_offset: float = math.pi
    """Maximal yaw offset from the target orientation in rad. Defaults to pi rad."""
    goal_min_pitch_offset: float = 0.0
    """Minimal pitch offset from the target orientation in rad. Defaults to 0.0 rad."""
    goal_max_pitch_offset: float = math.pi / 2
    """Maximal pitch offset from the target orientation in rad. Defaults to pi/2 rad."""
    goal_min_roll_offset: float = 0.0
    """Minimal roll offset from the target orientation in rad. Defaults to 0.0 rad."""
    goal_max_roll_offset: float = math.pi / 2
    """Maximal roll offset from the target orientation in rad. Defaults to pi/2 rad."""
    max_num_goals: int = 10
    """Maximal number of goals. Defaults to 10."""
    min_num_goals: int = 6
    """Minimal number of goals. Defaults to 6."""
    loop: bool = True
    """Whether the goals should loop or not. Defaults to True."""

    # Observation
    num_subsequent_goals: int = 2
    """Number of subsequent goals available in the observation. Defaults to 2."""

    # Tolerance
    position_tolerance: float = 0.1
    """Tolerance for the position of the robot. Defaults to 1cm."""
    orientation_tolerance: float = math.pi / 20  # 9 degrees
    """Tolerance for the orientation of the robot. Defaults to 5 degrees."""
    maximum_robot_distance: float = 30.0
    """Maximal distance between the robot and the target position. Defaults to 10 m."""

    # Reward Would be good to have a config for each reward type
    position_exponential_reward_coeff: float = 0.5
    orientation_exponential_reward_coeff: float = 0.5
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    linear_velocity_weight: float = -0.005
    angular_velocity_weight: float = -0.05
    boundary_weight: float = -10.0
    time_penalty: float = -0.0
    reached_bonus: float = 10.0
    pose_weight: float = 0.3
    progress_weight: float = 3.0

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 3), (3, 9), (9, 12), (12, 15)]
        + [(15 + i * 9, 15 + i * 9 + 3) for i in range(num_subsequent_goals)]
        + [(15 + i * 9 + 3, 15 + i * 9 + 9) for i in range(num_subsequent_goals)],
        max_delta=[0.03, 0.01, 0.02, 0.01] + [0.03, 0.01] * num_subsequent_goals,
    )

    # Spaces
    observation_space: int = (
        15 + 9 * num_subsequent_goals
    )  # local pos err xyz + local orientn err rpy + root lin vel + root ang vel + local pos error xyz for each subsequent goal
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 16
