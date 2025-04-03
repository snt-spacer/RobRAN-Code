# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class GoToPose3DCfg(TaskCoreCfg):
    """Configuration for the GoToPosition task in 3D space."""

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
    """Maximal distance from the origin of the environment. Defaults to 2.0."""

    # Tolerance
    position_tolerance: float = 0.01
    """Tolerance for the position of the robot. Defaults to 1cm."""
    orientation_tolerance: float = math.pi / 36  # 5 degrees
    """Tolerance for the orientation of the robot. Defaults to 5 degrees."""
    maximum_robot_distance: float = 10.0
    """Maximal distance between the robot and the target pose. Defaults to 10 m."""
    reset_after_n_steps_in_tolerance: int = 100
    """Reset the environment after n steps in tolerance. Defaults to 100 steps."""

    # Reward
    position_exponential_reward_coeff: float = 0.5
    orientation_exponential_reward_coeff: float = 1.0
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    pose_weight: float = 2.0
    linear_velocity_weight: float = -0.08
    angular_velocity_weight: float = -0.05
    boundary_weight: float = -10.0
    progress_weight: float = 1.5

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 3), (3, 9), (9, 12), (12, 15)],
        max_delta=[0.03, 0.01, 0.03, 0.03],
    )

    # Spaces
    observation_space: int = 15  # pos err xyz + rotation + root lin vel + root ang vel
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 6
