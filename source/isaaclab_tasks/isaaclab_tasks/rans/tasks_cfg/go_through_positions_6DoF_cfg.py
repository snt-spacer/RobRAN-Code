# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .task_core_cfg import TaskCoreCfg


@configclass
class GoThroughPositions3DCfg(TaskCoreCfg):
    """Configuration for the GoThroughPosition task in 3D space."""

    # Initial conditions
    spawn_min_dist: float = 0.5
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 5.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 5.0 m."""
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
    position_tolerance: float = 0.10
    """Tolerance for the position of the robot. Defaults to 1cm."""
    maximum_robot_distance: float = 30.0
    """Maximal distance between the robot and the target pose. Defaults to 10 m."""

    # Reward Would be good to have a config for each reward type
    position_exponential_reward_coeff: float = 0.25
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    linear_velocity_weight: float = -0.005
    angular_velocity_weight: float = -0.05
    boundary_weight: float = -10.0
    time_penalty: float = -0.0
    reached_bonus: float = 15.0
    progress_weight: float = 4.0

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=True,
        randomization_modes=["uniform"],  # Use uniform noise
        slices=[
            (0, 3),
            (3, 9),
            (9, 12),
            (12, 15),
        ]
        + [(15 + i * 3, 15 + (i + 1) * 3) for i in range(num_subsequent_goals)],
        max_delta=[0.03, 0.01, 0.02, 0.01] + [0.03] * num_subsequent_goals,
    )

    # Spaces
    observation_space: int = (
        15 + 3 * num_subsequent_goals
    )  # local pos err xyz + local orientn err rpy + root lin vel + root ang vel + local pos error xyz for each subsequent goal
    state_space: int = 0
    action_space: int = 0
    gen_space: int = 10
