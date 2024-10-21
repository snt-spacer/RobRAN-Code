from omni.isaac.lab.utils import configclass
from dataclasses import MISSING

from .task_core_cfg import TaskCoreCfg

import math


@configclass
class GoToPositionCfg(TaskCoreCfg):
    """Configuration for the GoToPosition task."""

    # Initial conditions
    minimal_spawn_distance: float = 0.5
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""
    maximal_spawn_distance: float = 5.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 5.0 m."""
    minimal_spawn_cone: float = 0.0
    """Minimal angle between the spawn pose and the target pose in rad. Defaults to 0.0 rad."""
    maximal_spawn_cone: float = math.pi
    """Maximal angle between the spawn pose and the target pose in rad. Defaults to pi rad."""
    minimal_spawn_linear_velocity: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    maximal_spawn_linear_velocity: float = 0.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    minimal_spawn_angular_velocity: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    maximal_spawn_angular_velocity: float = 0.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    max_distance_from_origin: float = 0.0
    """Maximal distance from the origin of the environment. Defaults to 0.0."""

    # Tolerance
    position_tolerance: float = 0.01
    """Tolerance for the position of the robot. Defaults to 1cm."""
    maximum_robot_distance: float = 10.0
    """Maximal distance between the robot and the target pose. Defaults to 10 m."""
    reset_after_n_steps_in_tolerance: int = 100
    """Reset the environment after n steps in tolerance. Defaults to 100 steps."""

    # Reward Would be good to have a config for each reward type
    position_exponential_reward_coeff: float = 0.25
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    position_weight: float = 1.0
    linear_velocity_weight: float = -0.05
    angular_velocity_weight: float = -0.05
    boundary_weight: float = -10.0
