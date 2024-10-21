from omni.isaac.lab.utils import configclass
from dataclasses import MISSING

from .task_core_cfg import TaskCoreCfg

import math


@configclass
class TrackVelocityCfg(TaskCoreCfg):
    """Configuration for the TrackVelocityTask task."""

    # Initial conditions
    minimal_spawn_linear_velocity: float = 0.0
    """Minimal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    maximal_spawn_linear_velocity: float = 0.0
    """Maximal linear velocity at spawn pose in m/s. Defaults to 0.0 m/s."""
    minimal_spawn_angular_velocity: float = 0.0
    """Minimal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""
    maximal_spawn_angular_velocity: float = 0.0
    """Maximal angular velocity at spawn in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    enable_linear_velocity: bool = True  # A random sign is added
    minimal_linear_velocity: float = 0.0
    maximal_linear_velocity: float = 1.0
    enable_lateral_velocity: bool = False  # A random sign is added
    minimal_lateral_velocity: float = 0.0
    maximal_lateral_velocity: float = 0.0
    enable_angular_velocity: bool = True  # A random sign is added
    minimal_angular_velocity: float = 0.0
    maximal_angular_velocity: float = 0.4

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
