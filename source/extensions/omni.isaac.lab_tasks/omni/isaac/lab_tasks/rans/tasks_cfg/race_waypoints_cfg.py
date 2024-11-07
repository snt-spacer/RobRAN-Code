from omni.isaac.lab.utils import configclass
from dataclasses import MISSING

from .task_core_cfg import TaskCoreCfg

import math


@configclass
class RaceWaypointsCfg(TaskCoreCfg):
    """Configuration for the RaceWaypoints task."""

    # Initial conditions
    spawn_min_dist: float = 0.5
    """Minimal distance between the spawn position and the target position in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 5.0
    """Maximal distance between the spawn position and the target position in m. Defaults to 5.0 m."""
    spawn_min_heading_dist: float = 0.0
    """Minimal angle between the spawn orientation and the angle required to be looking at the target in rad.
    Defaults to 0.0 rad."""
    spawn_max_heading_dist: float = math.pi
    """Maximal angle between the spawn orientation and the angle required to be looking at the target in rad.
    Defaults to pi rad."""
    spawn_min_lin_vel: float = 0.0
    """Minimal linear velocity when spawned in m/s. Defaults to 0.0 m/s."""
    spawn_max_lin_vel: float = 0.0
    """Maximal linear velocity when spawned in m/s. Defaults to 0.0 m/s."""
    spawn_min_ang_vel: float = 0.0
    """Minimal angular velocity when spawned in rad/s. Defaults to 0.0 rad/s."""
    spawn_max_ang_vel: float = 0.0
    """Maximal angular velocity when spawned in rad/s. Defaults to 0.0 rad/s."""

    # Goal spawn
    max_num_corners: int = 13
    """Maximal number of corners. Defaults to 13."""
    min_num_corners: int = 9
    """Minimal number of corners. Defaults to 9."""
    track_rejection_angle: float = (12.5 / 180.0) * math.pi
    """Angle in radians to reject tracks that have too sharp corners. Defaults to 12.5 degrees.
    sharp corners can lead to self-intersecting tracks."""
    scale: float = 15.0
    """Scale of the track. Defaults to 20.0."""
    rad: float = 0.2
    """A coefficient that affects the smoothness of the track. Defaults to 0.2."""
    edgy: float = 0.0
    """A coefficient that affects the edginess of the track. Defaults to 0.0."""
    loop: bool = True
    """Whether the track should loop or not. Defaults to True."""

    # Observation
    num_subsequent_goals: int = 2
    """Number of subsequent goals available in the observation. Defaults to 2."""

    # Tolerance
    position_tolerance: float = 0.15
    """Tolerance for the position of the robot. Defaults to 1cm."""
    maximum_robot_distance: float = 30.0
    """Maximal distance between the robot and the target position. Defaults to 10 m."""

    # Reward Would be good to have a config for each reward type
    position_heading_exponential_reward_coeff: float = 0.25
    position_heading_weight: float = 0.05
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    linear_velocity_weight: float = -0.00
    angular_velocity_weight: float = -0.05
    boundary_weight: float = -10.0
    time_penalty: float = -0.0
    reached_bonus: float = 10.0
    progress_weight: float = 1.0
