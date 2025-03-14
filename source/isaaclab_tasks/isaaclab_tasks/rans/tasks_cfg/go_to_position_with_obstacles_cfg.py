# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

from .go_to_position_cfg import GoToPositionCfg


@configclass
class GoToPositionWithObstaclesCfg(GoToPositionCfg):
    """Configuration for the GoToPosition task."""

    # Initial conditions
    spawn_min_dist: float = 3.5
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 2.0 m."""
    spawn_max_dist: float = 5.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 5.0 m."""

    # Tolerance
    minimum_obstacle_distance_to_target: float = 1.0
    """Minimal distance between the target and the obstacles. Defaults to 0.5 m."""
    minimum_obstacle_distance_to_robot: float = 0.5
    """Minimal distance between the robot and the obstacles. Defaults to 0.1 m."""
    collision_threshold: float = 3.0
    """Collision threshold. Defaults to 10.0"""

    # Obstacles
    minimum_point_distance = 0.05
    """The minimum distance between the points sampled to create the obstacles grid. Should be between 0 and 1. Smaller values can create more complex env."""
    max_num_vis_obstacles: int = 8
    """Max number of obstacles visible in the environment. Defaults to 8."""
    obstacle_radius: float = 0.2
    """Radius of the obstacles. Defaults to 0.2 m."""
    obstacles_height: float = 0.5
    """Height of the obstacles. Defaults to 0.5 m."""
    obstacles_storage_height_pos: float = -3.0
    """Height where to store the obstacles. Defaults to -2.0 m."""
    max_obstacle_distance_from_target: float = 10
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""
    min_obstacle_distance_from_target: float = 1
    """Minimal distance between the target and the obstacles. Defaults to 1 m."""
    min_obstacle_distance_from_robot: float = 1
    """Minimal distance between the robot and the obstacles. Defaults to 1 m."""
    min_distance_between_obstacle: float = 0.5
    """Minimal distance between the obstacles. Defaults to 0.5 m."""

    # Spaces
    observation_space: int = 15

    collision_penalty: float = -10.0
    progress_weight: float = 1.0

    def __post_init__(self):
        assert self.min_distance_between_obstacle > self.obstacle_radius, "Min distance between obstacles is too small."
