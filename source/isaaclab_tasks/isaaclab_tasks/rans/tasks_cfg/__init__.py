# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# isort: off
from .task_core_cfg import TaskCoreCfg  # noqa: F401, F403

# isort: on
from isaaclab_tasks.rans.utils.misc import factory

from .go_through_poses_6DoF_cfg import GoThroughPoses3DCfg  # noqa: F401, F403
from .go_through_poses_cfg import GoThroughPosesCfg  # noqa: F401, F403
from .go_through_positions_6DoF_cfg import GoThroughPositions3DCfg  # noqa: F401, F403
from .go_through_positions_cfg import GoThroughPositionsCfg  # noqa: F401, F403
from .go_to_pose_6DoF_cfg import GoToPose3DCfg  # noqa: F401, F403
from .go_to_pose_cfg import GoToPoseCfg  # noqa: F401, F403
from .go_to_position_6DoF_cfg import GoToPosition3DCfg  # noqa: F401, F403
from .go_to_position_cfg import GoToPositionCfg  # noqa: F401, F403
from .go_to_position_with_obstacles_cfg import GoToPositionWithObstaclesCfg
from .push_block_cfg import PushBlockCfg  # noqa: F401, F403
from .race_gates_cfg import RaceGatesCfg  # noqa: F401, F403
from .race_waypoints_cfg import RaceWaypointsCfg  # noqa: F401, F403
from .race_wayposes_cfg import RaceWayposesCfg  # noqa: F401, F403
from .track_velocities_6DoF_cfg import TrackVelocities3DCfg  # noqa: F401, F403
from .track_velocities_cfg import TrackVelocitiesCfg  # noqa: F401, F403

TASK_CFG_FACTORY = factory()
TASK_CFG_FACTORY.register("GoThroughPoses", GoThroughPosesCfg)
TASK_CFG_FACTORY.register("GoThroughPoses3D", GoThroughPoses3DCfg)
TASK_CFG_FACTORY.register("GoThroughPositions", GoThroughPositionsCfg)
TASK_CFG_FACTORY.register("GoToPose", GoToPoseCfg)
TASK_CFG_FACTORY.register("GoToPose3D", GoToPose3DCfg)
TASK_CFG_FACTORY.register("GoToPosition", GoToPositionCfg)
TASK_CFG_FACTORY.register("GoToPosition3D", GoToPosition3DCfg)
TASK_CFG_FACTORY.register("PushBlock", PushBlockCfg)
TASK_CFG_FACTORY.register("RaceWaypoints", RaceWaypointsCfg)
TASK_CFG_FACTORY.register("RaceWayposes", RaceWayposesCfg)
TASK_CFG_FACTORY.register("TrackVelocities", TrackVelocitiesCfg)
TASK_CFG_FACTORY.register("TrackVelocities3D", TrackVelocities3DCfg)
TASK_CFG_FACTORY.register("GoToPositionWithObstacles", GoToPositionWithObstaclesCfg)
TASK_CFG_FACTORY.register("RaceGates", RaceGatesCfg)
