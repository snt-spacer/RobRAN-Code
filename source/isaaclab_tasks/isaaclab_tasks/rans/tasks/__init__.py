# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# isort: off
from .task_core import TaskCore  # noqa: F401, F403

# isort: on

from isaaclab_tasks.rans.utils.misc import factory

from .go_through_poses import GoThroughPosesTask  # noqa: F401, F403
from .go_through_positions import GoThroughPositionsTask  # noqa: F401, F403
from .go_to_pose import GoToPoseTask  # noqa: F401, F403
from .go_to_position import GoToPositionTask  # noqa: F401, F403
from .go_to_position_with_obstacles import GoToPositionWithObstaclesTask  # noqa: F401, F403
from .push_block import PushBlockTask  # noqa: F401, F403
from .race_gates import RaceGatesTask  # noqa: F401, F403
from .race_waypoints import RaceWaypointsTask  # noqa: F401, F403
from .race_wayposes import RaceWayposesTask  # noqa: F401, F403
from .task_core import TaskCore  # noqa: F401, F403
from .track_velocities import TrackVelocitiesTask  # noqa: F401, F403

TASK_FACTORY = factory()
TASK_FACTORY.register("GoThroughPoses", GoThroughPosesTask)
TASK_FACTORY.register("GoThroughPositions", GoThroughPositionsTask)
TASK_FACTORY.register("GoToPose", GoToPoseTask)
TASK_FACTORY.register("GoToPosition", GoToPositionTask)
TASK_FACTORY.register("PushBlock", PushBlockTask)
TASK_FACTORY.register("RaceWaypoints", RaceWaypointsTask)
TASK_FACTORY.register("RaceWayposes", RaceWayposesTask)
TASK_FACTORY.register("TrackVelocities", TrackVelocitiesTask)
TASK_FACTORY.register("GoToPositionWithObstacles", GoToPositionWithObstaclesTask)
TASK_FACTORY.register("RaceGates", RaceGatesTask)
