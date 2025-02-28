# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .turtlebot2_go_through_poses import TurtleBot2GoThroughPosesEnv, TurtleBot2GoThroughPosesEnvCfg
from .turtlebot2_go_through_positions import TurtleBot2GoThroughPositionsEnv, TurtleBot2GoThroughPositionsEnvCfg
from .turtlebot2_go_to_pose import TurtleBot2GoToPoseEnv, TurtleBot2GoToPoseEnvCfg
from .turtlebot2_go_to_position import TurtleBot2GoToPositionEnv, TurtleBot2GoToPositionEnvCfg
from .turtlebot2_push_block import TurtleBot2PushBlockEnv, TurtleBot2PushBlockEnvCfg
from .turtlebot2_race_waypoints import TurtleBot2RaceWaypointsEnv, TurtleBot2RaceWaypointsEnvCfg
from .turtlebot2_race_wayposes import TurtleBot2RaceWayposesEnv, TurtleBot2RaceWayposesEnvCfg
from .turtlebot2_track_velocities import TurtleBot2TrackVelocitiesEnv, TurtleBot2TrackVelocitiesEnvCfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Leatherback-TrackVelocities-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2TrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2TrackVelocitiesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Leatherback-GoThroughPoses-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2GoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2GoThroughPosesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoThroughPositions-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2GoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2GoThroughPositionsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoToPose-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2GoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2GoToPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2GoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2GoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-PushBlock-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2PushBlockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2PushBlockEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-RaceWaypoints-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2RaceWaypointsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2RaceWaypointsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-RaceWayposes-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:TurtleBot2RaceWayposesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TurtleBot2RaceWayposesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
