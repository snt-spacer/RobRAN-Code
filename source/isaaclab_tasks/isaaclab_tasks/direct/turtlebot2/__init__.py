# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-TurtleBot2-TrackVelocities-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_track_velocities_env:TurtleBot2TrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_track_velocities_env:TurtleBot2TrackVelocitiesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-TurtleBot2-GoThroughPoses-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_go_through_poses_env:TurtleBot2GoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_go_through_poses_env:TurtleBot2GoThroughPosesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TurtleBot2-GoThroughPositions-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_go_through_positions_env:TurtleBot2GoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_go_through_positions_env:TurtleBot2GoThroughPositionsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TurtleBot2-GoToPose-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_go_to_pose_env:TurtleBot2GoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_go_to_pose_env:TurtleBot2GoToPoseEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TurtleBot2-GoToPosition-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_go_to_position_env:TurtleBot2GoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_go_to_position_env:TurtleBot2GoToPositionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TurtleBot2-PushBlock-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_push_block_env:TurtleBot2PushBlockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_push_block_env:TurtleBot2PushBlockEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TurtleBot2-RaceWaypoints-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_race_waypoints_env:TurtleBot2RaceWaypointsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_race_waypoints_env:TurtleBot2RaceWaypointsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TurtleBot2-RaceWayposes-Direct-v0",
    entry_point=f"{__name__}.turtlebot2_race_wayposes_env:TurtleBot2RaceWayposesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot2_race_wayposes_env:TurtleBot2RaceWayposesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
