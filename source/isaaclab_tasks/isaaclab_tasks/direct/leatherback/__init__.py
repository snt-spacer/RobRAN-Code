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
    id="Isaac-Leatherback-GoToPosition-Direct-v0",
    entry_point=f"{__name__}.leatherback_go_to_position_env:LeatherbackGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_go_to_position_env:LeatherbackGoToPositionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoToPose-Direct-v0",
    entry_point=f"{__name__}.leatherback_go_to_pose_env:LeatherbackGoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_go_to_pose_env:LeatherbackGoToPoseEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-TrackVelocities-Direct-v0",
    entry_point=f"{__name__}.leatherback_track_velocities_env:LeatherbackTrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_track_velocities_env:LeatherbackTrackVelocitiesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoThroughPositions-Direct-v0",
    entry_point=f"{__name__}.leatherback_go_through_positions_env:LeatherbackGoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_go_through_positions_env:LeatherbackGoThroughPositionsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoThroughPoses-Direct-v0",
    entry_point=f"{__name__}.leatherback_go_through_poses_env:LeatherbackGoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_go_through_poses_env:LeatherbackGoThroughPosesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-RaceWaypoints-Direct-v0",
    entry_point=f"{__name__}.leatherback_race_waypoints_env:LeatherbackRaceWaypointsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_race_waypoints_env:LeatherbackRaceWaypointsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-RaceWayposes-Direct-v0",
    entry_point=f"{__name__}.leatherback_race_wayposes_env:LeatherbackRaceWayposesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_race_wayposes_env:LeatherbackRaceWayposesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-PushBlock-Direct-v0",
    entry_point=f"{__name__}.leatherback_push_block_env:LeatherbackPushBlockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_push_block_env:LeatherbackPushBlockEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
