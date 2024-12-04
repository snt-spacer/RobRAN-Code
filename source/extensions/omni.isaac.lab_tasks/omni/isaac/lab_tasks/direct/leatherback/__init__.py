# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .leatherback_go_through_poses import LeatherbackGoThroughPosesEnv, LeatherbackGoThroughPosesEnvCfg
from .leatherback_go_through_positions import LeatherbackGoThroughPositionsEnv, LeatherbackGoThroughPositionsEnvCfg
from .leatherback_go_to_pose import LeatherbackGoToPoseEnv, LeatherbackGoToPoseEnvCfg
from .leatherback_go_to_position import LeatherbackGoToPositionEnv, LeatherbackGoToPositionEnvCfg
from .leatherback_push_block import LeatherbackPushBlockEnv, LeatherbackPushBlockEnvCfg
from .leatherback_race_waypoints import LeatherbackRaceWaypointsEnv, LeatherbackRaceWaypointsEnvCfg
from .leatherback_race_wayposes import LeatherbackRaceWayposesEnv, LeatherbackRaceWayposesEnvCfg
from .leatherback_track_velocities import LeatherbackTrackVelocitiesEnv, LeatherbackTrackVelocitiesEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Leatherback-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackGoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoToPose-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackGoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackGoToPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-TrackVelocities-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackTrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackTrackVelocitiesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoThroughPositions-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackGoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackGoThroughPositionsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-GoThroughPoses-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackGoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackGoThroughPosesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-RaceWaypoints-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackRaceWaypointsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackRaceWaypointsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-RaceWayposes-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackRaceWayposesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackRaceWayposesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Leatherback-PushBlock-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.leatherback:LeatherbackPushBlockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackPushBlockEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
