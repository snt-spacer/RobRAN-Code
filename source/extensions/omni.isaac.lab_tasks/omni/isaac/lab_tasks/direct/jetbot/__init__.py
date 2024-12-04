# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .jetbot_go_through_poses_env import JetbotGoThroughPosesEnv, JetbotGoThroughPosesEnvCfg
from .jetbot_go_through_positions_env import JetbotGoThroughPositionsEnv, JetbotGoThroughPositionsEnvCfg
from .jetbot_go_to_pose_env import JetbotGoToPoseEnv, JetbotGoToPoseEnvCfg
from .jetbot_go_to_position_env import JetbotGoToPositionEnv, JetbotGoToPositionEnvCfg
from .jetbot_push_block_env import JetbotPushBlockEnv, JetbotPushBlockEnvCfg
from .jetbot_track_velocities_env import JetbotTrackVelocitiesEnv, JetbotTrackVelocitiesEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Jetbot-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jetbot:JetbotGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotGoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jetbot-GoToPose-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jetbot:JetbotGoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotGoToPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jetbot-TrackVelocities-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jetbot:JetbotTrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotTrackVelocitiesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jetbot-GoThroughPositions-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jetbot:JetbotGoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotGoThroughPositionsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jetbot-GoThroughPoses-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jetbot:JetbotGoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotGoThroughPosesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Jetbot-PushBlock-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jetbot:JetbotPushBlockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotPushBlockEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
