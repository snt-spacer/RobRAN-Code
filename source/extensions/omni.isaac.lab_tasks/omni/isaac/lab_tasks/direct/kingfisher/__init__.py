# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .kingfisher_go_through_poses import KingfisherGoThroughPosesEnv, KingfisherGoThroughPosesEnvCfg
from .kingfisher_go_through_positions import KingfisherGoThroughPositionsEnv, KingfisherGoThroughPositionsEnvCfg
from .kingfisher_go_to_pose import KingfisherGoToPoseEnv, KingfisherGoToPoseEnvCfg
from .kingfisher_go_to_position import KingfisherGoToPositionEnv, KingfisherGoToPositionEnvCfg
from .kingfisher_track_velocities import KingfisherTrackVelocitiesEnv, KingfisherTrackVelocitiesEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Kingfisher-GoThroughPositions-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.kingfisher:KingfisherGoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KingfisherGoThroughPositionsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Kingfisher-GoThroughPoses-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.kingfisher:KingfisherGoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KingfisherGoThroughPosesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Kingfisher-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.kingfisher:KingfisherGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KingfisherGoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Kingfisher-GoToPose-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.kingfisher:KingfisherGoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KingfisherGoToPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Kingfisher-TrackVelocities-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.kingfisher:KingfisherTrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KingfisherTrackVelocitiesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
