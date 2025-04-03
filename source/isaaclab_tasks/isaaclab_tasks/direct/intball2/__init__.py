# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
IntBall2 environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-IntBall2-GoToPosition3D-Direct-v0",
    entry_point=f"{__name__}.intball2_go_to_position:IntBall2GoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.intball2_go_to_position:IntBall2GoToPositionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-IntBall2-GoToPose3D-Direct-v0",
    entry_point=f"{__name__}.intball2_go_to_pose:IntBall2GoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.intball2_go_to_pose:IntBall2GoToPoseEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-IntBall2-GoThroughPositions3D-Direct-v0",
    entry_point=f"{__name__}.intball2_go_through_positions:IntBall2GoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.intball2_go_through_positions:IntBall2GoThroughPositionsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-IntBall2-GoThroughPoses3D-Direct-v0",
    entry_point=f"{__name__}.intball2_go_through_poses:IntBall2GoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.intball2_go_through_poses:IntBall2GoThroughPosesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-IntBall2-TrackVelocities3D-Direct-v0",
    entry_point=f"{__name__}.intball2_track_velocities:IntBall2TrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.intball2_track_velocities:IntBall2TrackVelocitiesEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
