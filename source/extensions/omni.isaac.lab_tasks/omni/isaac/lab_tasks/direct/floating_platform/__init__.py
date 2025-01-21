# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .floating_platform_go_through_poses import FloatingPlatformGoThroughPosesEnv, FloatingPlatformGoThroughPosesEnvCfg
from .floating_platform_go_through_positions import (
    FloatingPlatformGoThroughPositionsEnv,
    FloatingPlatformGoThroughPositionsEnvCfg,
)
from .floating_platform_go_to_pose import FloatingPlatformGoToPoseEnv, FloatingPlatformGoToPoseEnvCfg
from .floating_platform_go_to_position import FloatingPlatformGoToPositionEnv, FloatingPlatformGoToPositionEnvCfg
from .floating_platform_track_velocities import (
    FloatingPlatformTrackVelocitiesEnv,
    FloatingPlatformTrackVelocitiesEnvCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-FloatingPlatform-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.floating_platform:FloatingPlatformGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FloatingPlatformGoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FloatingPlatformPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FloatingPlatform-GoToPose-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.floating_platform:FloatingPlatformGoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FloatingPlatformGoToPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FloatingPlatformPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FloatingPlatform-TrackVelocities-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.floating_platform:FloatingPlatformTrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FloatingPlatformTrackVelocitiesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FloatingPlatformPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FloatingPlatform-GoThroughPositions-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.floating_platform:FloatingPlatformGoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FloatingPlatformGoThroughPositionsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FloatingPlatformPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FloatingPlatform-GoThroughPoses-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.floating_platform:FloatingPlatformGoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FloatingPlatformGoThroughPosesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FloatingPlatformPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
