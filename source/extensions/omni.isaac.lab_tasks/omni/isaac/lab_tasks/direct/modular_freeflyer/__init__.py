# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .modular_freeflyer_go_through_poses import ModularFreeflyerGoThroughPosesEnv, ModularFreeflyerGoThroughPosesEnvCfg
from .modular_freeflyer_go_through_positions import (
    ModularFreeflyerGoThroughPositionsEnv,
    ModularFreeflyerGoThroughPositionsEnvCfg,
)
from .modular_freeflyer_go_to_pose import ModularFreeflyerGoToPoseEnv, ModularFreeflyerGoToPoseEnvCfg
from .modular_freeflyer_go_to_position import ModularFreeflyerGoToPositionEnv, ModularFreeflyerGoToPositionEnvCfg
from .modular_freeflyer_track_velocities import (
    ModularFreeflyerTrackVelocitiesEnv,
    ModularFreeflyerTrackVelocitiesEnvCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ModularFreeflyer-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.modular_freeflyer:ModularFreeflyerGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ModularFreeflyerGoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ModularFreeflyerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ModularFreeflyer-GoToPose-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.modular_freeflyer:ModularFreeflyerGoToPoseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ModularFreeflyerGoToPoseEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ModularFreeflyerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ModularFreeflyer-TrackVelocities-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.modular_freeflyer:ModularFreeflyerTrackVelocitiesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ModularFreeflyerTrackVelocitiesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ModularFreeflyerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ModularFreeflyer-GoThroughPositions-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.modular_freeflyer:ModularFreeflyerGoThroughPositionsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ModularFreeflyerGoThroughPositionsEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ModularFreeflyerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ModularFreeflyer-GoThroughPoses-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.modular_freeflyer:ModularFreeflyerGoThroughPosesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ModularFreeflyerGoThroughPosesEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ModularFreeflyerPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
