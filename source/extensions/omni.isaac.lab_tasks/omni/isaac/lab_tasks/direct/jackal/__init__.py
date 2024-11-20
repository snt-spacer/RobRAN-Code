import gymnasium as gym

from . import agents
from .jackal_go_to_position import JackalGoToPositionEnv, JackalGoToPositionEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Jackal-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.jackal:JackalGoToPositionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JackalGoToPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)