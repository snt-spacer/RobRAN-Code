import gymnasium as gym

from . import agents
from .jetbot_go_to_position import JetbotGoToPositionEnv, JetbotGoToPositionEnvCfg

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