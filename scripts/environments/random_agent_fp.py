# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Retrieve robot configuration and determine action space structure
    robot_cfg = env_cfg.robot_cfg
    num_thrusters = robot_cfg.num_thrusters  # TODO: change to num_actions
    is_reaction_wheel = robot_cfg.is_reaction_wheel

    # Reset the environment
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # Generate binary actions (0 or 1) for thrusters
            thruster_actions = torch.randint(
                0, 2, (args_cli.num_envs, num_thrusters), device=env.unwrapped.device
            ).float()

            # Generate continuous action for the reaction wheel if present
            if is_reaction_wheel:
                reaction_wheel_action = torch.rand((args_cli.num_envs, 1), device=env.unwrapped.device) * 2 - 1
                actions = torch.cat((thruster_actions, reaction_wheel_action), dim=1)
            else:
                actions = thruster_actions

            # Apply actions to the environment
            env.step(actions)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
