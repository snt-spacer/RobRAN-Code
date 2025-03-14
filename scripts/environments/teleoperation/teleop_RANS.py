# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    elif "Jetbot" in hydra_args[0]:
        if delta_pose[0][0] > 0:
            return torch.tensor([[1, 1]], device=delta_pose.device)
        elif delta_pose[0][0] < 0:
            return torch.tensor([[-1, -1]], device=delta_pose.device)
        elif delta_pose[0][1] > 0:
            return torch.tensor([[-0.25, 0.25]], device=delta_pose.device)
        elif delta_pose[0][1] < 0:
            return torch.tensor([[0.25, -0.25]], device=delta_pose.device)
        else:
            return torch.tensor([[0, 0]], device=delta_pose.device)
    elif "Leatherback" in hydra_args[0]:
        action = torch.tensor([[0, 0]], device=delta_pose.device)

        if delta_pose[0][0] > 0:
            action += torch.tensor([[1, 0]], device=delta_pose.device)
        if delta_pose[0][0] < 0:
            action += torch.tensor([[-1, 0]], device=delta_pose.device)
        if delta_pose[0][1] > 0:
            action += torch.tensor([[0, 1]], device=delta_pose.device)
        if delta_pose[0][1] < 0:
            action += torch.tensor([[0, -1]], device=delta_pose.device)
        return action
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # modify configuration
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose, gripper_command)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
