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
parser.add_argument("--robot_name", type=str, default=None, help="Name of the robot.")
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
a = 0.99
b = 0.90


def pre_process_actions(prev_action: torch.Tensor, delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    elif args_cli.robot_name == "Jetbot":
        if delta_pose[0][0] > 0:  # forward
            prev_action = prev_action * 0.9 + torch.tensor([[1, 1]], device=delta_pose.device) * 0.01
        elif delta_pose[0][0] < 0:  # backward
            prev_action = prev_action * 0.9 + torch.tensor([[-1, -1]], device=delta_pose.device) * 0.01
        elif delta_pose[0][1] > 0:  # left
            prev_action = prev_action * 0.9 + torch.tensor([[-1, 1]], device=delta_pose.device) * 0.01
        elif delta_pose[0][1] < 0:  # right
            prev_action = prev_action * 0.9 + torch.tensor([[1, -1]], device=delta_pose.device) * 0.01
        if gripper_command:
            prev_action.fill_(0)
        prev_action.clamp_(-1, 1)
        return prev_action
    elif args_cli.robot_name == "Leatherback":
        if delta_pose[0][0] > 0:  # forward
            prev_action[:, 0] = prev_action[:, 0] * a + 1 * (1 - a)
        if delta_pose[0][0] < 0:  # backward
            prev_action[:, 0] = prev_action[:, 0] * a - 1 * (1 - a)
        if delta_pose[0][1] > 0:  # left
            prev_action[:, 1] = prev_action[:, 1] * b + 1 * (1 - b)
        if delta_pose[0][1] < 0:  # right
            prev_action[:, 1] = prev_action[:, 1] * b - 1 * (1 - b)
        if gripper_command:
            prev_action.fill_(0)
        prev_action.clamp_(-1, 1)
        return prev_action
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
    prev_action = torch.zeros((1, 2), device=env.unwrapped.device)
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
            prev_action = pre_process_actions(prev_action, delta_pose, gripper_command)
            # apply actions
            env.step(prev_action)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
