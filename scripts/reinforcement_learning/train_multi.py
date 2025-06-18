# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# script to launch a sequence of training jobs for each robot-task pair over different seeds using either skrl or rl_games or rsl_rl

# sample output:
# ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-RANS-Single-v0 --num_envs 4096 env.robot_name=FloatingPlatform env.task_name=GoToPosition --headless --algorithm ppo-discrete --headless --wandb_project test

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))


def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple robot-task pairs with different seeds.")
    parser.add_argument(
        "--robot", type=str, required=True, help="Robot name (e.g., FloatingPlatform, Kingfisher, Turtlebot2, etc.)"
    )
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., GoToPosition, TrackVelocities, etc.)")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to use for training")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo-discrete",
        help="Algorithm to use for training (e.g., ppo-discrete, sac, etc.)",
    )
    parser.add_argument("--wandb_project", type=str, default="test", help="WandB project name")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="skrl",
        choices=["skrl", "rl_games", "rsl_rl"],
        help="Reinforcement learning library to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    robot_name = args.robot
    task_name = args.task
    num_seeds = args.num_seeds
    algorithm = args.algorithm
    wandb_project = args.wandb_project
    rl_library = args.rl_library

    # check args validity
    valid_robots = ["FloatingPlatform", "Kingfisher", "Turtlebot2", "Jetbot", "Leatherback"]
    valid_tasks = ["GoToPosition", "TrackVelocities", "GoToPose", "GoThroughPositions"]
    valid_rl_libraries = ["skrl", "rl_games", "rsl_rl"]
    if robot_name not in valid_robots:
        raise ValueError(f"Invalid robot name: {robot_name}. Valid options are: {valid_robots}")
    if task_name not in valid_tasks:
        raise ValueError(f"Invalid task name: {task_name}. Valid options are: {valid_tasks}")
    if rl_library not in valid_rl_libraries:
        raise ValueError(f"Invalid RL library: {rl_library}. Valid options are: {valid_rl_libraries}")

    # Define the base command
    base_command = [
        "./isaaclab.sh",
        "-p",
        f"scripts/reinforcement_learning/{rl_library}/train.py",
        "--task",
        "Isaac-RANS-Single-v0",
        "--num_envs",
        "4096",
        f"env.robot_name={robot_name}",
        f"env.task_name={task_name}",
        "--headless",
        "--algorithm",
        algorithm,
        "--wandb_project",
        wandb_project,
        "--wandb_entity",
        "spacer-rl",
    ]

    # Generate and execute commands for each seed
    for seed in range(num_seeds):
        # Set the random seed in the command
        command = base_command + ["--seed", str(seed)]

        if task_name == "GoThroughPositions":
            command += [f"env.episode_length_s={str(50.0)}"]
        # Print the command for debugging
        print(f"Executing command: {' '.join(command)}")

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing command: {e}")
            continue

        # Optional: Sleep for a short duration to avoid overwhelming the system
        time.sleep(1)


if __name__ == "__main__":
    main()


# Sample command to run the script:
# python scripts/reinforcement_learning/train_multi.py --robot FloatingPlatform --task GoToPosition --num_seeds 5 --algorithm ppo-discrete --wandb_project test --rl_library skrl
