# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
from itertools import product

robots = ["FloatingPlatform", "Kingfisher", "Turtlebot2", "Jetbot", "Leatherback"]
tasks = ["GoToPosition", "GoToPose", "GoThroughPositions", "TrackVelocities"]
rl_libs = ["skrl", "rl_games"]

# Root directories
checkpoint_dir = "navbench_models"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Common parameters
horizon = 600
num_envs = 1024

for robot, task, rl_lib in product(robots, tasks, rl_libs):
    combo = f"{robot}_{task}_{rl_lib}"
    # get the right checkpoint path from the checkpoint directory (robot-task-rl_lib)
    checkpoint_path = os.path.join(checkpoint_dir, combo)
    episode_length_s = 60.0 if task == "GoThroughPositions" else 40.0
    horizon = 1000 if task == "GoThroughPositions" else 600

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for combo: {combo}. Skipping...")
        continue
    results_path = os.path.join(results_dir, f"{combo}.csv")

    print(f"Running eval for {combo}...")

    algorithm = "ppo-discrete" if robot == "FloatingPlatform" else "ppo"

    model_dir = "checkpoints/best_agent.pt" if rl_lib == "skrl" else "nn/" + str(robot) + "-" + str(task) + ".pth"
    checkpoint_path = os.path.join(checkpoint_path, model_dir)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found at {checkpoint_path}. Skipping...")
        continue

    base_command = [
        "./isaaclab.sh",
        "-p",
        f"scripts/reinforcement_learning/{rl_lib}/eval.py",
        "--task",
        "Isaac-RANS-Single-v0",
        "--num_envs",
        str(num_envs),
        "--checkpoint",
        checkpoint_path,
        "--headless",
        "--algorithm",
        algorithm,
        "--horizon",
        str(horizon),
        f"env.robot_name={robot}",
        f"env.task_name={task}",
        f"env.episode_length_s={str(episode_length_s)}",
        # "--save_metrics_path", results_path,
    ]
    # Execute the command
    try:
        subprocess.run(base_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing command: {e}")
        continue
