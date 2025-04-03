# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse  # To get task from command line
import matplotlib.pyplot as plt
import os
from glob import glob

import pandas as pd
import seaborn as sns

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Plot training performance comparison")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g., 'TrackVelocities'")
args = parser.parse_args()

task = args.task  # Get task from command line
folder_path = f"./learning_data/{task}"  # Path to CSV folder
save_path = os.path.join(folder_path, "results")
os.makedirs(save_path, exist_ok=True)

# Get all CSV files in the folder
all_files = glob(os.path.join(folder_path, "*.csv"))

# Seaborn color palette
sns.set_style("whitegrid")
palette = sns.color_palette("Paired", n_colors=6)

# Assign fixed colors
robot_colors = {
    "kingfisher": palette[1],
    "floatingplatform": palette[5],
    "turtlebot2": palette[3],
}

robot_data = {}

for file_path in all_files:
    file_name = os.path.basename(file_path)
    robot_name = file_name.split("_")[0]

    # Read CSV
    df = pd.read_csv(file_path)

    reward_cols = [col for col in df.columns if "Reward / Total reward (mean)" in col and "__" not in col]

    # Compute Mean and Std across seeds
    mean_rewards = df[reward_cols].mean(axis=1)
    std_rewards = df[reward_cols].std(axis=1)
    steps = df["Step"]

    # Store data
    robot_data[robot_name] = {"steps": steps, "mean": mean_rewards, "std": std_rewards}

# Plot
plt.figure(figsize=(10, 5))

for robot, data in robot_data.items():
    sns.lineplot(x=data["steps"], y=data["mean"], label=f"{robot}", color=robot_colors.get(robot, "black"))
    plt.fill_between(
        data["steps"],
        data["mean"] - data["std"],
        data["mean"] + data["std"],
        color=robot_colors.get(robot, "black"),
        alpha=0.2,
    )

plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Mean Reward", fontsize=14)
plt.title(f"Training Performance Comparison: {task} Task", fontsize=16)
plt.legend(title="Robots", fontsize=12, title_fontsize=14, loc="best")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

# **Save the plot with tighter layout and no white spacing**
plot_filename = os.path.join(save_path, f"training_comparison_{task}.png")
plt.savefig(plot_filename, dpi=200, bbox_inches="tight", pad_inches=0)
plt.close()

print(f"Plot saved at: {plot_filename}")
