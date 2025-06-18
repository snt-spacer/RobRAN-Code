# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

from navbench_plot import plot_radar, plot_ribbon_tracking, plot_violin_goals

robots = ["Turtlebot2", "Kingfisher", "FloatingPlatform"]
tasks = ["GoToPosition", "GoToPose", "GoThroughPositions", "TrackVelocities"]
metrics_per_task = {
    "GoToPosition": ["success_rate", "final_distance_error", "avg_time_to_target", "control_variation"],
    "GoToPose": ["success_rate", "heading_error", "final_distance_error", "avg_time_to_target", "control_variation"],
    "GoThroughPositions": ["num_goals_reached", "final_distance_error", "control_variation"],
    "TrackVelocities": ["tracking_error", "control_variation", "success_rate"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--out-dir", type=str, default="figs")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

for robot in robots:
    for task in tasks:
        combo = f"{robot}_{task}"
        metrics = metrics_per_task[task]
        title = f"{robot} – {task}"

        try:
            radar_out = os.path.join(args.out_dir, f"{combo}_radar.png")
            plot_radar(combo, metrics, ["skrl", "rl_games"], title, args.results_dir, radar_out)
        except Exception as e:
            print(f"⚠️ Radar plot error for {combo}: {e}")

        if task == "GoThroughPositions":
            try:
                violin_out = os.path.join(args.out_dir, f"{combo}_goals_violin.png")
                plot_violin_goals(combo, title, args.results_dir, violin_out)
            except Exception as e:
                print(f"⚠️ Violin plot error for {combo}: {e}")

        if task == "TrackVelocities":
            try:
                ribbon_out = os.path.join(args.out_dir, f"{combo}_track_vel.png")
                plot_ribbon_tracking(combo, title, args.results_dir, ribbon_out)
            except Exception as e:
                print(f"⚠️ Ribbon plot error for {combo}: {e}")
