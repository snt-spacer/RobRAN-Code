# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from skrl."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    # choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=512,
    help="The maximum number of steps in an episode.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.rans.utils.performance_evaluator_v2 import PerformanceEvaluatorV2
from isaaclab_tasks.rans.utils.plot_eval_multi import plot_episode_data_virtual
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    experiment_cfg = agent_cfg

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # specify directory for logging experiments (load checkpoint)
    if "wandb" in experiment_cfg["agent"]["experiment"]:
        experiment_cfg["agent"]["experiment"]["wandb"] = False
    if "Single" in args_cli.task:
        experiment_cfg["agent"]["experiment"]["directory"] = "Single"
        experiment_cfg["agent"]["experiment"]["experiment_name"] = env.env.cfg.robot_name + "-" + env.env.cfg.task_name
        log_root_path = os.path.join("logs", "skrl", args_cli.task.split("-")[2])
    else:
        experiment_cfg["agent"]["experiment"]["experiment_name"] = args_cli.task.split("-")[2]
        log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    # log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if "Single" in args_cli.task:
        task_name = env.env.cfg.task_name
    else:
        task_name = args_cli.task.split("-")[2]

    print_all_agents = False

    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # Declare dictionary to store obs, actions, and rewards
    ep_data = {"act": [], "obs": [], "rews": [], "dones": [], "terminations": []}

    # #if horizon is an argument, use it, otherwise use 250
    # if hasattr(env.env.cfg, "horizon"):
    #     horizon = env.env.cfg.horizon
    # else:
    horizon = args_cli.horizon if args_cli.horizon is not None else 512

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    # while simulation_app.is_running():
    print("Evaluation started over ", horizon, " steps for ", args_cli.num_envs, " environments.")

    for _ in range(horizon):  # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            # env stepping
            obs, rews, dones, terminations, _ = env.step(actions)
            ep_data["act"].append(actions.cpu().numpy())
            ep_data["obs"].append(obs.cpu().numpy())
            ep_data["rews"].append(rews.cpu().numpy())
            ep_data["dones"].append(dones.cpu().numpy())
            ep_data["terminations"].append(terminations.cpu().numpy())

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # Convert data to numpy arrays
    ep_data["rews"] = np.array(ep_data["rews"]).squeeze(axis=-1)
    ep_data["obs"], ep_data["rews"], ep_data["act"] = map(np.array, (ep_data["obs"], ep_data["rews"], ep_data["act"]))

    save_dir = os.path.join(log_root_path, log_dir, f"eval_{args_cli.num_envs}_envs", task_name)
    print("Saving plots in ", save_dir)
    # Plot the episode data
    if print_all_agents:
        print("Plotting data for all agents.")
        plot_episode_data_virtual(
            ep_data,
            save_dir=save_dir,
            task=task_name,
            all_agents=print_all_agents,
        )
    # Run performance evaluation
    # evaluator = PerformanceEvaluator(task_name, env.env.cfg.robot_name, ep_data, horizon)
    # evaluator = PerformanceMetrics(
    #     task_name, env.env.cfg.robot_name, ep_data, horizon, plot_metrics=False, save_path=save_dir
    # )
    # evaluator.compute_basic_metrics()
    # results = evaluator.evaluate()
    # print_dict(results, nesting=4)
    robot_name = env.env.cfg.robot_name

    combo_id = f"{robot_name}_{task_name}_skrl"  # lib is 'skrl' or 'rlgames'
    evaluator = PerformanceEvaluatorV2(task_name, robot_name, "skrl", ep_data, horizon, combo_id, seed=0)
    metrics = evaluator.evaluate()
    evaluator.save_csv()  # writes per-run CSV
    evaluator.export_timeseries_metrics()

    print_dict(metrics, nesting=4)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
