# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.rans import JetbotRobot, JetbotRobotCfg, TrackVelocitiesCfg, TrackVelocitiesTask


@configclass
class JetbotTrackVelocitiesEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=7.5, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: JetbotRobotCfg = JetbotRobotCfg()
    task_cfg: TrackVelocitiesCfg = TrackVelocitiesCfg()
    debug_vis: bool = True

    action_space = robot_cfg.action_space + task_cfg.action_space
    observation_space = robot_cfg.observation_space + task_cfg.observation_space
    state_space = robot_cfg.state_space + task_cfg.state_space
    gen_space = robot_cfg.gen_space + task_cfg.gen_space


class JetbotTrackVelocitiesEnv(DirectRLEnv):
    # Workflow: Step
    #   - self._pre_physics_step
    #   - (Loop over N skipped steps)
    #       - self._apply_actions
    #       - self.scene.write_data_to_sim()
    #       - self.sim.step(render=False)
    #       - (Check if rendering is required)
    #           - self.sim.render()
    #       - self.scene.update()
    #   - self._get_dones
    #   - self._get_rewards
    #   - (Check if reset is required)
    #       - self._reset_idx
    #       - (Check if RTX sensors)
    #           - self.scene.render()
    #   - (Check for events)
    #       - self.event_manager.apply()
    #   - self._get_observations
    #   - (Check if noise is required)
    #       - self._add_noise

    cfg: JetbotTrackVelocitiesEnvCfg

    def __init__(
        self,
        cfg: JetbotTrackVelocitiesEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.env_seeds = torch.randint(0, 100000, (self.num_envs,), dtype=torch.int32, device=self.device)
        self.robot_api.run_setup(self.robot)
        self.task_api.run_setup(self.robot_api, self.scene.env_origins)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg.robot_cfg)
        self.robot_api = JetbotRobot(self.cfg.robot_cfg, robot_uid=0, num_envs=self.num_envs, device=self.device)
        self.task_api = TrackVelocitiesTask(self.cfg.task_cfg, task_uid=0, num_envs=self.num_envs, device=self.device)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["jetbot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_api.process_actions(actions)

    def _apply_action(self) -> None:
        self.robot_api.apply_actions()

    def _get_observations(self) -> dict:
        task_obs = self.task_api.get_observations()
        observations = {"policy": task_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.task_api.compute_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_early_termination, robot_clean_termination = self.robot_api.get_dones()
        task_early_termination, task_clean_termination = self.task_api.get_dones()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        early_termination = robot_early_termination | task_early_termination
        clean_termination = robot_clean_termination | task_clean_termination | time_out
        return early_termination, clean_termination

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if (env_ids is None) or (len(env_ids) == self.num_envs):
            env_ids = self.robot._ALL_INDICES

        # Logging
        self.task_api.reset_logs(env_ids, self.episode_length_buf)
        task_extras = self.task_api.compute_logs()
        self.robot_api.reset_logs(env_ids, self.episode_length_buf)
        robot_extras = self.robot_api.compute_logs()
        self.extras["log"] = dict()
        self.extras["log"].update(task_extras)
        self.extras["log"].update(robot_extras)

        super()._reset_idx(env_ids)

        self.task_api.reset(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            self.task_api.create_task_visualization()

    def _debug_vis_callback(self, event) -> None:
        if self.cfg.debug_vis:
            self.task_api.update_task_visualization()
