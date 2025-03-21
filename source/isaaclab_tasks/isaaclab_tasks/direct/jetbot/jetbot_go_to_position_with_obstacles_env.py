# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_tasks.rans import GoToPositionWithObstaclesCfg, GoToPositionWithObstaclesTask, JetbotRobot, JetbotRobotCfg

from .jetbot_go_to_position_env import JetbotGoToPositionEnv, JetbotGoToPositionEnvCfg


@configclass
class JetbotGoToPositionWithObstaclesEnvCfg(JetbotGoToPositionEnvCfg):

    task_cfg: GoToPositionWithObstaclesCfg = GoToPositionWithObstaclesCfg()
    robot_cfg: JetbotRobotCfg = JetbotRobotCfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=15, replicate_physics=True)

    observation_space = robot_cfg.observation_space + task_cfg.observation_space


class JetbotGoToPositionWithObstaclesEnv(JetbotGoToPositionEnv):
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

    cfg: JetbotGoToPositionWithObstaclesEnvCfg

    def __init__(
        self,
        cfg: JetbotGoToPositionWithObstaclesEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg.robot_cfg)
        self.robot_api = JetbotRobot(
            scene=self.scene, robot_cfg=self.cfg.robot_cfg, robot_uid=0, num_envs=self.num_envs, device=self.device
        )
        self.task_api = GoToPositionWithObstaclesTask(
            scene=self.scene, task_cfg=self.cfg.task_cfg, task_uid=0, num_envs=self.num_envs, device=self.device
        )

        self.task_api.register_robot(self.robot_api)
        self.task_api.register_sensors()

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
