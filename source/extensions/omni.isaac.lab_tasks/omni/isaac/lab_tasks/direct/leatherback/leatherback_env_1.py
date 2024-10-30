# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, get_yaw_from_quat, quat_from_angle_axis
from omni.isaac.lab_tasks.rans import (
    LeatherbackRobotCfg,
    GoToPoseCfg,
    LeatherbackRobot,
    GoToPoseTask,
    GoToPositionCfg,
    GoToPositionTask,
    TrackVelocityCfg,
    TrackVelocityTask,
    GoThroughPositionsCfg,
    GoThroughPositionsTask,
)

import omni
import numpy as np
from usdrt import Usd as UsdRT
from usdrt import UsdGeom as UsdGeomRT
from usdrt import Sdf as SdfRT
from usdrt import UsdPhysics as UsdPhysicsRT
from usdrt import PhysxSchema as PhysxSchemaRT
from usdrt import Vt as VtRT
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf


from omni.isaac.lab.static_colliders import StaticCollidersRTCfg, StaticCollidersRT


@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    num_actions = 2
    num_observations = 11
    num_states = 0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=20.0, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: LeatherbackRobotCfg = LeatherbackRobotCfg()
    task_cfg: GoThroughPositionsCfg = GoThroughPositionsCfg()
    debug_vis: bool = True


class LeatherbackEnv(DirectRLEnv):
    # Workflow: Step
    #   - self._pre_physics_step
    #   - (Loop over N skiped steps)
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

    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.env_seeds = torch.randint(0, 100000, (self.num_envs,), dtype=torch.int32, device=self.device)
        self.robot_api.run_setup(self.robot)
        self.task_api.run_setup(self.robot, self.scene.env_origins)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg.robot_cfg)
        self.robot_api = LeatherbackRobot(self.cfg.robot_cfg, robot_uid=0, num_envs=self.num_envs, device=self.device)
        self.task_api = GoThroughPositionsTask(
            self.cfg.task_cfg, task_uid=0, num_envs=self.num_envs, device=self.device
        )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["leatherback"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.robot_api.process_actions(actions)

    def _apply_action(self) -> None:
        self.robot_api.apply_actions(self.robot)

    def _get_observations(self) -> dict:
        robot_obs = self.robot_api.get_observations(self.robot.data)
        task_obs = self.task_api.get_observations()

        obs = torch.cat(
            (
                robot_obs,
                task_obs,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        task_rewards = self.task_api.compute_rewards()
        robot_rewards = self.robot_api.compute_rewards(self.robot.data)
        return task_rewards + robot_rewards

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
        super()._reset_idx(env_ids)

        task_actions = torch.rand((len(env_ids), self.task_api.num_actions), device=self.device)
        self.robot_api.reset(None, self.env_seeds[env_ids], self.robot, env_ids)
        self.task_api.reset(task_actions, self.env_seeds[env_ids], env_ids)
        self._pos = np.random.rand(3000).reshape(-1, 3) * 30
        self._pos[:, 2] = 0

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            self.task_api.create_task_visualization()

            # diamond = sim_utils.DiamondCfg(
            #    diamond_height=0.6,
            #    diamond_width=0.4,
            #    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            # )
            # marker_cfg = StaticCollidersRTCfg(colliders={"goal_pos": diamond})
            # -- goal pose
            # marker_cfg.prim_path = "/World/envs/goal_position"
            # self.goal_pos_visualizer = StaticCollidersRT(marker_cfg)
            # self.goal_pos_visualizer.set_visibility(True)

            stage_rt = UsdRT.Stage.Attach(omni.usd.get_context().get_stage_id())
            stage = omni.usd.get_context().get_stage()
            # Create the sphere and instancer in the native USD pipeline
            if not stage.GetPrimAtPath("/World/Instancer"):
                print("Creating stuff")
                instancer_manager = UsdGeom.PointInstancer.Define(stage, "/World/Instancer")
                sphere_usd = UsdGeom.Sphere.Define(stage, "/World/Instancer/Sphere")
                prim = stage.GetPrimAtPath("/World/Instancer/Sphere")
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                collision_api.CreateCollisionEnabledAttr().Set(True)
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                instancer_manager.GetPrototypesRel().AddTarget("/World/Instancer/Sphere")
                instancer_manager.GetProtoIndicesAttr().Set(list(range(1)))
                instancer_manager.GetPositionsAttr().Set([Gf.Vec3f(0.0)] * 1)
            # Modify in Fabric
            self.instancer = stage_rt.GetPrimAtPath("/World/Instancer")
            self.instancer = UsdGeomRT.PointInstancer(self.instancer)
            pos = np.random.rand(3000).reshape(-1, 3) * 30
            pos[:, 2] = 0
            self._idx = np.zeros((pos.shape[0],), dtype=int)
            self.instancer.GetPositionsAttr().Set(VtRT.Vec3fArray(pos))
            self.instancer.GetProtoIndicesAttr().Set(VtRT.IntArray(self._idx))
            print("I was called! Did something change?")
            stage_rt.WriteToStage()

    def _debug_vis_callback(self, event) -> None:
        if self.cfg.debug_vis:
            self.task_api.update_task_visualization()
            self.instancer.GetPositionsAttr().Set(VtRT.Vec3fArray(self._pos))
            self.instancer.GetProtoIndicesAttr().Set(VtRT.IntArray(self._idx))
            stage_rt = UsdRT.Stage.Attach(omni.usd.get_context().get_stage_id())
            stage_rt.WriteToStage()

            # self.goal_pos_visualizer.visualize(self._pos)
