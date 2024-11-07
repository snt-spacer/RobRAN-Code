# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence

import logger
import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, Usd, UsdGeom, Vt
from usdrt import Gf as GfRT

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    StaticColliderObject,
    StaticColliderObjectCfg,
)
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers  # , StaticColliders
from omni.isaac.lab.markers import (  # isort: skip
    ARROW_CFG,
    BICOLOR_DIAMOND_CFG,
    DIAMOND_CFG,
    GATE_2D_CFG,
    GATE_3D_CFG,
    GATE_PYLONS_CFG,
    PIN_ARROW_CFG,
    PIN_DIAMOND_CFG,
    PIN_SPHERE_CFG,
    POSE_MARKER_3D_CFG,
    POSITION_MARKER_3D_CFG,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import get_yaw_from_quat, quat_from_angle_axis, sample_uniform

from omni.isaac.lab_assets.leatherback import LEATHERBACK_CFG  # isort: skip

from omni.isaac.lab.sim.spawners.racing_shapes.racing_shapes_cfg import Gate3DCfg, Gate2DCfg  # isort: skip




@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    steering_scale = math.pi / 4.0  # [rad]
    throttle_scale = 20.0  # [rad/s]
    num_actions = 2
    num_observations = 6
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    # colliders_cfg = StaticColliderObjectCfg(
    #    prim_path="/World/envs/env_.*/StaticColliders",
    #    spawn=Gate2DCfg(
    #        width=1.0,
    #        height=1.0,
    #        depth=0.05,
    #        thickness=0.05,
    #        corner_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #        front_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    #        back_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
    #    ),
    #    init_state=StaticColliderObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )
    # colliders_cfg = RigidObjectCfg(
    #    prim_path="/World/envs/env_.*/StaticColliders",
    #    spawn=Gate2DCfg(
    #        width=1.0,
    #        height=1.0,
    #        depth=0.05,
    #        thickness=0.05,
    #        corner_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #        front_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    #        back_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False),
    #    ),
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )

    p = []
    q = []
    for i in range(32):
        theta = 2 * math.pi / 32
        r = 3.0
        x = math.cos(i * theta)
        y = math.sin(i * theta)
        p.append([x * r, y * r, 0.0])
        theta = (theta) * i + math.pi / 2
        q.append([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])
    is_ = StaticColliderObjectCfg.InitialStateCfg(pos=[p], rot=[q])
    colliders_cfg = StaticColliderObjectCfg(
        prim_path="/World/envs/env_.*/StaticColliders",
        spawn=sim_utils.ManyAssetSpawnerCfg(
            assets_cfg=[
                Gate2DCfg(
                    width=1.0,
                    height=1.0,
                    depth=0.05,
                    thickness=0.05,
                    corner_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    front_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    back_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            ],
            num_assets=[32],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False),
        ),
        init_state=is_,
    )

    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left",
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # reset
    max_car_pos = 10.0  # the car is reset if it exceeds that position [m]
    initial_car_dist_range = [-0.001, 0.0001]  # the range in which the car is sampled from on reset [m]

    # reward scales
    rew_scale_pos = 1.0
    rew_coeff_pos = 1.0
    rew_scale_terminated = -1.0
    rew_action_rate_scale = -0.05
    rew_joint_accel_scale = -2.5e-7


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

        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        self.throttle_scale = self.cfg.throttle_scale
        self.steering_scale = self.cfg.steering_scale

        self.body_pos = self.leatherback.data.root_pos_w
        self.body_quat = self.leatherback.data.root_quat_w
        self.body_vel = self.leatherback.data.root_lin_vel_b  # Compute
        self.body_ang_vel = self.leatherback.data.root_ang_vel_b

        self.target = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

    def _setup_scene(self):
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.colliders = StaticColliderObject(self.cfg.colliders_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.make_markers()
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["leatherback"] = self.leatherback
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self.throttle_action = (
            actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self.throttle_scale * 0 + 0.5 * self.throttle_scale
        )
        self.steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self.steering_scale * 0

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self.throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self.steering_action, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        body_state = self.leatherback.data.root_state_w
        dist = torch.norm(body_state[:, :2] - self.target, dim=1)
        heading = get_yaw_from_quat(body_state[:, 3:7])
        target_heading = torch.atan2(self.target[:, 1] - body_state[:, 1], self.target[:, 0] - body_state[:, 0])
        heading_error = target_heading - heading
        vel_norm = torch.norm(body_state[:, 7:9], dim=1)

        obs = torch.cat(
            (
                dist.unsqueeze(dim=1),
                heading_error.cos().unsqueeze(dim=1),
                heading_error.sin().unsqueeze(dim=1),
                (vel_norm * heading.cos()).unsqueeze(dim=1),
                (vel_norm * heading.sin()).unsqueeze(dim=1),
                body_state[:, -1].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pos,
            self.cfg.rew_coeff_pos,
            self.cfg.rew_action_rate_scale,
            self._actions,
            self._previous_actions,
            self.body_pos[:, :2],
            self.target,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.body_pos = self.leatherback.data.root_pos_w
        dist = torch.norm(self.body_pos[:, :2] - self.target, dim=1)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = dist > self.cfg.max_car_pos
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if (env_ids is None) or (len(env_ids) == self.num_envs):
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_envs = len(env_ids)

        default_root_state = self.leatherback.data.default_root_state[env_ids]

        # Reset the car to a random pose
        r = sample_uniform(
            self.cfg.initial_car_dist_range[0],
            self.cfg.initial_car_dist_range[1],
            size=(num_envs, 1),
            device=self.device,
        )
        theta = sample_uniform(-math.pi, math.pi, size=(num_envs, 1), device=self.device) * 0
        px = r * torch.cos(theta)
        py = r * torch.sin(theta)
        default_root_state[:, :2] = torch.cat((px, py), dim=1) + self.scene.env_origins[env_ids, :2]
        default_root_state[:, 2] = 0.2

        theta = sample_uniform(-math.pi, math.pi, size=(num_envs), device=self.device) * 0.0 + math.pi / 2.0
        axis = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        axis[:, 2] = 1.0
        default_root_state[:, 3:7] = quat_from_angle_axis(theta, axis)

        # Set the car joint states to zero
        joint_pos = torch.zeros((num_envs, self.leatherback.num_joints), dtype=torch.float32, device=self.device)
        joint_vel = torch.zeros((num_envs, self.leatherback.num_joints), dtype=torch.float32, device=self.device)

        self.leatherback.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.leatherback.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.leatherback.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Set the target to a random position
        self.target[env_ids] = (
            sample_uniform(
                self.cfg.initial_car_dist_range[0] / 2,
                self.cfg.initial_car_dist_range[1] / 2,
                (num_envs, 2),
                device=self.device,
            )
            + self.scene.env_origins[env_ids, :2]
        )
        self.update_markers()
        num_instances = self.colliders.num_instances
        print(num_instances)
        xyz_wxyz = torch.zeros((num_instances, 7), dtype=torch.float32, device=self.device)
        xyz_wxyz[:, 3] = 1.0
        xyz_wxyz[:, 0] = torch.arange(num_instances, dtype=torch.float32, device=self.device) * 0.2
        self.colliders.write_root_pose_to_sim(
            root_pose=xyz_wxyz, env_ids=torch.arange(num_instances, dtype=torch.int32, device=self.device)
        )

    def make_markers(self):
        # create markers if necessary for the first tome
        if not hasattr(self, "goal_pos_visualizer"):
            # marker_cfg = PIN_SPHERE_CFG.copy()
            marker_cfg = DIAMOND_CFG.copy()
            # -- goal pose
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
        # set their visibility to true
        self.goal_pos_visualizer.set_visibility(True)

    def update_markers(self):
        # pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # pos[:, :2] = self.target
        pos = 2.5 - torch.rand((200, 3), dtype=torch.float32, device=self.device) * 5
        pos[:, 2] = 0
        # update the markers
        # self.goal_pos_visualizer.instantiate()
        self.goal_pos_visualizer.visualize(pos)


@torch.jit.script
def compute_rewards(
    rew_scale_terminated: float,
    rew_pos_scale: float,
    rew_pos_coeff: float,
    rew_action_rate_scale: float,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    action_rate = torch.sum(torch.square(actions - previous_actions), dim=1) * rew_action_rate_scale
    position_error = torch.norm(pos - target, dim=1)
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_position = torch.exp(-position_error / rew_pos_coeff) * rew_pos_scale
    total_reward = rew_termination + rew_position + action_rate
    return total_reward
