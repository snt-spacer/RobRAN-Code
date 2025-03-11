# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.utils.spaces import sample_space
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.rans import ROBOT_CFG_FACTORY, ROBOT_FACTORY, TASK_CFG_FACTORY, TASK_FACTORY


@configclass
class SingleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 6
    episode_length_s = 20.0

    robot_name = "Leatherback"
    task_name = "GoToPosition"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=7.5, replicate_physics=True)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)
    # Simulation
    # sim = SimulationCfg(
    #     disable_contact_processing=True,
    #     physx=sim_utils.PhysxCfg(
    #         enable_ccd=True,
    #         enable_stabilization=True,
    #         bounce_threshold_velocity=0.0,
    #         friction_correlation_distance=0.005,
    #         min_velocity_iteration_count=2,
    #         # GPU settings
    #         gpu_temp_buffer_capacity=2 ** (24 - 4),
    #         gpu_max_rigid_contact_count=2 ** (22 - 5),
    #         gpu_max_rigid_patch_count=2 ** (13 - 3),
    #         gpu_heap_capacity=2 ** (26 - 3),
    #         gpu_found_lost_pairs_capacity=2 ** (18 - 3),
    #         gpu_found_lost_aggregate_pairs_capacity=2 ** (10 - 2),
    #         gpu_total_aggregate_pairs_capacity=2 ** (10 - 2),
    #         gpu_max_soft_body_contacts=2 ** (20 - 5),
    #         gpu_max_particle_contacts=2 ** (20 - 5),
    #         gpu_collision_stack_size=2 ** (26 - 5),
    #         gpu_max_num_partitions=8,
    #     ),
    #     render=sim_utils.RenderCfg(
    #         enable_reflections=True,
    #     ),
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #     ),
    # )
    debug_vis: bool = True

    action_space = 0
    observation_space = 0
    state_space = 0
    gen_space = 0


class SingleEnv(DirectRLEnv):

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

    cfg: SingleEnvCfg

    def __init__(
        self,
        cfg: SingleEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg = self.edit_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self.env_seeds = torch.randint(0, 100000, (self.num_envs,), dtype=torch.int32, device=self.device)
        self.robot_api.run_setup(self.robot)
        self.task_api.run_setup(self.robot_api, self.scene.env_origins)
        self.set_debug_vis(self.cfg.debug_vis)
        self.task_api.register_rigid_objects()

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        super()._configure_gym_env_spaces()
        self.single_action_space, self.action_space = self.robot_api.configure_gym_env_spaces()
        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

    def edit_cfg(self, cfg: SingleEnvCfg) -> SingleEnvCfg:
        self.robot_cfg = ROBOT_CFG_FACTORY(cfg.robot_name)
        self.task_cfg = TASK_CFG_FACTORY(cfg.task_name)

        cfg.action_space = self.robot_cfg.action_space + self.task_cfg.action_space
        cfg.observation_space = self.robot_cfg.observation_space + self.task_cfg.observation_space
        cfg.state_space = self.robot_cfg.state_space + self.task_cfg.state_space
        cfg.gen_space = self.robot_cfg.gen_space + self.task_cfg.gen_space
        return cfg

    def _setup_scene(self):
        self.robot = Articulation(self.robot_cfg.robot_cfg)
        self.robot_api = ROBOT_FACTORY(
            self.cfg.robot_name,
            scene=self.scene,
            robot_cfg=self.robot_cfg,
            robot_uid=0,
            num_envs=self.num_envs,
            decimation=self.cfg.decimation,
            device=self.device,
        )
        self.task_api = TASK_FACTORY(
            self.cfg.task_name,
            scene=self.scene,
            task_cfg=self.task_cfg,
            task_uid=0,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.task_api.register_robot(self.robot_api)
        self.task_api.register_sensors()

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations[self.cfg.robot_name] = self.robot
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
