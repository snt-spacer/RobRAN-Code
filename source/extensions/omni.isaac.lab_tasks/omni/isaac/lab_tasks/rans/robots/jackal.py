# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.assets import Articulation, ArticulationData

from omni.isaac.lab_tasks.rans import JackalRobotCfg

from .robot_core import RobotCore


class JackalRobot(RobotCore):
    def __init__(
        self,
        robot_cfg: JackalRobotCfg,
        robot_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        super().__init__(robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = 2
        self._dim_robot_act = 2
        self._dim_gen_act = 0

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act),
            device=self._device,
            dtype=torch.float32,
        )
        self._wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._wheels_dof_idx, _ = self._robot.find_joints(self._robot_cfg.wheels_dof_names)

    def create_logs(self):
        super().create_logs()
        torch_zeros = lambda: torch.zeros(
            self._num_envs,
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )
        self._logs["state"]["wheels"] = torch_zeros()
        self._logs["state"]["action_rate"] = torch_zeros()
        self._logs["state"]["joint_acceleration"] = torch_zeros()
        self._logs["reward"]["action_rate"] = torch_zeros()
        self._logs["reward"]["joint_acceleration"] = torch_zeros()

    def get_observations(self) -> torch.Tensor:
        return self._actions

    def compute_rewards(self):
        # Compute
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self._logs["state"]["action_rate"] = action_rate
        self._logs["state"]["joint_acceleration"] = joint_accelerations
        self._logs["reward"]["action_rate"] = action_rate
        self._logs["reward"]["joint_acceleration"] = joint_accelerations
        return (
            action_rate * self._robot_cfg.rew_action_rate_scale
            + joint_accelerations * self._robot_cfg.rew_joint_accel_scale
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        task_done = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        return task_failed, task_done

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ):
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0

    def reset_logs(self, env_ids: torch.Tensor) -> None:
        # Reset logs
        self._logs["state"]["wheels"][env_ids] = 0
        self._logs["state"]["action_rate"][env_ids] = 0
        self._logs["state"]["joint_acceleration"][env_ids] = 0
        self._logs["reward"]["action_rate"][env_ids] = 0
        self._logs["reward"]["joint_acceleration"][env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None):
        wheels_reset = torch.zeros(
            (len(env_ids), self._wheel_action.shape[1]),
            device=self._device,
            dtype=torch.float32,
        )
        self._robot.set_joint_velocity_target(wheels_reset, joint_ids=self._wheels_dof_idx, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        self._wheel_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self._robot_cfg.wheel_scale

        # Log data
        self._logs["state"]["throttle"] = self._wheel_action[:, 0]

    def compute_physics(self):
        pass  # Model motor 

    def apply_actions(self):
        self._robot.set_joint_velocity_target(self._wheel_action, joint_ids=self._wheels_dof_idx)
