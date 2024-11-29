# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.assets import Articulation

from omni.isaac.lab_tasks.rans import LeatherbackRobotCfg

from .robot_core import RobotCore


class LeatherbackRobot(RobotCore):
    def __init__(
        self,
        robot_cfg: LeatherbackRobotCfg,
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
        self._throttle_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._steering_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._throttle_dof_idx, _ = self._robot.find_joints(self._robot_cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self._robot.find_joints(self._robot_cfg.steering_dof_name)

    def create_logs(self):
        super().create_logs()

        def torch_zeros():
            return torch.zeros(
                self._num_envs,
                dtype=torch.float32,
                device=self._device,
                requires_grad=False,
            )

        state_keys = ["AVG/throttle_action", "AVG/steering_action", "AVG/action_rate", "AVG/joint_acceleration"]
        reward_keys = ["AVG/action_rate", "AVG/joint_acceleration"]

        # Populate dictionaries with torch_zeros()
        for key in state_keys:
            self._step_logs["robot_state"][key] = torch_zeros()
            self._episode_logs["robot_state"][key] = torch_zeros()

        for key in reward_keys:
            self._step_logs["robot_reward"][key] = torch_zeros()
            self._episode_logs["robot_reward"][key] = torch_zeros()

        self._average_logs["robot_state"]["AVG/throttle_action"] = True
        self._average_logs["robot_state"]["AVG/steering_action"] = True
        self._average_logs["robot_state"]["AVG/action_rate"] = True
        self._average_logs["robot_state"]["AVG/joint_acceleration"] = True
        self._average_logs["robot_reward"]["AVG/action_rate"] = True
        self._average_logs["robot_reward"]["AVG/joint_acceleration"] = True

    def get_observations(self) -> torch.Tensor:
        return self._actions

    def compute_rewards(self):
        # TODO: DT should be factored in?

        # Compute
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self._step_logs["robot_state"]["AVG/action_rate"] = action_rate
        self._step_logs["robot_state"]["AVG/joint_acceleration"] = joint_accelerations
        self._step_logs["robot_reward"]["AVG/action_rate"] = action_rate
        self._step_logs["robot_reward"]["AVG/joint_acceleration"] = joint_accelerations
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

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None):
        throttle_reset = torch.zeros(
            (len(env_ids), self._throttle_action.shape[1]),
            device=self._device,
            dtype=torch.float32,
        )
        steering_reset = torch.zeros(
            (len(env_ids), self._steering_action.shape[1]),
            device=self._device,
            dtype=torch.float32,
        )
        self._robot.set_joint_velocity_target(throttle_reset, joint_ids=self._throttle_dof_idx, env_ids=env_ids)
        self._robot.set_joint_position_target(steering_reset, joint_ids=self._steering_dof_idx, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self._robot_cfg.throttle_scale
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self._robot_cfg.steering_scale

        # Log data
        self._step_logs["robot_state"]["AVG/throttle_action"] = self._throttle_action[:, 0]
        self._step_logs["robot_state"]["AVG/steering_action"] = self._steering_action[:, 0]

    def compute_physics(self):
        pass  # Model motor + ackermann steering here

    def apply_actions(self):
        self._robot.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self._robot.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)
