# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from gymnasium import spaces, vector

from omni.isaac.lab.assets import Articulation

from omni.isaac.lab_tasks.rans import JetbotRobotCfg

from .robot_core import RobotCore


class JetbotRobot(RobotCore):
    def __init__(
        self,
        robot_cfg: JetbotRobotCfg = JetbotRobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        super().__init__(robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = self._robot_cfg.observation_space
        self._dim_robot_act = self._robot_cfg.action_space
        self._dim_gen_act = self._robot_cfg.gen_space

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act),
            device=self._device,
            dtype=torch.float32,
        )
        self.left_wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self.right_wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._wheels_dof_idx, _ = self._robot.find_joints(self._robot_cfg.wheels_dof_names)

    def create_logs(self):
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/left_wheel_action", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/right_wheel_action", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/joint_acceleration", "mean")

    def get_observations(self) -> torch.Tensor:
        return self._actions

    def compute_rewards(self):
        # Compute
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self.scalar_logger.log("robot_state", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations)

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
        wheels_reset = torch.zeros(
            (len(env_ids), self._dim_robot_act),
            device=self._device,
            dtype=torch.float32,
        )
        self._robot.set_joint_velocity_target(wheels_reset, joint_ids=self._wheels_dof_idx, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        self.left_wheel_action = actions[:, 0] * self._robot_cfg.wheel_scale
        self.right_wheel_action = actions[:, 1] * self._robot_cfg.wheel_scale

        # Log data
        self.scalar_logger.log("robot_state", "AVG/left_wheel_action", self.left_wheel_action)
        self.scalar_logger.log("robot_state", "AVG/right_wheel_action", self.right_wheel_action)

    def compute_physics(self):
        pass  # Model motor

    def apply_actions(self):
        wheel_action = torch.cat((self.left_wheel_action.unsqueeze(-1), self.right_wheel_action.unsqueeze(-1)), dim=1)
        self._robot.set_joint_velocity_target(wheel_action, joint_ids=self._wheels_dof_idx)

    def configure_gym_env_spaces(self):
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = vector.utils.batch_space(single_action_space, self._num_envs)

        return single_action_space, action_space
