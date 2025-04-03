# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import ARROW_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import TrackVelocities3DCfg

from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class TrackVelocities3DTask(TaskCore):
    """
    Implements the TrackVelocity task. The robot has to reach a target velocity.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: TrackVelocities3DCfg = TrackVelocities3DCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Initializes the TrackVelocities task.

        Args:
            scene: Interactive scene containing sim entities for the task.
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            env_ids: The ids of the environments used by this task.
        """

        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers()

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task.

        Returns:
            dict: The dictionary containing the statistics."""

        super().create_logs()

        self.scalar_logger.add_log("task_state", "AVG/absolute_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_lateral_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_vertical_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_yaw_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_pitch_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_roll_velocity", "mean")
        self.scalar_logger.add_log("task_state", "EMA/linear_velocity_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/lateral_velocity_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/vertical_velocity_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/yaw_velocity_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/pitch_velocity_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/roll_velocity_distance", "ema")
        self.scalar_logger.add_log("task_reward", "EMA/linear_velocity", "ema")
        self.scalar_logger.add_log("task_reward", "EMA/lateral_velocity", "ema")
        self.scalar_logger.add_log("task_reward", "EMA/vertical_velocity", "ema")
        self.scalar_logger.add_log("task_reward", "EMA/yaw_velocity", "ema")
        self.scalar_logger.add_log("task_reward", "EMA/pitch_velocity", "ema")
        self.scalar_logger.add_log("task_reward", "EMA/roll_velocity", "ema")
        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        super().initialize_buffers(env_ids)
        # Target velocities
        self._linear_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._lateral_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._vertical_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._yaw_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._pitch_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._roll_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        # Desired velocities
        self._linear_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._lateral_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._vertical_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._yaw_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._pitch_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._roll_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        # Number of steps (used to compute when to change goals)
        self._num_steps = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)
        self._smoothing_factor = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._update_after_n_steps = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        The observation tensor is given in robot's frame. The task provides 5 elements.
        - The linear velocity error in the robot's frame.
        - The lateral velocity error in the robot's frame.
        - The angular velocity error in the robot's frame.
        - The linear velocity of the robot in the robot's frame.
        - The angular velocity of the robot in the robot's frame.

        The observation tensor is composed of the following elements:
        - task_data[:, 0]: The linear velocity error in the robot's frame.
        - task_data[:, 1]: The lateral velocity error in the robot's frame.
        - task_data[:, 2]: The angular velocity error in the robot's frame.
        - task_data[:, 3]: The linear velocity of the robot along the x axis.
        - task_data[:, 4]: The lateral velocity of the robot along the y axis.
        - task_data[:, 5]: The angular velocity of the robot.
        - task_data[:, 6:9]: The linear velocity of the robot in the robot's frame.
        - task_data[:, 9:12]: The angular velocity of the robot in the robot's frame.

        Note: Depending on the task configuration, some of these elements might be disabled, e.g. the lateral velocity
        tracking. Disabling an element will set the corresponding value to 0.

        Returns:
            torch.Tensor: The observation tensor.
        """

        # linear velocity error in the robot's frame
        err_lin_vel = self._linear_velocity_target - self._robot.root_com_lin_vel_b[:, 0]
        # lateral velocity error in the robot's frame
        err_lat_vel = self._lateral_velocity_target - self._robot.root_com_lin_vel_b[:, 1]
        # vertical velocity error in the robot's frame
        err_ver_vel = self._vertical_velocity_target - self._robot.root_com_lin_vel_b[:, 2]
        # yaw velocity error in the robot's frame
        err_yaw_vel = self._yaw_velocity_target - self._robot.root_com_ang_vel_b[:, 2]
        # pitch velocity error in the robot's frame
        err_pitch_vel = self._pitch_velocity_target - self._robot.root_com_ang_vel_b[:, 1]
        # roll velocity error in the robot's frame
        err_roll_vel = self._roll_velocity_target - self._robot.root_com_ang_vel_b[:, 0]

        # Store in buffer
        self._task_data[:, 0] = err_lin_vel * self._task_cfg.enable_linear_velocity
        self._task_data[:, 1] = err_lat_vel * self._task_cfg.enable_lateral_velocity
        self._task_data[:, 2] = err_ver_vel * self._task_cfg.enable_vertical_velocity
        self._task_data[:, 3] = err_yaw_vel * self._task_cfg.enable_yaw_velocity
        self._task_data[:, 4] = err_pitch_vel * self._task_cfg.enable_pitch_velocity
        self._task_data[:, 5] = err_roll_vel * self._task_cfg.enable_roll_velocity
        self._task_data[:, 6:9] = self._robot.root_com_lin_vel_b[self._env_ids]  # Linear velocity (x, y, z)
        self._task_data[:, 9:12] = self._robot.root_com_ang_vel_b[self._env_ids]  # Angular velocity (roll, pitch, yaw)

        # Update logs
        self.scalar_logger.log(
            "task_state", "AVG/absolute_linear_velocity", torch.abs(self._robot.root_com_lin_vel_b[:, 0])
        )
        self.scalar_logger.log(
            "task_state", "AVG/absolute_lateral_velocity", torch.abs(self._robot.root_com_lin_vel_b[:, 1])
        )
        self.scalar_logger.log(
            "task_state", "AVG/absolute_vertical_velocity", torch.abs(self._robot.root_com_lin_vel_b[:, 2])
        )
        self.scalar_logger.log(
            "task_state", "AVG/absolute_yaw_velocity", torch.abs(self._robot.root_com_ang_vel_b[:, 2])
        )
        self.scalar_logger.log(
            "task_state", "AVG/absolute_pitch_velocity", torch.abs(self._robot.root_com_ang_vel_b[:, 1])
        )
        self.scalar_logger.log(
            "task_state", "AVG/absolute_roll_velocity", torch.abs(self._robot.root_com_ang_vel_b[:, 0])
        )

        # Concatenate the task observations with the robot observations
        return torch.concat((self._task_data, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # Linear velocity error
        linear_velocity_distance = torch.abs(self._linear_velocity_target - self._robot.root_com_lin_vel_b[:, 0])
        # Lateral velocity error
        lateral_velocity_distance = torch.abs(self._lateral_velocity_target - self._robot.root_com_lin_vel_b[:, 1])
        # Vertical velocity error
        vertical_velocity_distance = torch.abs(self._vertical_velocity_target - self._robot.root_com_lin_vel_b[:, 2])
        # Yaw velocity error
        yaw_velocity_distance = torch.abs(self._yaw_velocity_target - self._robot.root_com_ang_vel_b[:, 2])
        # Pitch velocity error
        pitch_velocity_distance = torch.abs(self._pitch_velocity_target - self._robot.root_com_ang_vel_b[:, 1])
        # Roll velocity error
        roll_velocity_distance = torch.abs(self._roll_velocity_target - self._robot.root_com_ang_vel_b[:, 0])

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self.scalar_logger.log("task_state", "EMA/linear_velocity_distance", linear_velocity_distance)
        self.scalar_logger.log("task_state", "EMA/lateral_velocity_distance", lateral_velocity_distance)
        self.scalar_logger.log("task_state", "EMA/vertical_velocity_distance", vertical_velocity_distance)
        self.scalar_logger.log("task_state", "EMA/yaw_velocity_distance", yaw_velocity_distance)
        self.scalar_logger.log("task_state", "EMA/pitch_velocity_distance", pitch_velocity_distance)
        self.scalar_logger.log("task_state", "EMA/roll_velocity_distance", roll_velocity_distance)

        # linear velotiy reward
        linear_velocity_rew = torch.exp(
            -linear_velocity_distance / self._task_cfg.lin_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_linear_velocity)
        # lateral velocity reward
        lateral_velocity_rew = torch.exp(
            -lateral_velocity_distance / self._task_cfg.lat_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_lateral_velocity)
        vertical_velocity_rew = torch.exp(
            -vertical_velocity_distance / self._task_cfg.ver_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_vertical_velocity)
        yaw_velocity_rew = torch.exp(-yaw_velocity_distance / self._task_cfg.yaw_vel_exponential_reward_coeff) * int(
            self._task_cfg.enable_yaw_velocity
        )
        pitch_velocity_rew = torch.exp(
            -pitch_velocity_distance / self._task_cfg.pitch_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_pitch_velocity)
        roll_velocity_rew = torch.exp(-roll_velocity_distance / self._task_cfg.roll_vel_exponential_reward_coeff) * int(
            self._task_cfg.enable_roll_velocity
        )

        # Check if the goal is reached
        if self._task_cfg.enable_linear_velocity:
            linear_goal_is_reached = (linear_velocity_distance < self._task_cfg.linear_velocity_tolerance).int()
        else:
            linear_goal_is_reached = torch.ones_like(linear_velocity_distance, dtype=torch.int32)
        if self._task_cfg.enable_lateral_velocity:
            lateral_goal_is_reached = (lateral_velocity_distance < self._task_cfg.lateral_velocity_tolerance).int()
        else:
            lateral_goal_is_reached = torch.ones_like(lateral_velocity_distance, dtype=torch.int32)
        if self._task_cfg.enable_vertical_velocity:
            vertical_goal_is_reached = (vertical_velocity_distance < self._task_cfg.vertical_velocity_tolerance).int()
        else:
            vertical_goal_is_reached = torch.ones_like(vertical_velocity_distance, dtype=torch.int32)

        if self._task_cfg.enable_yaw_velocity:
            yaw_goal_is_reached = (yaw_velocity_distance < self._task_cfg.yaw_velocity_tolerance).int()
        else:
            yaw_goal_is_reached = torch.ones_like(yaw_velocity_distance, dtype=torch.int32)
        if self._task_cfg.enable_pitch_velocity:
            pitch_goal_is_reached = (pitch_velocity_distance < self._task_cfg.pitch_velocity_tolerance).int()
        else:
            pitch_goal_is_reached = torch.ones_like(pitch_velocity_distance, dtype=torch.int32)
        if self._task_cfg.enable_roll_velocity:
            roll_goal_is_reached = (roll_velocity_distance < self._task_cfg.roll_velocity_tolerance).int()
        else:
            roll_goal_is_reached = torch.ones_like(roll_velocity_distance, dtype=torch.int32)

        goal_is_reached = (
            linear_goal_is_reached
            * lateral_goal_is_reached
            * vertical_goal_is_reached
            * yaw_goal_is_reached
            * pitch_goal_is_reached
            * roll_goal_is_reached
        )
        self._goal_reached += goal_is_reached

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self.scalar_logger.log("task_reward", "EMA/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "EMA/lateral_velocity", lateral_velocity_rew)
        self.scalar_logger.log("task_reward", "EMA/vertical_velocity", vertical_velocity_rew)
        self.scalar_logger.log("task_reward", "EMA/yaw_velocity", yaw_velocity_rew)
        self.scalar_logger.log("task_reward", "EMA/pitch_velocity", pitch_velocity_rew)
        self.scalar_logger.log("task_reward", "EMA/roll_velocity", roll_velocity_rew)

        # Return the reward by combining the different components and adding the robot rewards
        return (
            linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + lateral_velocity_rew * self._task_cfg.lateral_velocity_weight
            + vertical_velocity_rew * self._task_cfg.vertical_velocity_weight
            + yaw_velocity_rew * self._task_cfg.yaw_velocity_weight
            + pitch_velocity_rew * self._task_cfg.pitch_velocity_weight
            + roll_velocity_rew * self._task_cfg.roll_velocity_weight
        ) + self._robot.compute_rewards()

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        """
        Resets the task to its initial state.

        The environment actions for this task are the following all belong to the [0,1] range:
        - gen_actions[0]: The value used to sample the target linear velocity.
        - gen_actions[1]: The value used to sample the target lateral velocity.
        - gen_actions[2]: The value used to sample the target vertical velocity.
        - gen_actions[3]: The value used to sample the target yaw velocity.
        - gen_actions[4]: The value used to sample the target pitch velocity.
        - gen_actions[5]: The value used to sample the target roll velocity.
        - gen_actions[6]: The value used to sample the linear velocities of the robot at spawn.
        - gen_actions[7]: The value used to sample the angular velocities of the robot at spawn.

        Args:
            task_actions (torch.Tensor): The actions for the task.
            env_seeds (torch.Tensor): The seeds for the environments.
            env_ids (torch.Tensor): The ids of the environments.
        """
        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        self._num_steps[env_ids] = 0
        self.update_goals()

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not.
        """

        # Kill the robot if it goes too far, but don't count it as an early termination.
        # Distances (world) in all 3 dimensions
        position_distance = torch.norm(
            self._env_origins[:, :3] - self._robot.root_link_pos_w[self._env_ids, :3], dim=-1
        )
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(
            position_distance > self._task_cfg.maximum_robot_distance,
            ones,
            task_completed,
        )

        # This task cannot be failed.
        zeros = torch.zeros_like(self._goal_reached, dtype=torch.long)
        return zeros, task_completed

    def set_goals(self, env_ids: torch.Tensor) -> None:
        """
        Generates a random goal for the task.
        These goals are generated in a way allowing to precisely control the difficulty of the task through the environment action. In this task, the environment actions control 3 different elements:
        - env_actions[0]: The value used to sample the target linear velocity.
        - env_actions[1]: The value used to sample the target lateral velocity.
        - env_actions[2]: The value used to sample the target vertical velocity.
        - env_actions[3]: The value used to sample the target yaw velocity.
        - env_actions[4]: The value used to sample the target pitch velocity.
        - env_actions[5]: The value used to sample the target roll velocity.

        In this tasks goals are constantly updated. The target velocities are updated at regular intervals, and
        an EMA is used to generate target velocities that smoothly change over time. The EMA rate and the interval
        at which the goals are updated are controlled by the task configuration and randomly sampled.
        These cannot be controlled through environment actions.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """
        # Set velocity targets
        if self._task_cfg.enable_linear_velocity:
            self._linear_velocity_target[env_ids] = (
                self._gen_actions[env_ids, 0] * (self._task_cfg.goal_max_lin_vel - self._task_cfg.goal_min_lin_vel)
                + self._task_cfg.goal_min_lin_vel
            ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            self._linear_velocity_desired[env_ids] = self._linear_velocity_target.clone()
        if self._task_cfg.enable_lateral_velocity:
            self._lateral_velocity_target[env_ids] = (
                self._gen_actions[env_ids, 1] * (self._task_cfg.goal_max_lat_vel - self._task_cfg.goal_min_lat_vel)
                + self._task_cfg.goal_min_lat_vel
            ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            self._lateral_velocity_desired[env_ids] = self._lateral_velocity_target.clone()
        if self._task_cfg.enable_vertical_velocity:
            self._vertical_velocity_target[env_ids] = (
                self._gen_actions[env_ids, 2] * (self._task_cfg.goal_max_ver_vel - self._task_cfg.goal_min_ver_vel)
                + self._task_cfg.goal_min_ver_vel
            ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            self._vertical_velocity_desired[env_ids] = self._vertical_velocity_target.clone()
        if self._task_cfg.enable_yaw_velocity:
            self._yaw_velocity_target[env_ids] = (
                self._gen_actions[env_ids, 3] * (self._task_cfg.goal_max_yaw_vel - self._task_cfg.goal_min_yaw_vel)
                + self._task_cfg.goal_min_yaw_vel
            ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            self._yaw_velocity_desired[env_ids] = self._yaw_velocity_target.clone()
        if self._task_cfg.enable_pitch_velocity:
            self._pitch_velocity_target[env_ids] = (
                self._gen_actions[env_ids, 4] * (self._task_cfg.goal_max_pitch_vel - self._task_cfg.goal_min_pitch_vel)
                + self._task_cfg.goal_min_pitch_vel
            ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            self._pitch_velocity_desired[env_ids] = self._pitch_velocity_target.clone()
        if self._task_cfg.enable_roll_velocity:
            self._roll_velocity_target[env_ids] = (
                self._gen_actions[env_ids, 5] * (self._task_cfg.goal_max_roll_vel - self._task_cfg.goal_min_roll_vel)
                + self._task_cfg.goal_min_roll_vel
            ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            self._roll_velocity_desired[env_ids] = self._roll_velocity_target.clone()

        # Pick a random smoothing factor
        self._smoothing_factor[env_ids] = (
            self._rng.sample_uniform_torch(0.0, 1.0, 1, ids=env_ids)
            * (self._task_cfg.smoothing_factor[1] - self._task_cfg.smoothing_factor[0])
            + self._task_cfg.smoothing_factor[0]
        )
        # Pick a random number of steps to update the goals
        self._update_after_n_steps[env_ids] = self._rng.sample_integer_torch(
            self._task_cfg.interval[0],
            self._task_cfg.interval[1],
            1,
            ids=env_ids,
        )

    def update_goals(self) -> None:
        """
        Updates the goals for the task.
        """

        # Update the number of steps
        self._num_steps += 1

        # Use EMA to update the target velocities
        if self._task_cfg.enable_linear_velocity:
            self._linear_velocity_target = (
                self._linear_velocity_desired * (1 - self._smoothing_factor)
                + self._linear_velocity_target * self._smoothing_factor
            )  # slowly increase the target velocity
        if self._task_cfg.enable_lateral_velocity:
            self._lateral_velocity_target = (
                self._lateral_velocity_desired * (1 - self._smoothing_factor)
                + self._lateral_velocity_target * self._smoothing_factor
            )
        if self._task_cfg.enable_vertical_velocity:
            self._vertical_velocity_target = (
                self._vertical_velocity_desired * (1 - self._smoothing_factor)
                + self._vertical_velocity_target * self._smoothing_factor
            )
        if self._task_cfg.enable_yaw_velocity:
            self._yaw_velocity_target = (
                self._yaw_velocity_desired * (1 - self._smoothing_factor)
                + self._yaw_velocity_target * self._smoothing_factor
            )
        if self._task_cfg.enable_pitch_velocity:
            self._pitch_velocity_target = (
                self._pitch_velocity_desired * (1 - self._smoothing_factor)
                + self._pitch_velocity_target * self._smoothing_factor
            )
        if self._task_cfg.enable_roll_velocity:
            self._roll_velocity_target = (
                self._roll_velocity_desired * (1 - self._smoothing_factor)
                + self._roll_velocity_target * self._smoothing_factor
            )
        # Check if the goals should be updated
        idx_to_update = torch.where(self._update_after_n_steps < self._num_steps)[0]
        num_updates = len(idx_to_update)
        if num_updates > 0:
            # Update the desired velocities
            if self._task_cfg.enable_linear_velocity:
                self._linear_velocity_desired[idx_to_update] = (
                    self._gen_actions[idx_to_update, 0]
                    * (self._task_cfg.goal_max_lin_vel - self._task_cfg.goal_min_lin_vel)
                    + self._task_cfg.goal_min_lin_vel
                ) * self._rng.sample_sign_torch("float", 1, ids=idx_to_update)
            if self._task_cfg.enable_lateral_velocity:
                self._lateral_velocity_desired[idx_to_update] = (
                    self._gen_actions[idx_to_update, 1]
                    * (self._task_cfg.goal_max_lat_vel - self._task_cfg.goal_min_lat_vel)
                    + self._task_cfg.goal_min_lat_vel
                ) * self._rng.sample_sign_torch("float", 1, ids=idx_to_update)
            if self._task_cfg.enable_vertical_velocity:
                self._vertical_velocity_desired[idx_to_update] = (
                    self._gen_actions[idx_to_update, 2]
                    * (self._task_cfg.goal_max_ver_vel - self._task_cfg.goal_min_ver_vel)
                    + self._task_cfg.goal_min_ver_vel
                ) * self._rng.sample_sign_torch("float", 1, ids=idx_to_update)
            if self._task_cfg.enable_yaw_velocity:
                self._yaw_velocity_desired[idx_to_update] = (
                    self._gen_actions[idx_to_update, 3]
                    * (self._task_cfg.goal_max_yaw_vel - self._task_cfg.goal_min_yaw_vel)
                    + self._task_cfg.goal_min_yaw_vel
                ) * self._rng.sample_sign_torch("float", 1, ids=idx_to_update)
            if self._task_cfg.enable_pitch_velocity:
                self._pitch_velocity_desired[idx_to_update] = (
                    self._gen_actions[idx_to_update, 4]
                    * (self._task_cfg.goal_max_pitch_vel - self._task_cfg.goal_min_pitch_vel)
                    + self._task_cfg.goal_min_pitch_vel
                ) * self._rng.sample_sign_torch("float", 1, ids=idx_to_update)
            if self._task_cfg.enable_roll_velocity:
                self._roll_velocity_desired[idx_to_update] = (
                    self._gen_actions[idx_to_update, 5]
                    * (self._task_cfg.goal_max_roll_vel - self._task_cfg.goal_min_roll_vel)
                    + self._task_cfg.goal_min_roll_vel
                ) * self._rng.sample_sign_torch("float", 1, ids=idx_to_update)
            # Pick a random smoothing factor
            self._smoothing_factor[idx_to_update] = (
                self._rng.sample_uniform_torch(0.0, 1.0, 1, ids=idx_to_update)
                * (self._task_cfg.smoothing_factor[1] - self._task_cfg.smoothing_factor[0])
                + self._task_cfg.smoothing_factor[0]
            )
            # Pick a random number of steps to update the goals
            self._update_after_n_steps[idx_to_update] = self._rng.sample_integer_torch(
                self._task_cfg.interval[0],
                self._task_cfg.interval[1],
                1,
                ids=idx_to_update,
            )
            self._num_steps[idx_to_update] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Generates the initial conditions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """

        num_resets = len(env_ids)
        # Randomizes the initial pose of the robot
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)
        # The position is not randomized no point in doing it
        initial_pose[:, :2] = self._env_origins[env_ids, :2]
        # Launch the robot at some height, otherwise it'll likely collide with the ground
        initial_pose[:, 2] = self._task_cfg.spawn_initial_height
        # The orientation is not randomized no point in doing it
        initial_pose[:, 3] = 1.0
        initial_pose[:, 3:] = torch.nn.functional.normalize(initial_pose[:, 3:], dim=-1)

        # Randomize the linear and angular velocities
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocities
        velocity_norm = (
            self._gen_actions[env_ids, 6] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )

        # To get a random 3D direction uniformly, can sample 2 angles
        # phi: azimuthal angle in [0, 2pi]
        phi = self._rng.sample_uniform_torch(0.0, 2 * math.pi, 1, ids=env_ids)
        # cos_theta: cosine of the polar angle in [-1, 1] (yields a uniform distb. over the sphere)
        cos_theta = self._rng.sample_uniform_torch(-1.0, 1.0, 1, ids=env_ids)
        sin_theta = torch.sqrt(1.0 - cos_theta**2)

        # Decompose into x, y, z components
        initial_velocity[:, 0] = velocity_norm * sin_theta * torch.cos(phi)
        initial_velocity[:, 1] = velocity_norm * sin_theta * torch.sin(phi)
        initial_velocity[:, 2] = velocity_norm * cos_theta

        roll_ang_vel = (
            self._gen_actions[env_ids, 7] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        pitch_ang_vel = (
            self._gen_actions[env_ids, 8] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        yaw_ang_vel = (
            self._gen_actions[env_ids, 9] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)

        initial_velocity[:, 3] = roll_ang_vel
        initial_velocity[:, 4] = pitch_ang_vel
        initial_velocity[:, 5] = yaw_ang_vel

        # Apply to articulation
        self._robot.set_pose(initial_pose, env_ids)
        self._robot.set_velocity(initial_velocity, env_ids)

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are four markers in the scene:
        - The target linear velocity. It is represented by a red arrow.
        - The target angular velocity. It is represented by a green arrow.
        - The robot linear velocity. It is represented by a purple arrow.
        - The robot angular velocity. It is represented by a cyan arrow.

        These arrows are used to visually assess the performance of the robot with regard to its velocity tracking
        task."""

        # Linear velocity goal
        marker_cfg = ARROW_CFG.copy()
        marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_linear_velocity"
        marker_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        self.goal_linvel_visualizer = VisualizationMarkers(marker_cfg)
        # Angular velocity goal
        marker_cfg = ARROW_CFG.copy()
        marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_angular_velocity"
        marker_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        self.goal_angvel_visualizer = VisualizationMarkers(marker_cfg)
        # Robot linear velocity
        marker_cfg = ARROW_CFG.copy()
        marker_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0))
        marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_linear_velocity"
        self.robot_linvel_visualizer = VisualizationMarkers(marker_cfg)
        # Robot angular velocity
        marker_cfg = ARROW_CFG.copy()
        marker_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0))
        marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_angular_velocity"
        self.robot_angvel_visualizer = VisualizationMarkers(marker_cfg)

    def update_task_visualization(self) -> None:
        """
        Updates four 3D arrow markers in the scene:
        1) Target linear velocity (red)
        2) Target angular velocity (green)
        3) Robot linear velocity (purple)
        4) Robot angular velocity (cyan)

        Each arrow is oriented along the velocity direction (in world frame) and scaled
        by the velocity magnitude.
        """

        def direction_to_quaternion(direction: torch.Tensor) -> torch.Tensor:
            """
            Helper: convert a direction vector -> quaternion (+X -> direction)
            direction vec: (N, 3)

            Returns:   (N, 4) quaternion [w, x, y, z] that rotates +X to 'direction'.
            Falls back to identity if direction is near zero.
            """
            n_envs = direction.shape[0]
            orientation = torch.zeros((n_envs, 4), dtype=direction.dtype, device=direction.device)

            # Normalize
            norms = torch.norm(direction, dim=-1, keepdim=True)

            if torch.all(norms == 0):
                return torch.tensor([1.0, 0.0, 0.0, 0.0], device=direction.device, dtype=direction.dtype).expand(
                    n_envs, 4
                )

            valid_mask = norms.squeeze(-1) > EPS
            dir_normed = torch.where(
                valid_mask.unsqueeze(-1),
                direction / (norms + EPS),
                torch.tensor([1.0, 0.0, 0.0], device=direction.device, dtype=direction.dtype),
            )

            # +X in marker’s local frame
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=direction.device, dtype=direction.dtype).expand_as(dir_normed)

            # Dot/cross => axis-angle
            dotval = (x_axis * dir_normed).sum(dim=-1).clamp(-1.0, 1.0)
            crossval = torch.cross(x_axis, dir_normed, dim=-1)

            angles = torch.acos(dotval)  # in [0, pi]
            axis_norm = torch.norm(crossval, dim=-1, keepdim=True)
            axis_valid = axis_norm.squeeze(-1) > EPS
            axis = torch.where(
                axis_valid.unsqueeze(-1),
                crossval / (axis_norm + EPS),
                torch.tensor([0.0, 1.0, 0.0], device=direction.device, dtype=direction.dtype),
            )

            half_angle = angles * 0.5
            sin_half = torch.sin(half_angle)
            # w, x, y, z
            orientation[:, 0] = torch.cos(half_angle)
            orientation[:, 1] = axis[:, 0] * sin_half
            orientation[:, 2] = axis[:, 1] * sin_half
            orientation[:, 3] = axis[:, 2] * sin_half

            # For zero-length direction, use identity [1,0,0,0]
            orientation[~valid_mask, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=direction.device, dtype=direction.dtype
            )
            return orientation

        # target linear velocity in world frame
        target_lin_b = torch.stack(
            (self._linear_velocity_target, self._lateral_velocity_target, self._vertical_velocity_target), dim=-1
        )  # shape [N, 3], in body frame
        # Rotate from body to world
        target_lin_w = math_utils.quat_rotate(self._robot.root_link_quat_w, target_lin_b)

        marker_pos = self._robot.root_link_pos_w.clone()

        # Orientation: align +X with velocity direction
        marker_orientation = direction_to_quaternion(target_lin_w)

        # Scale arrow by velocity magnitude
        marker_scale = torch.ones_like(marker_pos)
        lin_magnitudes = torch.norm(target_lin_w, dim=-1)
        marker_scale[:, 0] = lin_magnitudes * self._task_cfg.visualization_linear_velocity_scale

        self.goal_linvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)

        # target angular velocity in world frame
        target_ang_b = torch.stack(
            (self._roll_velocity_target, self._pitch_velocity_target, self._yaw_velocity_target), dim=-1
        )
        # rotate from body to world
        target_ang_w = math_utils.quat_rotate(self._robot.root_link_quat_w, target_ang_b)

        marker_pos = self._robot.root_link_pos_w.clone()
        marker_orientation = direction_to_quaternion(target_ang_w)
        ang_magnitudes = torch.norm(target_ang_w, dim=-1)
        marker_scale = torch.ones_like(marker_pos)
        marker_scale[:, 0] = ang_magnitudes * self._task_cfg.visualization_angular_velocity_scale

        self.goal_angvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)

        # robot's linear velocity in world frame
        robot_lin_w = self._robot.root_com_lin_vel_w[:, :3]

        marker_pos = self._robot.root_link_pos_w.clone()
        marker_pos[:, 2] += 0.2  # offset

        marker_orientation = direction_to_quaternion(robot_lin_w)
        lin_magnitudes = torch.norm(robot_lin_w, dim=-1)
        marker_scale = torch.ones_like(marker_pos)
        marker_scale[:, 0] = lin_magnitudes * self._task_cfg.visualization_linear_velocity_scale

        self.robot_linvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)

        # robot's angular velocity in world frame
        robot_ang_w = self._robot.root_com_ang_vel_w[:, :3]

        marker_pos = self._robot.root_link_pos_w.clone()
        marker_pos[:, 2] += 0.2

        marker_orientation = direction_to_quaternion(robot_ang_w)
        ang_magnitudes = torch.norm(robot_ang_w, dim=-1)
        marker_scale = torch.ones_like(marker_pos)
        # marker_scale[:, 0] = ang_magnitudes * self._task_cfg.visualization_angular_velocity_scale
        marker_scale[:, 0] = torch.clamp(
            ang_magnitudes * self._task_cfg.visualization_angular_velocity_scale,
            max=2.0,
        )

        self.robot_angvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)
