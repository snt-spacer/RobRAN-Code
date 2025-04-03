# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.markers import POSE_MARKER_3D_CFG, SPHERE_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import GoThroughPoses3DCfg

from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoThroughPoses3DTask(TaskCore):
    """
    Implements the GoThroughPosition task. The robot has to reach a target position.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoThroughPoses3DCfg = GoThroughPoses3DCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Initializes the GoThroughPoses task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task."""

        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the 3D task.

        Args:
            env_ids: The ids of the environments used by this task.
        """

        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)  # (x, y, z)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals, 3), device=self._device, dtype=torch.float32
        )
        self._local_pos_error = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        # Orientation tracking (Quaternion instead of heading angle)
        self._target_orientations = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals, 4), device=self._device, dtype=torch.float32
        )  # (qw, qx, qy, qz)
        self._orientation_error = torch.zeros(
            (self._num_envs,), device=self._device, dtype=torch.float32
        )  # Orientation error metric
        self._target_index = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._trajectory_completed = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._num_goals = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task.
        Tracks 3D velocity, angular velocity, and position distance.
        """

        super().create_logs()

        self.scalar_logger.add_log("task_state", "AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "EMA/position_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/orientation_error", "ema")
        self.scalar_logger.add_log("task_state", "EMA/boundary_distance", "ema")

        self.scalar_logger.add_log("task_reward", "AVG/position", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/orientation", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/progress", "mean")
        self.scalar_logger.add_log("task_reward", "SUM/num_goals", "sum")

        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.
        Tracks 6DoF (position + quaternion orientation).

        The observation tensor is composed of the following elements:
        - The position error between the robot and the target in the robot's local frame.
        - The orientation error between the robot and the target in the robot's local frame (RPY).
        - The linear velocity of the robot in the robot's local frame.
        - The angular velocity of the robot in the robot's local frame.
        - Depending on the task configuration, a number of subsequent poses are added to the observation. For each of
            them, the following elements are added:
            - The position error between the nth and n+1th goal in the robot's local frame is computed.
            - The orientation error between the nth and n+1th goal in the robot's local frame is computed.

        The observation space will be = 12 + 6 * num_subsequent_goals

        Returns:
            torch.Tensor: The observation tensor.
        """

        # position error in world frame
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :3]
        )
        # rotate into robot's local frame via inverse of current orientation
        current_quat_w = self._robot.root_link_quat_w[self._env_ids]
        self._local_pos_error = math_utils.quat_rotate_inverse(current_quat_w, self._position_error)

        # log the global distance for debugging
        self._position_dist = self._position_error.norm(dim=-1)  # shape [N]
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)

        # robot orientation
        # Compute a relative quaternion from robot -> target
        target_quat_w = self._target_orientations[self._ALL_INDICES, self._target_index]
        # rel_quat = conj(current) * target
        rel_quat = math_utils.quat_mul(
            math_utils.quat_conjugate(current_quat_w), target_quat_w
        )  # rotation from robot's orientation to target's orientation in robot local frame

        # Rotation matrix magic:
        rel_mat = math_utils.matrix_from_quat(rel_quat)
        # Extract the first two columns
        col0 = rel_mat[:, :, 0]  # shape [N, 3]
        col1 = rel_mat[:, :, 1]  # shape [N, 3]
        # Re-orthonormalize using Gram-Schmidt:
        col0 = torch.nn.functional.normalize(col0, dim=-1)
        proj = (col1 * col0).sum(dim=-1, keepdim=True)
        col1 = col1 - proj * col0
        col1 = torch.nn.functional.normalize(col1, dim=-1)
        # Stack to get 6D representation
        rel_mat_6 = torch.cat([col0, col1], dim=-1)  # shape [N, 6]

        # Store in buffer [position_dist, rotation error, linear_vel_xyz, angular_vel_xyz]
        self._task_data[:, 0:3] = self._local_pos_error
        self._task_data[:, 3:9] = rel_mat_6
        self._task_data[:, 9:12] = self._robot.root_com_lin_vel_b[self._env_ids]
        self._task_data[:, 12:15] = self._robot.root_com_ang_vel_b[self._env_ids]

        # Update also the orientation error magnitude
        target_quat_w = torch.nn.functional.normalize(target_quat_w, dim=-1, eps=EPS)
        current_quat_w = torch.nn.functional.normalize(current_quat_w, dim=-1, eps=EPS)
        self._orientation_error[self._env_ids] = math_utils.quat_error_magnitude(target_quat_w, current_quat_w)

        # We compute the observations of the subsequent goals in the previous goal's frame.
        for i in range(self._task_cfg.num_subsequent_goals - 1):
            next_indices = torch.where(
                (self._target_index + i + 1) >= self._num_goals,
                torch.zeros_like(self._target_index),  # Set to 0 if overflowing
                self._target_index + i + 1,
            )

            if self._task_cfg.loop:
                next_indices = (self._target_index + i + 1) % self._num_goals

            # Compute position error between nth and (n+1)th goal
            subsequent_goal = self._target_positions[self._ALL_INDICES, next_indices, :]
            subsequent_error = subsequent_goal - self._target_positions[self._ALL_INDICES, self._target_index]
            local_subsequent_error = math_utils.quat_rotate_inverse(current_quat_w, subsequent_error)

            # Compute orientation error between nth and (n+1)th goal in the current goal's frame
            next_goal_quat_w = self._target_orientations[self._ALL_INDICES, next_indices]
            goal_rel_quat = math_utils.quat_mul(math_utils.quat_conjugate(target_quat_w), next_goal_quat_w)

            # Convert to 3x3 rot matrix, stored as 6 floats
            goal_rel_mat = math_utils.matrix_from_quat(goal_rel_quat)
            col0g = goal_rel_mat[:, :, 0]
            col1g = goal_rel_mat[:, :, 1]
            col0g = torch.nn.functional.normalize(col0g, dim=-1, eps=EPS)
            projg = (col1g * col0g).sum(dim=-1, keepdim=True)
            col1g = col1g - projg * col0g
            col1g = torch.nn.functional.normalize(col1g, dim=-1, eps=EPS)
            goal_rel_mat_6 = torch.cat([col0g, col1g], dim=-1)  # shape [N, 6]

            # If looping is disabled, set error to zero where indices overflow
            if not self._task_cfg.loop:
                # valid = torch.logical_not((self._target_index + i + 1) >= self._num_goals).unsqueeze(-1)
                valid_mask = ((self._target_index + i + 1) < self._num_goals).unsqueeze(-1)
                local_subsequent_error = local_subsequent_error * valid_mask
                goal_rel_mat_6 = goal_rel_mat_6 * valid_mask

            # Store in buffer
            start_idx = 15 + 9 * i
            self._task_data[:, start_idx : start_idx + 3] = local_subsequent_error  # Position error
            self._task_data[:, start_idx + 3 : start_idx + 9] = goal_rel_mat_6  # rotation to next goal

        # Concatenate task observations with robot's internal observations
        return torch.concat((self._task_data, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The computed reward for the current state.
        """

        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self._robot.root_com_vel_w[self._env_ids], dim=-1)
        # normed angular velocity
        angular_velocity = torch.norm(self._robot.root_com_ang_vel_w[self._env_ids], dim=-1)

        # Linear velocity reward
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)

        # Angular velocity reward
        angular_velocity_rew = angular_velocity - self._task_cfg.angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew
            > (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        ] = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)

        # boundary reward
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # progress reward
        progress_rew = self._previous_position_dist - self._position_dist

        # Logging
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "EMA/orientation_error", torch.norm(self._orientation_error))
        self.scalar_logger.log("task_state", "EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)

        # Position reward (exponential decay based on distance)
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)
        # orientation reward (encourages the robot to orient the same way as the target)
        orientation_rew = torch.exp(-self._orientation_error / self._task_cfg.orientation_exponential_reward_coeff)
        # Check if goal is reached (both position and orientation must be within tolerance)
        position_goal_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        orientation_goal_reached = (self._orientation_error < self._task_cfg.orientation_tolerance).int()
        goal_reached = position_goal_reached * orientation_goal_reached
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # If goal is reached, update the target index
        self._target_index = self._target_index + goal_reached
        # Check if trajectory is completed
        self._trajectory_completed = self._target_index >= self._num_goals
        # If the trajectory is completed and looping is disabled, reset the index to 0
        self._target_index = self._target_index * (~self._trajectory_completed)
        # If goal is reached, reset progress tracking
        self._previous_position_dist[reached_ids] = 0

        # Logging rewards
        self.scalar_logger.log("task_reward", "AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "AVG/orientation", orientation_rew)
        self.scalar_logger.log("task_reward", "AVG/progress", progress_rew)
        self.scalar_logger.log("task_reward", "SUM/num_goals", goal_reached)

        # Compute final reward
        total_reward = (
            progress_rew * self._task_cfg.progress_weight
            + (position_rew * orientation_rew) * self._task_cfg.pose_weight
            # + orientation_rew * self._task_cfg.pose_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + self._task_cfg.time_penalty
            + self._task_cfg.reached_bonus * goal_reached
        ) + self._robot.compute_rewards()

        return total_reward

    def reset(
        self, env_ids: torch.Tensor, gen_actions: torch.Tensor | None = None, env_seeds: torch.Tensor | None = None
    ) -> None:
        """
        Resets the task to its initial state.

        If gen_actions is None, then the environment is generated at random. This is the default mode.
        If env_seeds is None, then the seed is generated at random. This is the default mode.

        The environment generation actions (gen_actions) control the difficulty of the task.
        Each action belongs to the [0,1] range and scales the corresponding parameter as follows:
        - gen_actions[0]: The lower bound of the range used to sample the yaw offset between the goals.
        - gen_actions[1]: The range used to sample the difference in yaw between the goals.
        - gen_actions[2]: The lower bound of the range used to sample the pitch offset between the goals.
        - gen_actions[3]: The range used to sample the difference in pitch between the goals.
        - gen_actions[4]: The lower bound of the range used to sample the roll offset between the goals.
        - gen_actions[5]: The range used to sample the difference in roll between the goals.
        - gen_actions[6]: The lower bound of the range used to sample the distance between the goals.
        - gen_actions[7]: The range used to sample the distance between the goals.
        - gen_actions[8]: The lower bound of the range used to sample the (polar) inclination angle of the goals.
        - gen_actions[9]: The range used to sample the (polar) inclination angle of the goals.
        - gen_actions[10]: The lower bound of the range used to sample the (polar) azimuthal angle of the goals.
        - gen_actions[11]: The range used to sample the (polar) azimulathal angle of the goals.
        - gen_actions[12]: The value used to sample the distance between the spawn position and the first goal.
        - gen_actions[13]: The value used to sample the orientation error the spawn position and the first goal.
        - gen_actions[14]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[15]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            env_ids (torch.Tensor): The IDs of the environments to reset.
            gen_actions (torch.Tensor | None): The task-specific generation actions for sampling initial conditions.
                If None, defaults to uniform random sampling.
            env_seeds (torch.Tensor | None): The seeds for the environments to ensure reproducibility. Defaults to None.
        """
        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # Reset the target index and trajectory completed
        self._target_index[env_ids] = 0
        self._trajectory_completed[env_ids] = False

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids, self._target_index[env_ids]]
            - self._robot.root_link_pos_w[self._env_ids][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()
        # Make sure the orientation error is also updated
        current_quat = self._robot.root_link_quat_w[self._env_ids][env_ids]  # Current robot orientation
        target_quat = self._target_orientations[env_ids, self._target_index[env_ids]]  # Target orientation
        self._orientation_error[env_ids] = math_utils.quat_error_magnitude(target_quat, current_quat)

        # The first 12 env actions define ranges, we need to make sure they don't exceed the [0, 1] range.
        # They are given as [min, delta], we will convert them to [min, max], that is max = min + delta
        # Note that they are defined as [min, delta] to make sure the min is the min and the max is the max. This
        # is always true as they are strictly positive.
        for i in range(0, 12, 2):  # Iterate over lower bounds (even indices)
            self._gen_actions[env_ids, i + 1] = torch.clip(
                self._gen_actions[env_ids, i] + self._gen_actions[env_ids, i + 1], max=1.0
            )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            task_failed (torch.Tensor): Environments where the robot has exceeded max distance.
            task_completed (torch.Tensor): Environments where the goal has been reached for enough steps.
        """
        # Compute position error in world frame (extend to 3D)
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index] - self._robot.root_link_pos_w[self._env_ids]
        )
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        current_quat = self._robot.root_quat_w[self._env_ids]  # (w, x, y, z)
        target_quat = self._target_orientations[self._env_ids, self._target_index[self._env_ids]]  # (w, x, y, z)
        self._orientation_error = math_utils.quat_error_magnitude(target_quat, current_quat)

        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self._task_cfg.maximum_robot_distance,
            ones,
            task_failed,
        )
        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        # If the task is set to loop, don't terminate the episode early.
        if not self._task_cfg.loop:
            task_completed = torch.where(self._trajectory_completed > 0, ones, task_completed)

        return task_failed, task_completed

    def set_goals(self, env_ids: torch.Tensor) -> None:
        """
        Generates a sequence of 3D goals with orientations for the task.
        These goals are generated in a way that allows precise control over the difficulty of the task
        through environment actions.

        - First goal: Random position within a cubic region centered on the environment origin.
        - Subsequent goals: Placed at a controlled distance from the previous goal using sampled
        spherical coordinates (radius, polar, and azimuthal angles).
        - Orientation sampling: Each goal has an associated orientation, defined as a quaternion.
        Orientation offsets (yaw, pitch, roll) are sampled and applied from one goal to the next.

        Args:
            env_ids (torch.Tensor): The IDs of the environments.

        Returns:
            None
        """

        # Select how many random goals we want to generate per environment
        self._num_goals[env_ids] = self._rng.sample_integer_torch(
            self._task_cfg.min_num_goals, self._task_cfg.max_num_goals, 1, ids=env_ids
        ).to(torch.long)

        # Since we are using tensor operations, we cannot have different number of goals per environment: the
        # tensor containing the target positions must have the same number of goals for all environments.
        # Hence, we will duplicate the last goals for the environments that have less goals.
        for i in range(self._task_cfg.max_num_goals):
            if i == 0:
                # First goal: Random position in a cubic space centered around the environment origin
                self._target_positions[env_ids, i] = (
                    self._rng.sample_uniform_torch(
                        -self._task_cfg.goal_max_dist_from_origin,
                        self._task_cfg.goal_max_dist_from_origin,
                        3,
                        ids=env_ids,
                    )
                    + self._env_origins[env_ids]
                )
                # First goal: Random orientation (yaw, pitch, roll)
                yaw_offset = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)
                pitch_offset = self._rng.sample_uniform_torch(
                    -math.pi / 2, math.pi / 2, 1, ids=env_ids
                )  # Avoid flipping
                roll_offset = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)
                self._target_orientations[env_ids, i] = math_utils.quat_from_euler_xyz(
                    roll_offset, pitch_offset, yaw_offset
                )
            else:
                # If needed, randomize the next goals
                # Sample goal-to-goal distance using environment actions
                r = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 6], self._gen_actions[env_ids, 7], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_dist - self._task_cfg.goal_min_dist)
                    + self._task_cfg.goal_min_dist
                )
                phi = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 8], self._gen_actions[env_ids, 9], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_polar_angle - self._task_cfg.goal_min_polar_angle)
                    + self._task_cfg.goal_min_polar_angle
                )
                theta = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 10], self._gen_actions[env_ids, 11], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_azimuthal_angle - self._task_cfg.goal_min_azimuthal_angle)
                    + self._task_cfg.goal_min_azimuthal_angle
                )

                # Compute new goal position using spherical coordinates
                self._target_positions[env_ids, i, 0] = (
                    r * torch.sin(phi) * torch.cos(theta) + self._target_positions[env_ids, i - 1, 0]
                )
                self._target_positions[env_ids, i, 1] = (
                    r * torch.sin(phi) * torch.sin(theta) + self._target_positions[env_ids, i - 1, 1]
                )
                self._target_positions[env_ids, i, 2] = r * torch.cos(phi) + self._target_positions[env_ids, i - 1, 2]

                # Sample orientation offset using environment actions
                yaw_offset = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 0], self._gen_actions[env_ids, 1], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_yaw_offset - self._task_cfg.goal_min_yaw_offset)
                    + self._task_cfg.goal_min_yaw_offset
                )
                pitch_offset = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 2], self._gen_actions[env_ids, 3], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_pitch_offset - self._task_cfg.goal_min_pitch_offset)
                    + self._task_cfg.goal_min_pitch_offset
                )
                roll_offset = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 4], self._gen_actions[env_ids, 5], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_roll_offset - self._task_cfg.goal_min_roll_offset)
                    + self._task_cfg.goal_min_roll_offset
                )

                # Apply the orientation offset to the previous goal's orientation
                prev_quat = self._target_orientations[env_ids, i - 1]
                offset_quat = math_utils.quat_from_euler_xyz(roll_offset, pitch_offset, yaw_offset)
                offset_quat = torch.nn.functional.normalize(offset_quat, dim=-1, eps=EPS)
                self._target_orientations[env_ids, i] = math_utils.quat_mul(prev_quat, offset_quat)

                # Ensure all environments have the same number of goals by duplicating the last goal
                self._target_positions[env_ids, i] = torch.where(
                    self._num_goals[env_ids].repeat_interleave(3).reshape(-1, 3) <= i,
                    self._target_positions[env_ids, self._num_goals[env_ids] - 1],
                    self._target_positions[env_ids, i],
                )
                self._target_orientations[env_ids, i] = torch.where(
                    self._num_goals[env_ids].repeat_interleave(4).reshape(-1, 4) <= i,
                    self._target_orientations[env_ids, self._num_goals[env_ids] - 1],
                    self._target_orientations[env_ids, i],
                )

    def set_initial_conditions(self, env_ids):
        """
        Generates the initial conditions for the robots. The initial conditions are randomized based on the
        environment actions. The generation of the initial conditions is done so that if the environment actions are
        close to 0 then the task is the easiest, if they are close to 1 then the task is hardest. The configuration of
        the task defines the ranges within which the initial conditions are randomized.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot.
        """
        num_resets = len(env_ids)

        # Randomizes the initial pose of the flyer
        initial_pose = torch.zeros(
            (num_resets, 7), device=self._device, dtype=torch.float32
        )  # (x, y, z, qw, qx, qy, qz)

        # Define random 3D spawn positions in a sphere around the goal
        r = (
            self._gen_actions[env_ids, 12] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        phi = (
            self._rng.sample_uniform_torch(self._gen_actions[env_ids, 8], self._gen_actions[env_ids, 9], 1, ids=env_ids)
            * (self._task_cfg.goal_max_polar_angle - self._task_cfg.goal_min_polar_angle)
            + self._task_cfg.goal_min_polar_angle
        )  # Inclination
        theta = (
            self._rng.sample_uniform_torch(
                self._gen_actions[env_ids, 10], self._gen_actions[env_ids, 11], 1, ids=env_ids
            )
            * (self._task_cfg.goal_max_azimuthal_angle - self._task_cfg.goal_min_azimuthal_angle)
            + self._task_cfg.goal_min_azimuthal_angle
        )  # Azimuth

        initial_pose[:, 0] = r * torch.sin(phi) * torch.cos(theta) + self._target_positions[env_ids, 0, 0]  # x
        initial_pose[:, 1] = r * torch.sin(phi) * torch.sin(theta) + self._target_positions[env_ids, 0, 1]  # y
        initial_pose[:, 2] = r * torch.cos(phi) + self._target_positions[env_ids, 0, 2]  # z

        # Sample orientation error relative to the first goal
        yaw_offset = (
            self._gen_actions[env_ids, 13]
            * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            + self._task_cfg.spawn_min_heading_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        pitch_offset = (
            self._gen_actions[env_ids, 13] * (self._task_cfg.spawn_max_pitch_dist - self._task_cfg.spawn_min_pitch_dist)
            + self._task_cfg.spawn_min_pitch_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        roll_offset = (
            self._gen_actions[env_ids, 13] * (self._task_cfg.spawn_max_roll_dist - self._task_cfg.spawn_min_roll_dist)
            + self._task_cfg.spawn_min_roll_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)

        # Compute initial orientation as quaternion
        goal_quat = self._target_orientations[env_ids, 0]  # Goal quaternion
        offset_quat = math_utils.quat_from_euler_xyz(roll_offset, pitch_offset, yaw_offset)
        initial_pose[:, 3:] = math_utils.quat_mul(goal_quat, offset_quat)

        # Initialize velocity buffer: (vx, vy, vz, wx, wy, wz)
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity scaled by difficulty
        velocity_norm = (
            self._gen_actions[env_ids, 14] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        theta = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)  # Full 360° azimuth
        phi = self._rng.sample_uniform_torch(0, math.pi, 1, ids=env_ids)  # Full 180° inclination

        initial_velocity[:, 0] = velocity_norm * torch.sin(phi) * torch.cos(theta)  # vx
        initial_velocity[:, 1] = velocity_norm * torch.sin(phi) * torch.sin(theta)  # vy
        initial_velocity[:, 2] = velocity_norm * torch.cos(phi)  # vz

        # Angular velocity scaled by difficulty
        initial_velocity[:, 3:] = (
            self._gen_actions[env_ids, 15].unsqueeze(-1)
            * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        ) * torch.randn((num_resets, 3), device=self._device)

        # Apply to articulation
        self._robot.set_pose(initial_pose, env_ids)
        self._robot.set_velocity(initial_velocity, env_ids)

    def create_task_visualization(self) -> None:
        """
        Adds the visual marker to the scene.

        There are markers for two main entities for the task: the goal and the robot.

        - The goal marker is a pose marker with a sphere in at its origin. The next pose goal sphere is green, the others remain red, while the goal already reached is grey.
        - The robot is represented by a pose marker.
        """

        # Define the visual markers for goals (XYZ + Sphere)
        goal_marker_cfg = POSE_MARKER_3D_CFG.copy()  # XYZ marker (default colors)

        sphere_marker_cfg_green = SPHERE_CFG.copy()  # Small sphere (Next goal)
        sphere_marker_cfg_green.markers["sphere"].radius = 0.05
        sphere_marker_cfg_green.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)

        sphere_marker_cfg_grey = SPHERE_CFG.copy()  # Small sphere (Passed goal)
        sphere_marker_cfg_grey.markers["sphere"].radius = 0.05
        sphere_marker_cfg_grey.markers["sphere"].visual_material.diffuse_color = (0.5, 0.5, 0.5)

        sphere_marker_cfg_red = SPHERE_CFG.copy()  # Subsequent goals sphere (Red)
        sphere_marker_cfg_red.markers["sphere"].radius = 0.05
        sphere_marker_cfg_red.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)

        # Define the robot marker (XYZ pose marker)
        robot_marker_cfg = POSE_MARKER_3D_CFG.copy()
        robot_marker_cfg.markers["pose_marker_3d"].arrow_body_length = 0.2
        robot_marker_cfg.markers["pose_marker_3d"].arrow_body_radius = 0.01

        # Assign paths to markers
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_marker"
        sphere_marker_cfg_green.prim_path = f"/Visuals/Command/task_{self._task_uid}/next_goal_sphere"
        sphere_marker_cfg_grey.prim_path = f"/Visuals/Command/task_{self._task_uid}/passed_goal_sphere"
        sphere_marker_cfg_red.prim_path = f"/Visuals/Command/task_{self._task_uid}/subsequent_goal_sphere"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"

        # Create visualization markers
        self.current_goal_visualizer = VisualizationMarkers(goal_marker_cfg)  # XYZ Axis Marker
        self.next_goal_sphere_visualizer = VisualizationMarkers(sphere_marker_cfg_green)  # Green Sphere
        self.passed_goal_sphere_visualizer = VisualizationMarkers(sphere_marker_cfg_grey)  # Grey Sphere
        self.subsequent_goal_sphere_visualizer = VisualizationMarkers(sphere_marker_cfg_red)  # Red Sphere
        self.robot_visualizer = VisualizationMarkers(robot_marker_cfg)  # Robot Marker

    def update_task_visualization(self) -> None:
        """
        Updates goal and robot visualization.

        - Goals: Pose markers depict the target pose.
        - A small sphere at the goal's origin changes color based on goal status:
            - Next goal: Green
            - Passed goals: Grey
            - Subsequent goals: Red
        - Robot: Pose markers depict the robot pose.
        """

        current_goal_pos = self._target_positions[self._ALL_INDICES, self._target_index]
        current_goal_quat = self._target_orientations[self._ALL_INDICES, self._target_index]

        passed_goals_pos_list = []
        next_goals_pos_list = []
        subsequent_goals_pos_list = []

        # Iterate through all goals and categorize them
        for i in range(self._task_cfg.max_num_goals):
            ok_goals = self._num_goals > i
            passed_goals = torch.logical_and(self._target_index > i, ok_goals)
            next_goals = torch.logical_and(self._target_index == i, ok_goals)  # The immediate next goal
            subsequent_goals = torch.logical_and(self._target_index < i, ok_goals)  # Goals after the next one

            passed_goals_pos_list.append(self._target_positions[passed_goals, i])
            next_goals_pos_list.append(self._target_positions[next_goals, i])
            subsequent_goals_pos_list.append(self._target_positions[subsequent_goals, i])

        # Convert lists to tensors (handle empty cases)
        passed_goals_pos = (
            torch.cat(passed_goals_pos_list, dim=0)
            if passed_goals_pos_list
            else torch.empty((0, 3), device=self._device)
        )
        next_goals_pos = (
            torch.cat(next_goals_pos_list, dim=0) if next_goals_pos_list else torch.empty((0, 3), device=self._device)
        )
        subsequent_goals_pos = (
            torch.cat(subsequent_goals_pos_list, dim=0)
            if subsequent_goals_pos_list
            else torch.empty((0, 3), device=self._device)
        )

        # Update pose markers for all goals
        self.current_goal_visualizer.visualize(current_goal_pos, current_goal_quat)

        # Ensure sphere markers are placed at goal marker origins
        if next_goals_pos.shape[0] > 0:
            self.next_goal_sphere_visualizer.set_visibility(True)
            self.next_goal_sphere_visualizer.visualize(next_goals_pos)
        else:
            self.next_goal_sphere_visualizer.set_visibility(False)

        if passed_goals_pos.shape[0] > 0:
            self.passed_goal_sphere_visualizer.set_visibility(True)
            self.passed_goal_sphere_visualizer.visualize(passed_goals_pos)
        else:
            self.passed_goal_sphere_visualizer.set_visibility(False)

        if subsequent_goals_pos.shape[0] > 0:
            self.subsequent_goal_sphere_visualizer.set_visibility(True)
            self.subsequent_goal_sphere_visualizer.visualize(subsequent_goals_pos)
        else:
            self.subsequent_goal_sphere_visualizer.set_visibility(False)

        # Update robot visualization
        self._robot_marker_pos[:, :3] = self._robot.root_link_pos_w[:, :3]
        self.robot_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w)
