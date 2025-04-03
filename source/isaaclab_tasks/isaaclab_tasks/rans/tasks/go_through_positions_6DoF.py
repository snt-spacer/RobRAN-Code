# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.markers import SPHERE_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import GoThroughPositions3DCfg

from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoThroughPositions3DTask(TaskCore):
    """
    Implements the GoThroughPositions task in 3D space. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoThroughPositions3DCfg = GoThroughPositions3DCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Initializes the 3D GoThroughPositions task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task.
        """

        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and action space sizes for this task
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
            (self._num_envs, 4), device=self._device, dtype=torch.float32
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
        # self.scalar_logger.add_log("task_state", "EMA/orientation_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/boundary_distance", "ema")

        self.scalar_logger.add_log("task_reward", "AVG/position", "mean")
        # self.scalar_logger.add_log("task_reward", "AVG/orientation", "mean")
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
        - Depending on the task configuration, a number of subsequent positions are added to the observation. For each of them, the position error in the robot's local frame is computed.

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
        target_quat_w = self._target_orientations[self._env_ids]
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

        # Make sure that the orientation error magnitude is also updated
        current_quat = self._robot.root_quat_w[self._env_ids]  # (w, x, y, z)
        target_quat = self._target_orientations[self._env_ids]  # (w, x, y, z)
        self._orientation_error = math_utils.quat_error_magnitude(target_quat, current_quat)

        # Subsequent goals
        for i in range(self._task_cfg.num_subsequent_goals - 1):
            # Compute the next goal index for each environment:
            # If (current_target_index + i + 1) exceeds the total number of goals, then if looping is enabled,
            # use the modulo; if not, set the index to 0.
            next_indices = torch.where(
                (self._target_index + i + 1) >= self._num_goals,
                torch.zeros_like(self._target_index),  # Set to 0 if overflowing
                self._target_index + i + 1,
            )

            # If looping is enabled, wrap around using modulo:
            if self._task_cfg.loop:
                next_indices = (self._target_index + i + 1) % self._num_goals
            # Compute the position error between the nth goal and the robot, in robot's local frame
            subsequent_goal = self._target_positions[self._ALL_INDICES, next_indices, :]
            subsequent_error = subsequent_goal - self._robot.root_link_pos_w[self._env_ids]
            local_subsequent_error = math_utils.quat_rotate_inverse(current_quat_w, subsequent_error)
            # If looping is disabled, then for environments where the index overflowed, set the error to zero.
            if not self._task_cfg.loop:
                valid = torch.logical_not((self._target_index + i + 1) >= self._num_goals).unsqueeze(
                    -1
                )  # shape: [N, 1]
                local_subsequent_error = local_subsequent_error * valid
            # Store in buffer
            start_idx = 15 + 3 * i
            self._task_data[:, start_idx : start_idx + 3] = local_subsequent_error

        # Concatenate task observations with robot's internal observations
        return torch.concat((self._task_data, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot in 3D space.

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

        # Check if goal is reached
        goal_reached = self._position_dist < self._task_cfg.position_tolerance
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached
        # Check if the trajectory is completed
        self._trajectory_completed = self._target_index >= self._num_goals
        # To avoid out of bounds errors, set the target index to 0 if the trajectory is completed
        # If the task loops, then the target index is set to 0 which will make the robot go back to the first goal
        # The episode termination is handled in the get_dones method (looping or not)
        self._target_index = self._target_index * (~self._trajectory_completed)
        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Logging
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)
        # self.scalar_logger.log("task_state", "EMA/orientation_distance", self._orientation_dist)
        self.scalar_logger.log("task_state", "EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)

        # Logging rewards
        self.scalar_logger.log("task_reward", "AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "AVG/progress", progress_rew)
        self.scalar_logger.log("task_reward", "SUM/num_goals", goal_reached)

        # Compute final reward
        total_reward = (
            progress_rew * self._task_cfg.progress_weight
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
        - gen_actions[0]: The lower bound of the range used to sample the distance between the goals.
        - gen_actions[1]: The range used to sample the distance between the goals.
        - gen_actions[2]: The value used to sample the distance between the spawn position and the first goal.
        - gen_actions[3]: Controls the randomness of the initial orientation of the robot.
          A value of 0 results in an identity quaternion, while a value of 1 leads to a fully randomized orientation.
        - gen_actions[4]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[5]: The value used to sample the angular velocity of the robot at spawn.

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

        # The first 2 env actions define ranges, we need to make sure they don't exceed the [0,1] range.
        # They are given as [min, delta] we will convert them to [min, max] that is max = min + delta
        # Note that they are defined as [min, delta] to make sure the min is the min and the max is the max.
        # This is always true as they are strictly positive.
        self._gen_actions[env_ids, 1] = torch.clip(
            self._gen_actions[env_ids, 0] + self._gen_actions[env_ids, 1], max=1.0
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            task_failed (torch.Tensor): Environments where the robot has exceeded max distance.
            task_completed (torch.Tensor): Environments where the goal has been reached for enough steps.
        """
        # Compute position error in world frame
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index] - self._robot.root_link_pos_w[self._env_ids]
        )
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)

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
        Generates a random goal for the task.
        These goals are generated in a way allowing to precisely control the difficulty of the task through the
        environment action. In this task, there is no specific actions related to the goals.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations.
        """

        # Select how many random goals we want to generate.
        self._num_goals[env_ids] = self._rng.sample_integer_torch(
            self._task_cfg.min_num_goals, self._task_cfg.max_num_goals, 1, ids=env_ids
        ).to(torch.long)

        # Since we are using tensor operations, we cannot have different number of goals per environment: the
        # tensor containing the target positions must have the same number of goals for all environments.
        # Hence, we will duplicate the last goals for the environments that have less goals.
        for i in range(self._task_cfg.max_num_goals):
            if i == 0:
                # The first goal is picked randomly in a cubic region
                self._target_positions[env_ids, i] = (
                    self._rng.sample_uniform_torch(
                        -self._task_cfg.goal_max_dist_from_origin,
                        self._task_cfg.goal_max_dist_from_origin,
                        3,
                        ids=env_ids,
                    )
                    + self._env_origins[env_ids]
                )
                # Also set a simple target orientation for this task
                self._target_orientations[env_ids] = torch.tensor(
                    [1, 0, 0, 0], device=self._device, dtype=torch.float32
                ).expand(env_ids.shape[0], 4)
            else:
                # If needed, randomize the next goals
                r = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 0], self._gen_actions[env_ids, 1], 1, ids=env_ids
                    )
                    * (self._task_cfg.goal_max_dist - self._task_cfg.goal_min_dist)
                    + self._task_cfg.goal_min_dist
                )
                # Use spherical coordinates for 3D position generation
                phi = self._rng.sample_uniform_torch(0, math.pi, 1, ids=env_ids)  # Inclination
                theta = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)  # Azimuth
                self._target_positions[env_ids, i, 0] = (
                    r * torch.sin(phi) * torch.cos(theta) + self._target_positions[env_ids, i - 1, 0]
                )
                self._target_positions[env_ids, i, 1] = (
                    r * torch.sin(phi) * torch.sin(theta) + self._target_positions[env_ids, i - 1, 1]
                )
                self._target_positions[env_ids, i, 2] = r * torch.cos(phi) + self._target_positions[env_ids, i - 1, 2]
                # Check if the number of goals is less than the current index
                # If it is, then set the ith goal to the num_goal - 1
                self._target_positions[env_ids, i] = torch.where(
                    self._num_goals[env_ids].repeat_interleave(3).reshape(-1, 3) <= i,
                    self._target_positions[env_ids, self._num_goals[env_ids] - 1],
                    self._target_positions[env_ids, i],
                )

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
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
            self._gen_actions[env_ids, 2] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        phi = self._rng.sample_uniform_torch(0, math.pi, 1, ids=env_ids)  # polar angle (inclination)
        theta = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)  # azimuthal angle
        initial_pose[:, 0] = r * torch.sin(phi) * torch.cos(theta) + self._target_positions[env_ids, 0, 0]  # x
        initial_pose[:, 1] = r * torch.sin(phi) * torch.sin(theta) + self._target_positions[env_ids, 0, 1]  # y
        initial_pose[:, 2] = r * torch.cos(phi) + self._target_positions[env_ids, 0, 2]  # z

        # Orientation (quaternion) scaled by difficulty
        rand_quat = self._rng.sample_uniform_torch(-1, 1, 4, ids=env_ids)
        random_quat = torch.where(
            torch.norm(rand_quat, dim=-1, keepdim=True) > EPS,
            rand_quat,
            torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32),
        )
        identity_quat = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32).expand(num_resets, -1)
        initial_pose[:, 3:] = torch.lerp(
            identity_quat,
            torch.nn.functional.normalize(random_quat, dim=-1),
            self._gen_actions[env_ids, 3].unsqueeze(-1),
        )

        # Velocity initialization
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity scaled by difficulty
        velocity_norm = (
            self._gen_actions[env_ids, 4] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        # theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        # phi = torch.rand((num_resets,), device=self._device) * math.pi
        theta = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)
        phi = self._rng.sample_uniform_torch(0, math.pi, 1, ids=env_ids)
        initial_velocity[:, 0] = velocity_norm * torch.sin(phi) * torch.cos(theta)  # vx
        initial_velocity[:, 1] = velocity_norm * torch.sin(phi) * torch.sin(theta)  # vy
        initial_velocity[:, 2] = velocity_norm * torch.cos(phi)  # vz

        # Angular velocity scaled by difficulty
        initial_velocity[:, 3:] = (
            self._gen_actions[env_ids, 5].unsqueeze(-1)
            * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        ) * torch.randn((num_resets, 3), device=self._device)

        # Apply to articulation
        self._robot.set_pose(initial_pose, env_ids)
        self._robot.set_velocity(initial_velocity, env_ids)

    def create_task_visualization(self) -> None:
        """
        Adds the visual marker to the scene.

        There are two markers: one for the goal and one for the robot.

        The goal marker is a small sphere. The target position marker is green while the subsequent goals are red. Passed goals are grey.
        """

        # Define visual markers: sphere for the goal and pose marker for the robot
        goal_marker_cfg_green = SPHERE_CFG.copy()
        goal_marker_cfg_green.markers["sphere"].radius = 0.075
        goal_marker_cfg_green.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # Green
        goal_marker_cfg_grey = SPHERE_CFG.copy()
        goal_marker_cfg_grey.markers["sphere"].radius = 0.075
        goal_marker_cfg_grey.markers["sphere"].visual_material.diffuse_color = (0.5, 0.5, 0.5)  # Grey
        goal_marker_cfg_red = SPHERE_CFG.copy()
        goal_marker_cfg_red.markers["sphere"].radius = 0.075
        goal_marker_cfg_red.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)  # Red

        robot_marker_cfg = SPHERE_CFG.copy()
        robot_marker_cfg.markers["sphere"].radius = 0.01

        # Update prim paths to match task ID
        goal_marker_cfg_red.prim_path = f"/Visuals/Command/task_{self._task_uid}/next_goal"
        goal_marker_cfg_grey.prim_path = f"/Visuals/Command/task_{self._task_uid}/passed_goals"
        goal_marker_cfg_green.prim_path = f"/Visuals/Command/task_{self._task_uid}/current_goals"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # Create the visualization markers
        self.next_goal_visualizer = VisualizationMarkers(goal_marker_cfg_red)
        self.passed_goals_visualizer = VisualizationMarkers(goal_marker_cfg_grey)
        self.current_goals_visualizer = VisualizationMarkers(goal_marker_cfg_green)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """
        Updates the visual marker to the scene.
        Implements the logic to check to use the appropriate colors. It also discards duplicate goals.
        Since the number of goals is flexible, but the length of the tensor is fixed, we need to discard some of the
        goals.
        """

        # The current goals are the ones that the robot is currently trying to reach.
        current_goals = self._target_positions[self._ALL_INDICES, self._target_index]

        # For the remainder of the goals, check if they are passed or upcoming.
        # We do this by iterating over the 2nd axis of the self._target_position tensor.
        # The update time scales linearly with the number of goals.
        passed_goals_list = []
        next_goals_list = []
        for i in range(self._task_cfg.max_num_goals):
            ok_goals = self._num_goals >= i
            next_goals = torch.logical_and(self._target_index < i, ok_goals)
            passed_goals = torch.logical_and(self._target_index > i, ok_goals)
            passed_goals_list.append(self._target_positions[passed_goals, i])
            next_goals_list.append(self._target_positions[next_goals, i])
        passed_goals = (
            torch.cat(passed_goals_list, dim=0) if passed_goals_list else torch.empty((0, 3), device=self._device)
        )
        next_goals = torch.cat(next_goals_list, dim=0) if next_goals_list else torch.empty((0, 3), device=self._device)

        # Assign the positions to the visual markers
        current_goals_pos = current_goals.clone()
        passed_goals_pos = passed_goals.clone()
        next_goals_pos = next_goals.clone()

        # If there are no goals of a given type, hide the markers.
        if next_goals_pos.shape[0] == 0:
            self.next_goal_visualizer.set_visibility(False)
        else:
            self.next_goal_visualizer.set_visibility(True)
            self.next_goal_visualizer.visualize(next_goals_pos)

        if passed_goals_pos.shape[0] == 0:
            self.passed_goals_visualizer.set_visibility(False)
        else:
            self.passed_goals_visualizer.set_visibility(True)
            self.passed_goals_visualizer.visualize(passed_goals_pos)

        if current_goals_pos.shape[0] == 0:
            self.current_goals_visualizer.set_visibility(False)
        else:
            self.current_goals_visualizer.set_visibility(True)
            self.current_goals_visualizer.visualize(current_goals_pos)

        self._robot_marker_pos[:, :3] = self._robot.root_link_pos_w[:, :3]
        self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w)
