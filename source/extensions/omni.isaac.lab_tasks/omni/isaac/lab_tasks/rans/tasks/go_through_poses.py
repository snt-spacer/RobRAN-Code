# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from omni.isaac.lab.markers import BICOLOR_DIAMOND_CFG, PIN_ARROW_CFG, VisualizationMarkers
from omni.isaac.lab.scene import InteractiveScene

from omni.isaac.lab_tasks.rans import GoThroughPosesCfg

from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoThroughPosesTask(TaskCore):
    """
    Implements the GoThroughPosition task. The robot has to reach a target position.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoThroughPosesCfg = GoThroughPosesCfg(),
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
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals, 2),
            device=self._device,
            dtype=torch.float32,
        )
        self._target_heading = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals),
            device=self._device,
            dtype=torch.float32,
        )
        self._target_index = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._trajectory_completed = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._num_goals = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""

        super().create_logs()

        self.scalar_logger.add_log("task_state", "AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/position_distance", "mean")
        self.scalar_logger.add_log("task_state", "AVG/boundary_distance", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/heading", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/progress", "mean")
        self.scalar_logger.add_log("task_reward", "SUM/num_goals", "sum")

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        The observation is given in the robot's frame. The observation is composed of the following elements:
        - The linear velocity of the robot.
        - The angular velocity of the robot.
        - The distance to the target position.
        - The angle between the robot's heading and the target position.
        - The angle between the robot's heading and the target heading.
        - Depending on the task configuration, a number of subsequent poses are added to the observation. For each of
            them, the following elements are added:
            - The distance between the n th and the n+1 th goal.
            - The angle between the n th goal and the n+1 th goal's position (in the current's goal frame).
            - The angle between the n th goal and the n+1 th goal's heading.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        self._task_data[:, 0] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 1] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 2] = The angular velocity of the robot.
        self._task_data[:, 3] = The distance between the robot and the target position.
        self._task_data[:, 4] = The cosine of the angle between the robot's heading and the target position.
        self._task_data[:, 5] = The sine of the angle between the robot's heading and the target position.
        self._task_data[:, 6] = The cosine of the angle between the robot's heading and the target heading.
        self._task_data[:, 7] = The sine of the angle between the robot's heading and the target heading.
        self._task_data[:, 8 + i*5] = The distance between the n th and the n+1 th goal.
        self._task_data[:, 9 + i*5] = The cosine of the angle between the n th goal and the n+1 th goal's position.
        self._task_data[:, 10 + i*5] = The sine of the angle between the n th goal and the n+1 th goal's position.
        self._task_data[:, 11 + i*5] = The cosine of the angle between the n th goal and the n+1 th goal's heading.
        self._task_data[:, 12 + i*5] = The sine of the angle between the n th goal and the n+1 th goal's heading.

        Returns:
            torch.Tensor: The observation tensor."""

        # position error
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)

        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        heading_error = torch.atan2(
            torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
            torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
        )

        # Store in buffer
        self._task_data[:, 0:2] = self._robot.root_com_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 2] = self._robot.root_com_ang_vel_w[self._env_ids, -1]
        self._task_data[:, 3] = self._position_dist
        self._task_data[:, 4] = torch.cos(target_heading_error)
        self._task_data[:, 5] = torch.sin(target_heading_error)
        self._task_data[:, 6] = torch.cos(heading_error)
        self._task_data[:, 7] = torch.sin(heading_error)

        # We compute the observations of the subsequent goals in the previous goal's frame.
        for i in range(self._task_cfg.num_subsequent_goals - 1):
            # Check if the index is looking beyond the number of goals
            overflowing = (self._target_index + i + 1) >= self._num_goals
            # If it is, then set the next index to 0 (Loop around)
            indices = (self._target_index + i + 1) * torch.logical_not(overflowing)
            # Compute the distance between the nth-1 goal, and the nth goal
            goal_distance = torch.linalg.norm(
                self._target_positions[self._ALL_INDICES, indices - 1]
                - self._target_positions[self._ALL_INDICES, indices],
                dim=-1,
            )
            # Compute the angular distance between the nth-1 goal, and the nth goal (world goal frame)
            target_heading_w = torch.atan2(
                self._target_positions[self._ALL_INDICES, indices, 1]
                - self._target_positions[self._ALL_INDICES, indices - 1, 1],
                self._target_positions[self._ALL_INDICES, indices, 0]
                - self._target_positions[self._ALL_INDICES, indices - 1, 0],
            )
            # Compute the heading in the nth-1 goal frame
            target_heading_error = torch.atan2(
                torch.sin(target_heading_w - self._target_heading[self._ALL_INDICES, indices - 1]),
                torch.cos(target_heading_w - self._target_heading[self._ALL_INDICES, indices - 1]),
            )
            # Compute the heading delta between the nth-1 goal, and the nth goal
            heading_error = torch.atan2(
                torch.sin(
                    self._target_heading[self._ALL_INDICES, indices]
                    - self._target_heading[self._ALL_INDICES, indices - 1]
                ),
                torch.cos(
                    self._target_heading[self._ALL_INDICES, indices]
                    - self._target_heading[self._ALL_INDICES, indices - 1]
                ),
            )
            # If the task is not set to loop, we set the next goal to be 0.
            if not self._task_cfg.loop:
                goal_distance = goal_distance * torch.logical_not(overflowing)
                target_heading_error = target_heading_error * torch.logical_not(overflowing)
            # Add to buffer
            self._task_data[:, 8 + 5 * i] = goal_distance
            self._task_data[:, 9 + 5 * i] = torch.cos(target_heading_error)
            self._task_data[:, 10 + 5 * i] = torch.sin(target_heading_error)
            self._task_data[:, 11 + 5 * i] = torch.cos(heading_error)
            self._task_data[:, 12 + 5 * i] = torch.sin(heading_error)

        for randomizer in self.randomizers:
            randomizer.observations(observations=self._task_data)

        # Concatenate the task observations with the robot observations
        return torch.concat((self._task_data, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        heading_error = torch.atan2(
            torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
            torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
        )
        heading_dist = torch.abs(heading_error)

        # position error expressed as distance and angular error (to the position)
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        target_heading_dist = torch.abs(target_heading_error)
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.linalg.norm(self._robot.root_com_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])
        # progress
        progress_rew = self._previous_position_dist - self._position_dist

        # Update logs
        self.scalar_logger.log("task_state", "AVG/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "AVG/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)

        # heading reward (encourages the robot to face the target)
        heading_rew = torch.exp(-heading_dist / self._task_cfg.position_heading_exponential_reward_coeff)

        # target heading reward (encourages the robot to face the target)
        target_heading_rew = torch.exp(-target_heading_dist / self._task_cfg.position_heading_exponential_reward_coeff)

        # linear velocity reward
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        # angular velocity reward
        angular_velocity_rew = angular_velocity - self._task_cfg.angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew
            > (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        ] = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        # boundary rew
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # Checks if the goal is reached
        goal_position_reached = self._position_dist < self._task_cfg.position_tolerance
        goal_orientation_reached = heading_dist < self._task_cfg.heading_tolerance
        goal_reached = goal_position_reached * goal_orientation_reached
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached
        # Check if the trajectory is completed
        self._trajectory_completed = self._target_index > self._num_goals
        # To avoid out of bounds errors, set the target index to 0 if the trajectory is completed
        # If the task loops, then the target index is set to 0 which will make the robot go back to the first goal
        # The episode termination is handled in the get_dones method (looping or not)
        self._target_index = self._target_index * (~self._trajectory_completed)

        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Update logs
        self.scalar_logger.log("task_reward", "AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "AVG/heading", heading_rew)
        self.scalar_logger.log("task_reward", "AVG/progress", progress_rew)
        self.scalar_logger.log("task_reward", "SUM/num_goals", goal_reached)

        # Return the reward by combining the different components and adding the robot rewards
        return (
            progress_rew * self._task_cfg.progress_weight
            + heading_rew * self._task_cfg.position_heading_weight
            + target_heading_rew * self._task_cfg.position_heading_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + self._task_cfg.time_penalty
            + self._task_cfg.reached_bonus * goal_reached
        ) + self._robot.compute_rewards()

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        """
        Resets the task to its initial state.

        If gen_actions is None, then the environment is generated at random. This is the default mode.
        If env_seeds is None, then the seed is generated at random. This is the default mode.

        The environment actions for this task are the following all belong to the [0,1] range:
        - gen_actions[0]: The lower bound of the range used to sample the difference in heading between the goals.
        - gen_actions[1]: The range used to sample the difference in heading between the goals.
        - gen_actions[2]: The lower bound of the range used to sample the distance between the goals.
        - gen_actions[3]: The range used to sample the distance between the goals.
        - gen_actions[4]: The lower bound of the range used to sample the spread of the cone in which the goals are.
        - gen_actions[5]: The range used to sample the spread of the cone in which the goals are.
        - gen_actions[6]: The value used to sample the distance between the spawn position and the first goal.
        - gen_actions[7]: The value used to sample the angle between the spawn position and the first goal.
        - gen_actions[8]: The value used to sample the angle between the spawn heading and the first goal's heading.
        - gen_actions[9]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[10]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            gen_actions (torch.Tensor | None): The actions for the task. Defaults to None.
            env_seeds (torch.Tensor | None): The seeds for the environments. Defaults to None.
        """

        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # Reset the target index and trajectory completed
        self._target_index[env_ids] = 0
        self._trajectory_completed[env_ids] = False

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids, self._target_index[env_ids]]
            - self._robot.root_link_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()

        # The first 6 env actions define ranges, we need to make sure they don't exceed the [0,1] range.
        # They are given as [min, delta] we will convert them to [min, max] that is max = min + delta
        # Note that they are defined as [min, delta] to make sure the min is the min and the max is the max. This
        # is always true as they are strictly positive.
        self._gen_actions[env_ids, 1] = torch.clip(self._gen_actions[env_ids, 0] + self._gen_actions[env_ids, 1], max=1)
        self._gen_actions[env_ids, 3] = torch.clip(self._gen_actions[env_ids, 2] + self._gen_actions[env_ids, 3], max=1)
        self._gen_actions[env_ids, 5] = torch.clip(self._gen_actions[env_ids, 4] + self._gen_actions[env_ids, 5], max=1)

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""

        # Kill robots that would stray too far from the target.
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
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

    def set_goals(self, env_ids: torch.Tensor):
        """
        Generates a random sequence of oriented goals for the task.
        These goals are generated in a way allowing to precisely control the difficulty of the task through the
        environment action. This is done by randomizing ranges within which they can be generated. More information
        below:

        - The first goal is picked randomly in a square centered on the origin. Its orientation is picked randomly. This
            goal is the starting point of the trajectory and it cannot be changed through the environment action. We
            recommend setting that square to be 0. This way, the trajectory will always start at the origin.
        - The next goals are picked randomly in a cone originating from the previous goal.
            - For the heading, the environment action selects the range within which the goal will be picked for
                the whole trajectory. The new heading is picked randomly in a cone aligned with the direction to the
                previous goal. It uses the parameters `minimal_heading_distance` and `maximal_heading_distance`,
                and env_action[0], env_action[1] to set the range. The formula is the following:
                delta_heading = (U[env_action[0],env_action[1]] * (maximal_heading_distance - minimal_heading_distance) + minimal_heading_distance) * rand_sign()
            - For the position, we want to randomize at a given distance from the previous goal, and within a cone aligned
                with the direction to the previous goal. The environment action selects both the distance and the spread of
                the cone. The formula is the following:
                radius = U[env_action[2],env_action[3]] * (maximal_goal_radius - minimal_goal_radius) + minimal_spawn_radius
                spawn_angle_delta = (U[env_action[4],env_action[5]] * (maximal_cone_spread - minimal_cone_spread) + minimal_cone_spread) * rand_sign()
                position_x = radius * cos(spawn_angle_delta + previous_goal_heading) + previous_goal_x
                position_y = radius * sin(spawn_angle_delta + previous_goal_heading) + previous_goal_y

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations."""

        # Select how many random goals we want to generate.
        self._num_goals[env_ids] = self._rng.sample_integer_torch(
            self._task_cfg.min_num_goals, self._task_cfg.max_num_goals, 1, ids=env_ids
        ).to(torch.long)

        # Since we are using tensor operations, we cannot have different number of goals per environment: the
        # tensor containing the target positions must have the same number of goals for all environments.
        # Hence, we will duplicate the last goals for the environments that have less goals.
        for i in range(self._task_cfg.max_num_goals):
            if i == 0:
                # Randomize the first goal
                # The position is picked randomly in a square centered on the origin
                self._target_positions[env_ids, i] = (
                    self._rng.sample_uniform_torch(
                        -self._task_cfg.goal_max_dist_from_origin, self._task_cfg.goal_max_dist_from_origin, 2, env_ids
                    )
                    + self._env_origins[env_ids, :2]
                )
                # The orientation is picked randomly
                self._target_heading[env_ids, i] = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, env_ids)
            else:
                # If needed, randomize the next goals
                # The orientation is picked in a cone aligned with the direction to the previous goal
                self._target_heading[env_ids, i] = self._target_heading[env_ids, i - 1] + (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 0], self._gen_actions[env_ids, 1], 1, env_ids
                    )
                    * (self._task_cfg.goal_max_heading_dist - self._task_cfg.goal_min_heading_dist)
                    + self._task_cfg.goal_min_heading_dist
                ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
                # The position is picked randomly in a cone oriented towards the previous goal.
                # The environment action decides the randomization range of the radius and the spread of the cone.
                r = (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 2],
                        self._gen_actions[env_ids, 3],
                        1,
                        env_ids,
                    )
                    * (self._task_cfg.goal_max_dist - self._task_cfg.goal_min_dist)
                    + self._task_cfg.goal_min_dist
                )
                theta = self._target_heading[env_ids, i - 1] + (
                    self._rng.sample_uniform_torch(
                        self._gen_actions[env_ids, 4],
                        self._gen_actions[env_ids, 5],
                        1,
                        env_ids,
                    )
                    * (self._task_cfg.goal_max_cone_spread - self._task_cfg.goal_min_cone_spread)
                    + self._task_cfg.goal_min_cone_spread
                ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)

                self._target_positions[env_ids, i, 0] = r * torch.cos(theta) + self._target_positions[env_ids, i - 1, 0]
                self._target_positions[env_ids, i, 1] = r * torch.sin(theta) + self._target_positions[env_ids, i - 1, 1]

                # Check if the number of goals is less than the current index
                # If it is, then set the ith goal to the num_goal - 1
                self._target_positions[env_ids, i] = torch.where(
                    self._num_goals[env_ids].repeat_interleave(2).reshape(-1, 2) < i,
                    self._target_positions[env_ids, self._num_goals[env_ids]],
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
            orientation and velocity of the robot."""

        num_resets = len(env_ids)

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)

        # Position, the position is picked in a cone behind the first target.
        r = (
            self._gen_actions[env_ids, 6] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        # We add pi to make sure the robot is behind the first target
        # if the env_action is 0, then the robot is perfectly aligned with the target
        theta = (
            (
                self._gen_actions[env_ids, 7]
                * (self._task_cfg.spawn_max_cone_spread - self._task_cfg.spawn_min_cone_spread)
                + self._task_cfg.spawn_min_cone_spread
            )
            * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            + self._target_heading[env_ids, 0]
            + math.pi
        )
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, 0, 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, 0, 1]
        initial_pose[:, 2] = self._robot_origins[env_ids, 2]
        # Orientation
        delta_heading = (
            (
                self._gen_actions[env_ids, 8]
                * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            )
            + self._task_cfg.spawn_min_heading_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        theta = delta_heading + self._target_heading[env_ids, 0]
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        theta = self._rng.sample_uniform_torch(0, 2 * math.pi, 1, ids=env_ids)
        velocity_norm = (
            self._gen_actions[env_ids, 9] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        initial_velocity[:, 0] = velocity_norm * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._gen_actions[env_ids, 10] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        initial_velocity[:, 5] = angular_velocity

        # Apply to articulation
        self._robot.set_pose(initial_pose, env_ids)
        self._robot.set_velocity(initial_velocity, env_ids)

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are 3 makers for the goals:
        - The next goal is marked in red.
        - The passed goals are marked in grey.
        - The current goals are marked in green.

        They are represented by a pin with an arrow on top of it. The arrow's orientation is the same as the goal's.
        The pin is here to precisely visualize the position of the goal.

        The robot is represented by a diamond with two colors. The colors are used to represent the orientation of the
        robot. The green color represents the front of the robot, and the red color represents the back of the robot.
        """

        # Define the visual markers and edit their properties
        goal_marker_cfg_green = PIN_ARROW_CFG.copy()
        goal_marker_cfg_green.markers["pin_arrow"].visual_material.diffuse_color = (
            0.0,
            1.0,
            0.0,
        )
        goal_marker_cfg_grey = PIN_ARROW_CFG.copy()
        goal_marker_cfg_grey.markers["pin_arrow"].visual_material.diffuse_color = (
            0.5,
            0.5,
            0.5,
        )
        goal_marker_cfg_red = PIN_ARROW_CFG.copy()
        robot_marker_cfg = BICOLOR_DIAMOND_CFG.copy()
        goal_marker_cfg_red.prim_path = f"/Visuals/Command/task_{self._task_uid}/next_goal"
        goal_marker_cfg_grey.prim_path = f"/Visuals/Command/task_{self._task_uid}/passed_goals"
        goal_marker_cfg_green.prim_path = f"/Visuals/Command/task_{self._task_uid}/current_goals"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # We should create only one of them.
        self.next_goal_visualizer = VisualizationMarkers(goal_marker_cfg_red)
        self.passed_goals_visualizer = VisualizationMarkers(goal_marker_cfg_grey)
        self.current_goals_visualizer = VisualizationMarkers(goal_marker_cfg_green)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene.
        This implements the logic to check to use the appropriate colors. It also discards duplicate goals.
        Since the number of goals is flexible, but the length of the tensor is fixed, we need to discard some of the
        goals."""

        # For the reminder of the goals, we need to check if they are passed or not.
        # We do this by iterating over the 2nd axis of the self._target_position tensor.
        # The update time scales linearly with the number of goals.
        passed_goals_pos_list = []
        passed_goals_heading_list = []
        next_goals_pos_list = []
        next_goals_heading_list = []
        for i in range(self._task_cfg.max_num_goals):
            ok_goals = self._num_goals >= i
            passed_goals = torch.logical_and(self._target_index > i, ok_goals)
            passed_goals_pos_list.append(self._target_positions[passed_goals, i])
            passed_goals_heading_list.append(self._target_heading[passed_goals, i])
            next_goals = torch.logical_and(self._target_index < i, ok_goals)
            next_goals_pos_list.append(self._target_positions[next_goals, i])
            next_goals_heading_list.append(self._target_heading[next_goals, i])
        passed_pos_goals = torch.cat(passed_goals_pos_list, dim=0)
        passed_heading_goals = torch.cat(passed_goals_heading_list, dim=0)
        next_pos_goals = torch.cat(next_goals_pos_list, dim=0)
        next_heading_goals = torch.cat(next_goals_heading_list, dim=0)

        # Assign the positions to the visual markers (They need to be dynamically allocated)
        # Under the hood, these are converted to numpy arrays, so that's definitely a waste, but since it's
        # only for visualization, it's not a big deal.
        current_goals_pos = torch.zeros(
            (self._target_positions[self._ALL_INDICES, self._target_index].shape[0], 3),
            device=self._device,
        )
        current_goals_pos[:, :2] = self._target_positions[self._ALL_INDICES, self._target_index]
        passed_goals_pos = torch.zeros((passed_pos_goals.shape[0], 3), device=self._device)
        passed_goals_pos[:, :2] = passed_pos_goals
        next_goals_pos = torch.zeros((next_pos_goals.shape[0], 3), device=self._device)
        next_goals_pos[:, :2] = next_pos_goals

        # Assign the orientations to the visual markers (They need to be dynamically allocated)
        current_goals_quat = torch.zeros(
            (self._target_heading[self._ALL_INDICES, self._target_index].shape[0], 4),
            device=self._device,
        )
        current_goals_quat[:, 0] = torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] * 0.5)
        current_goals_quat[:, 3] = torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] * 0.5)
        passed_goals_quat = torch.zeros((passed_heading_goals.shape[0], 4), device=self._device)
        passed_goals_quat[:, 0] = torch.cos(passed_heading_goals * 0.5)
        passed_goals_quat[:, 3] = torch.sin(passed_heading_goals * 0.5)
        next_goals_quat = torch.zeros((next_heading_goals.shape[0], 4), device=self._device)
        next_goals_quat[:, 0] = torch.cos(next_heading_goals * 0.5)
        next_goals_quat[:, 3] = torch.sin(next_heading_goals * 0.5)

        # If there are no goals of a given type, we should hide the markers.
        if current_goals_pos.shape[0] == 0:
            self.next_goal_visualizer.set_visibility(False)
        else:
            self.next_goal_visualizer.set_visibility(True)
            self.next_goal_visualizer.visualize(next_goals_pos, orientations=next_goals_quat)
        if passed_goals_pos.shape[0] == 0:
            self.passed_goals_visualizer.set_visibility(False)
        else:
            self.passed_goals_visualizer.set_visibility(True)
            self.passed_goals_visualizer.visualize(passed_goals_pos, orientations=passed_goals_quat)
        if next_goals_pos.shape[0] == 0:
            self.current_goals_visualizer.set_visibility(False)
        else:
            self.current_goals_visualizer.set_visibility(True)
            self.current_goals_visualizer.visualize(current_goals_pos, orientations=current_goals_quat)

        # Update the robot visualization. TODO Ideally we should lift the diamond a bit.
        self._robot_marker_pos[:, :2] = self._robot.root_link_pos_w[:, :2]
        self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w)
