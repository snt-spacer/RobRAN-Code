from typing import Tuple
import numpy as np
import wandb
import torch
import math

from omni.isaac.lab.assets import ArticulationData, Articulation
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers import PIN_SPHERE_CFG, BICOLOR_DIAMOND_CFG

from omni.isaac.lab_tasks.rans import GoThroughPositionsCfg
from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoThroughPositionsTask(TaskCore):
    """
    Implements the GoThroughPosition task. The robot has to reach a target position.
    """

    def __init__(
        self,
        task_cfg: GoThroughPositionsCfg,
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Initializes the GoThroughPosition task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task.
        """
        super(GoThroughPositionsTask, self).__init__(
            task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids
        )

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = 3 + 3 * self._task_cfg.num_subsequent_goals
        self._dim_env_act = 4  # spawn distance, spawn_cone, linear velocity, angular velocity

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals, 2), device=self._device, dtype=torch.float32
        )
        self._target_index = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""

        super(GoThroughPositionsTask, self).create_logs()

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )
        self._logs["state"]["normed_linear_velocity"] = torch_zeros()
        self._logs["state"]["absolute_angular_velocity"] = torch_zeros()
        self._logs["state"]["position_distance"] = torch_zeros()
        self._logs["state"]["boundary_distance"] = torch_zeros()
        self._logs["reward"]["progress"] = torch_zeros()
        self._logs["reward"]["heading"] = torch_zeros()
        self._logs["reward"]["linear_velocity"] = torch_zeros()
        self._logs["reward"]["angular_velocity"] = torch_zeros()
        self._logs["reward"]["boundary"] = torch_zeros()

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            robot_data: The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor.
        """

        # position error
        self._position_error = (
            self._target_positions[:, self._target_index, :2] - self.robot.data.root_pos_w[self._env_ids, :2]
        )
        self._position_dist = torch.norm(self._position_error, dim=-1)

        # position error expressed as distance and angular error (to the position)
        heading = self.robot.data.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, self._target_index, 1] - self.robot.data.root_pos_w[self._env_ids, 1],
            self._target_positions[:, self._target_index, 0] - self.robot.data.root_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )

        # Store in buffer
        self._task_data[:, 0:2] = self.robot.data.root_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 2] = self.robot.data.root_ang_vel_b[self._env_ids, -1]
        self._task_data[:, 3] = self._position_dist
        self._task_data[:, 4] = torch.cos(target_heading_error)
        self._task_data[:, 5] = torch.sin(target_heading_error)
        # We compute the observations of the subsequent goals in the robot frame as the goals are not oriented.
        for i in range(self._task_cfg.num_subsequent_goals - 1):
            # Check if the index is looking beyond the number of goals
            overflowing = (self._target_index + i + 1 >= self._task_cfg.max_num_goals).int()
            # If it is, then set the next index to 0 (Loop around)
            indices = self._target_index + (i + 1) * (1 - overflowing)
            # Compute the distance between the nth goal, and the robot
            goal_distance = torch.norm(
                self.robot.data.root_pos_w[self._env_ids, 2] - self._target_positions[:, indices, :2],
                dim=-1,
            )
            # Compute the heading distance between the nth goal, and the robot
            target_heading_w = torch.atan2(
                self._target_positions[:, indices, 1] - self.robot.data.root_pos_w[self._env_ids, 1],
                self._target_positions[:, indices, 0] - self.robot.data.root_pos_w[self._env_ids, 0],
            )
            target_heading_error = torch.atan2(
                torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
            )
            # If the task is not set to loop, we set the next goal to be 0.
            if ~self._task_cfg.loop:
                goal_distance = goal_distance * (1 - overflowing)
                target_heading_error = target_heading_error * (1 - overflowing)
            # Add to buffer
            self._task_data[:, 6 + 3 * i] = goal_distance
            self._task_data[:, 7 + 3 * i] = torch.cos(target_heading_error)
            self._task_data[:, 8 + 3 * i] = torch.sin(target_heading_error)
        return self._task_data

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot.
        """
        # position distance
        # self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self.robot.data.root_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self.robot.data.root_vel_w[self._env_ids, -1])
        # progress
        progress = self._previous_position_dist - self._position_dist

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self._logs["state"]["position_distance"][self._env_ids] = (
            self._position_dist * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["position_distance"][self._env_ids] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["boundary_distance"][self._env_ids] = (
            boundary_dist * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["boundary_distance"][self._env_ids] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["normed_linear_velocity"][self._env_ids] = (
            linear_velocity * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["normed_linear_velocity"][self._env_ids] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["absolute_angular_velocity"][self._env_ids] = (
            angular_velocity * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["absolute_angular_velocity"][self._env_ids] * self._task_cfg.ema_coeff
        )

        # position reward
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)
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
        goal_is_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self._logs["reward"]["position"][self._env_ids] = (
            position_rew * (1 - self._task_cfg.ema_coeff) + self._logs["reward"]["position"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["linear_velocity"][self._env_ids] = (
            linear_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["linear_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["angular_velocity"][self._env_ids] = (
            angular_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["angular_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["boundary"][self._env_ids] = (
            boundary_rew * (1 - self._task_cfg.ema_coeff) + self._logs["reward"]["boundary"] * self._task_cfg.ema_coeff
        )

        return (
            position_rew * self._task_cfg.position_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """
        self._position_error = self._target_positions[:, :2] - self.robot.data.root_pos_w[self._env_ids, :2]
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(self._position_dist > self._task_cfg.maximum_robot_distance, ones, task_failed)

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(
            self._goal_reached > self._task_cfg.reset_after_n_steps_in_tolerance, ones, task_completed
        )
        return task_failed, task_completed

    def set_goals(self, env_ids: torch.Tensor):
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations.
        """
        num_goals = len(env_ids)
        # We are solving this problem in a local frame so it should not matter where the goal is.
        # Yet, we can also move the goal position, and we are going to do it, because why not!

        self._target_positions[env_ids] = (
            torch.rand((num_goals, 2), device=self._device) * self._task_cfg.max_distance_from_origin * 2
            - self._task_cfg.max_distance_from_origin
        ) + self._env_origins[env_ids, :2]

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Generates the initial conditions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot.
        """

        num_resets = len(env_ids)

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)

        # Postion
        r = (
            self._env_actions[env_ids, 0]
            * (self._task_cfg.maximal_spawn_distance - self._task_cfg.minimal_spawn_distance)
            + self._task_cfg.minimal_spawn_distance
        )
        theta = torch.rand((num_resets,), dtype=torch.float32, device=self._device) * math.pi
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, 1]
        initial_pose[:, 2] = self._robot_origins[env_ids, 2]

        # Orientation
        target_heading = torch.arctan2(
            self._target_positions[env_ids, 1] - initial_pose[:, 1],
            self._target_positions[env_ids, 0] - initial_pose[:, 0],
        )
        sampled_heading = (
            self._env_actions[env_ids, 1]
            * (self._task_cfg.maximal_spawn_cone - self._task_cfg.minimal_spawn_cone)
            * torch.sign(torch.rand((num_resets,), device=self._device) - 0.5)
        ) + self._task_cfg.minimal_spawn_cone
        theta = sampled_heading + target_heading
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        velocity_norm = (
            self._env_actions[env_ids, 2]
            * (self._task_cfg.maximal_spawn_linear_velocity - self._task_cfg.minimal_spawn_linear_velocity)
            + self._task_cfg.minimal_spawn_linear_velocity
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = velocity_norm * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._env_actions[env_ids, 3]
            * (self._task_cfg.maximal_spawn_angular_velocity - self._task_cfg.minimal_spawn_angular_velocity)
            + self._task_cfg.minimal_spawn_angular_velocity
        )
        initial_velocity[:, 5] = angular_velocity

        # Apply to articulation
        self.robot.write_root_pose_to_sim(initial_pose, env_ids)  # That's going to break
        self.robot.write_root_velocity_to_sim(initial_velocity, env_ids)

    def create_task_visualization(self):
        """Adds the visual marker to the scene."""
        goal_marker_cfg = PIN_SPHERE_CFG.copy()
        robot_marker_cfg = BICOLOR_DIAMOND_CFG.copy()
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_pose"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # We should create only one of them.
        self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self):
        """Updates the visual marker to the scene."""
        marker_pos = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        marker_pos[:, :2] = self._target_positions
        self.goal_pos_visualizer.visualize(marker_pos)
        self.robot_pos_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)
