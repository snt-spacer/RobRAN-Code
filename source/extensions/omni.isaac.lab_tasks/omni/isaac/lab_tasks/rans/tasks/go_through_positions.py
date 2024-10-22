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
            env_ids: The ids of the environments used by this task."""

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
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals, 2), device=self._device, dtype=torch.float32
        )
        self._target_index = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._trajectory_completed = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._num_goals = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

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
        self._logs["reward"]["heading"] = torch_zeros()
        self._logs["reward"]["progress"] = torch_zeros()
        self._logs["reward"]["num_goals"] = torch_zeros()

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Returns:
            torch.Tensor: The observation tensor."""

        # position error
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self.robot.data.root_pos_w[self._env_ids, :2]
        )
        self._position_dist = torch.norm(self._position_error, dim=-1)

        # position error expressed as distance and angular error (to the position)
        heading = self.robot.data.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self.robot.data.root_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self.robot.data.root_pos_w[self._env_ids, 0],
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
                self.robot.data.root_pos_w[self._env_ids, :2] - self._target_positions[self._ALL_INDICES, indices],
                dim=-1,
            )
            # Compute the heading distance between the nth goal, and the robot
            target_heading_w = torch.atan2(
                self._target_positions[self._ALL_INDICES, indices, 1] - self.robot.data.root_pos_w[self._env_ids, 1],
                self._target_positions[self._ALL_INDICES, indices, 0] - self.robot.data.root_pos_w[self._env_ids, 0],
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
            torch.Tensor: The reward for the current state of the robot."""

        # position error expressed as distance and angular error (to the position)
        heading = self.robot.data.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self.robot.data.root_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self.robot.data.root_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )
        heading_dist = torch.abs(target_heading_error)
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self.robot.data.root_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self.robot.data.root_vel_w[self._env_ids, -1])
        # progress
        progress_rew = self._previous_position_dist - self._position_dist

        # Update logs
        self._logs["state"]["position_distance"] += self._position_dist
        self._logs["state"]["boundary_distance"] += boundary_dist
        self._logs["state"]["normed_linear_velocity"] += linear_velocity
        self._logs["state"]["absolute_angular_velocity"] += angular_velocity

        # heading reward (encourages the robot to face the target)
        heading_rew = torch.exp(-heading_dist / self._task_cfg.position_heading_exponential_reward_coeff)

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
        goal_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached
        self._trajectory_completed = self._target_index >= self._task_cfg.max_num_goals
        # To avoid out of bounds errors, set the target index to 0 if the trajectory is completed
        self._target_index = self._target_index * (~self._trajectory_completed)

        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Update logs
        self._logs["reward"]["linear_velocity"] += linear_velocity_rew
        self._logs["reward"]["angular_velocity"] += angular_velocity_rew
        self._logs["reward"]["boundary"] += boundary_rew
        self._logs["reward"]["heading"] += heading_rew
        self._logs["reward"]["progress"] += progress_rew
        self._logs["reward"]["num_goals"] += goal_reached

        return (
            progress_rew * self._task_cfg.progress_weight
            + heading_rew * self._task_cfg.position_heading_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + self._task_cfg.time_penalty
            + self._task_cfg.reached_bonus * goal_reached
        )

    def reset(self, task_actions: torch.Tensor, env_seeds: torch.Tensor, env_ids: torch.Tensor) -> None:
        super().reset(task_actions, env_seeds, env_ids)
        # Reset the target index and trajectory completed
        self._target_index[env_ids] = 0
        self._trajectory_completed[env_ids] = False
        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids, self._target_index[env_ids]]
            - self.robot.data.root_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_dist[env_ids] = torch.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not."""

        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self.robot.data.root_pos_w[self._env_ids, :2]
        )
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(self._position_dist > self._task_cfg.maximum_robot_distance, ones, task_failed)

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(self._trajectory_completed > 0, ones, task_completed)
        return task_failed, task_completed

    def set_goals(self, env_ids: torch.Tensor):
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations."""

        num_goals = len(env_ids)
        # We are solving this problem in a local frame so it should not matter where the goal is.
        # Yet, we can also move the goal position, and we are going to do it, because why not!

        for i in range(self._task_cfg.max_num_goals):
            if i == 0:
                self._target_positions[env_ids, i] = (
                    torch.rand((num_goals, 2), device=self._device) * self._task_cfg.max_distance_from_origin * 2
                    - self._task_cfg.max_distance_from_origin
                ) + self._env_origins[env_ids, :2]
            else:
                r = (
                    self._env_actions[env_ids, 0]
                    * (self._task_cfg.maximal_spawn_distance - self._task_cfg.minimal_spawn_distance)
                    + self._task_cfg.minimal_spawn_distance
                )
                theta = torch.rand((num_goals,), dtype=torch.float32, device=self._device) * math.pi
                self._target_positions[env_ids, i, 0] = (
                    r * torch.cos(theta) + self._target_positions[env_ids, i - 1, 0]
                )
                self._target_positions[env_ids, i, 1] = (
                    r * torch.sin(theta) + self._target_positions[env_ids, i - 1, 1]
                )

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Generates the initial conditions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot."""

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
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, 0, 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, 0, 1]
        initial_pose[:, 2] = self._robot_origins[env_ids, 2]

        # Orientation
        target_heading = torch.arctan2(
            self._target_positions[env_ids, 0, 1] - initial_pose[:, 1],
            self._target_positions[env_ids, 0, 0] - initial_pose[:, 0],
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

        goal_marker_cfg_green = PIN_SPHERE_CFG.copy()
        goal_marker_cfg_green.markers["pin_sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
        goal_marker_cfg_grey = PIN_SPHERE_CFG.copy()
        goal_marker_cfg_grey.markers["pin_sphere"].visual_material.diffuse_color = (0.5, 0.5, 0.5)
        goal_marker_cfg_red = PIN_SPHERE_CFG.copy()
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

    def update_task_visualization(self):
        """Updates the visual marker to the scene."""

        current_goals = self._target_positions[self._ALL_INDICES, self._target_index]
        passed_goals_list = []
        next_goals_list = []
        for i in range(self._task_cfg.max_num_goals):
            next_goals = self._target_index < i
            passed_goals = self._target_index > i
            passed_goals_list.append(self._target_positions[passed_goals, i])
            next_goals_list.append(self._target_positions[next_goals, i])
        passed_goals = torch.cat(passed_goals_list, dim=0)
        next_goals = torch.cat(next_goals_list, dim=0)
        current_goals_pos = torch.zeros((current_goals.shape[0], 3), device=self._device)
        passed_goals_pos = torch.zeros((passed_goals.shape[0], 3), device=self._device)
        next_goals_pos = torch.zeros((next_goals.shape[0], 3), device=self._device)
        current_goals_pos[:, :2] = current_goals
        passed_goals_pos[:, :2] = passed_goals
        next_goals_pos[:, :2] = next_goals
        if current_goals_pos.shape[0] == 0:
            self.next_goal_visualizer.set_visibility(False)
        else:
            self.next_goal_visualizer.set_visibility(True)
            self.next_goal_visualizer.visualize(next_goals_pos)
        if passed_goals_pos.shape[0] == 0:
            self.passed_goals_visualizer.set_visibility(False)
        else:
            self.passed_goals_visualizer.set_visibility(True)
            self.passed_goals_visualizer.visualize(passed_goals_pos)
        if next_goals_pos.shape[0] == 0:
            self.current_goals_visualizer.set_visibility(False)
        else:
            self.current_goals_visualizer.set_visibility(True)
            self.current_goals_visualizer.visualize(current_goals_pos)
        self.robot_pos_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)
