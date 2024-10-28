from typing import Tuple
import numpy as np
import wandb
import torch
import math

from omni.isaac.lab.assets import ArticulationData, Articulation
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers import PIN_SPHERE_CFG, BICOLOR_DIAMOND_CFG
from omni.isaac.lab.utils.math import sample_uniform, sample_gaussian, sample_random_sign
from omni.isaac.lab_tasks.rans import GoToPositionCfg
from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPositionTask(TaskCore):
    """
    Implements the GoToPosition task. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        task_cfg: GoToPositionCfg,
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Initializes the GoToPosition task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task."""

        super(GoToPositionTask, self).__init__(task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = 6
        self._dim_env_act = 4

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
        self._target_positions = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""

        super(GoToPositionTask, self).create_logs()

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )
        self._logs["state"]["normed_linear_velocity"] = torch_zeros()
        self._logs["state"]["absolute_angular_velocity"] = torch_zeros()
        self._logs["state"]["position_distance"] = torch_zeros()
        self._logs["state"]["boundary_distance"] = torch_zeros()
        self._logs["reward"]["position"] = torch_zeros()
        self._logs["reward"]["linear_velocity"] = torch_zeros()
        self._logs["reward"]["angular_velocity"] = torch_zeros()
        self._logs["reward"]["boundary"] = torch_zeros()

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            robot_data: The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor."""

        # position error
        self._position_error = self._target_positions[:, :2] - self.robot.data.root_pos_w[self._env_ids, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        # position error expressed as distance and angular error (to the position)
        heading = self.robot.data.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self.robot.data.root_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self.robot.data.root_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )

        # Store in buffer
        self._task_data[:, 0] = self._position_dist
        self._task_data[:, 1] = torch.cos(target_heading_error)
        self._task_data[:, 2] = torch.sin(target_heading_error)
        self._task_data[:, 3:5] = self.robot.data.root_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 5] = self.robot.data.root_ang_vel_w[self._env_ids, -1]

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

        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self.robot.data.root_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self.robot.data.root_vel_w[self._env_ids, -1])

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self._logs["state"]["position_distance"] = (
            self._position_dist * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["position_distance"] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["boundary_distance"] = (
            boundary_dist * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["boundary_distance"] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["normed_linear_velocity"] = (
            linear_velocity * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["normed_linear_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["absolute_angular_velocity"] = (
            angular_velocity * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["absolute_angular_velocity"] * self._task_cfg.ema_coeff
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
        self._logs["reward"]["position"] = (
            position_rew * (1 - self._task_cfg.ema_coeff) + self._logs["reward"]["position"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["linear_velocity"] = (
            linear_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["linear_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["angular_velocity"] = (
            angular_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["angular_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["boundary"] = (
            boundary_rew * (1 - self._task_cfg.ema_coeff) + self._logs["reward"]["boundary"] * self._task_cfg.ema_coeff
        )

        return (
            position_rew * self._task_cfg.position_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
        )

    def reset(self, task_actions: torch.Tensor, env_seeds: torch.Tensor, env_ids: torch.Tensor) -> None:
        """
        Resets the task to its initial state.

        The environment actions for this task are the following all belong to the [0,1] range:
        - env_actions[0]: The value used to sample the distance between the spawn position and the goal.
        - env_actions[1]: The value used to sample the angle between the spawn heading and the heading required to be looking at the goal.
        - env_actions[2]: The value used to sample the linear velocity of the robot at spawn.
        - env_actions[3]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            task_actions (torch.Tensor): The actions for the task.
            env_seeds (torch.Tensor): The seeds for the environments.
            env_ids (torch.Tensor): The ids of the environments."""

        super().reset(task_actions, env_seeds, env_ids)

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids] - self.robot.data.root_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not."""

        self._position_error = self._target_positions[:, :2] - self.robot.data.root_pos_w[self._env_ids, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(self._position_dist > self._task_cfg.maximum_robot_distance, ones, task_failed)

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(
            self._goal_reached > self._task_cfg.reset_after_n_steps_in_tolerance, ones, task_completed
        )
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
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations."""

        num_goals = len(env_ids)

        # The position is picked randomly in a square centered on the origin
        self._target_positions[env_ids] = (
            sample_uniform(
                -self._task_cfg.goal_max_dist_from_origin,
                self._task_cfg.goal_max_dist_from_origin,
                (num_goals, 2),
                device=self._device,
            )
            + self._env_origins[env_ids, :2]
        )

        # Update the visual markers
        self._markers_pos[env_ids, :2] = self._target_positions[env_ids]

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

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)

        # Postion
        r = (
            self._env_actions[env_ids, 0] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        theta = sample_uniform(-math.pi, math.pi, (num_resets,), device=self._device)
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, 1]
        initial_pose[:, 2] = self._robot_origins[env_ids, 2]

        # Orientation
        # Compute the heading to the target
        target_heading = torch.arctan2(
            self._target_positions[env_ids, 1] - initial_pose[:, 1],
            self._target_positions[env_ids, 0] - initial_pose[:, 0],
        )
        # Randomizes the heading of the platform
        delta_heading = (
            self._env_actions[env_ids, 1]
            * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            + self._task_cfg.spawn_min_heading_dist
        ) * sample_random_sign(num_resets, device=self._device)
        # The spawn heading is the delta heading + the target heading
        theta = delta_heading + target_heading
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        velocity_norm = (
            self._env_actions[env_ids, 2] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = velocity_norm * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._env_actions[env_ids, 3] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        initial_velocity[:, 5] = angular_velocity

        # Apply to articulation
        self.robot.write_root_pose_to_sim(initial_pose, env_ids)  # That's going to break
        self.robot.write_root_velocity_to_sim(initial_velocity, env_ids)

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are two markers: one for the goal and one for the robot.

        The goal marker is a pin with a red sphere on top. The pin is here to precisely show the position of the goal.

        The robot is represented by a diamond with two colors. The colors are used to represent the orientation of the
        robot. The green color represents the front of the robot, and the red color represents the back of the robot.
        """

        # Define the visual markers and edit their properties
        goal_marker_cfg = PIN_SPHERE_CFG.copy()
        robot_marker_cfg = BICOLOR_DIAMOND_CFG.copy()
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_pose"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # We should create only one of them.
        self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene."""

        self.goal_pos_visualizer.visualize(self._markers_pos)
        self.robot_pos_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)
