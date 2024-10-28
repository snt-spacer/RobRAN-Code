from typing import Tuple
import numpy as np
import wandb
import torch
import math

from omni.isaac.lab.assets import ArticulationData, Articulation
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers import ARROW_CFG
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.math import sample_uniform, sample_gaussian, sample_random_sign
from omni.isaac.lab_tasks.rans import TrackVelocitiesCfg
from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class TrackVelocitiesTask(TaskCore):
    """
    Implements the TrackVelocity task. The robot has to reach a target velocity.
    """

    def __init__(
        self,
        task_cfg: TrackVelocitiesCfg,
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Initializes the TrackVelocities task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task.
        """
        super(TrackVelocitiesTask, self).__init__(task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = 6
        self._dim_env_act = 5

        # Buffers
        self.initialiaze_buffers()

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task.

        Returns:
            dict: The dictionary containing the statistics."""

        super(TrackVelocitiesTask, self).create_logs()

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )
        self._logs["state"]["linear_velocity"] = torch_zeros()
        self._logs["state"]["lateral_velocity"] = torch_zeros()
        self._logs["state"]["angular_velocity"] = torch_zeros()
        self._logs["state"]["linear_velocity_distance"] = torch_zeros()
        self._logs["state"]["lateral_velocity_distance"] = torch_zeros()
        self._logs["state"]["angular_velocity_distance"] = torch_zeros()
        self._logs["reward"]["linear_velocity"] = torch_zeros()
        self._logs["reward"]["lateral_velocity"] = torch_zeros()
        self._logs["reward"]["angular_velocity"] = torch_zeros()

    def initialiaze_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        super(TrackVelocitiesTask, self).initialize_buffers(env_ids)
        # Target velocities
        self._linear_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._lateral_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._angular_velocity_target = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        # Desired velocities
        self._linear_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._lateral_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._angular_velocity_desired = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        # Number of steps (used to compute when to change goals)
        self._num_steps = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)
        self._smoothing_factor = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self._update_after_n_steps = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Returns:
            torch.Tensor: The observation tensor."""

        # linear velocity error
        err_lin_vel = self._linear_velocity_target - self.robot.data.root_lin_vel_b[:, 0]
        # lateral velocity error
        err_lat_vel = self._lateral_velocity_target - self.robot.data.root_lin_vel_b[:, 1]
        # Angular velocity error
        err_ang_vel = self._angular_velocity_target - self.robot.data.root_ang_vel_b[:, 2]

        # Store in buffer
        self._task_data[:, 0] = err_lin_vel * self._task_cfg.enable_linear_velocity
        self._task_data[:, 1] = err_lat_vel * self._task_cfg.enable_lateral_velocity
        self._task_data[:, 2] = err_ang_vel * self._task_cfg.enable_angular_velocity
        self._task_data[:, 3:5] = self.robot.data.root_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 5] = self.robot.data.root_ang_vel_b[self._env_ids, -1]

        # Update logs
        self._logs["state"]["absolute_linear_velocity"] = torch.abs(self.robot.data.root_lin_vel_b[:, 0])
        self._logs["state"]["absolute_lateral_velocity"] = torch.abs(self.robot.data.root_lin_vel_b[:, 1])
        self._logs["state"]["absolute_angular_velocity"] = torch.abs(self.robot.data.root_ang_vel_b[:, 2])
        return self._task_data

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # Linear velocity error
        linear_velocity_distance = torch.abs(self._linear_velocity_target - self.robot.data.root_lin_vel_b[:, 0])
        # Lateral velocity error
        lateral_velocity_distance = torch.abs(self._lateral_velocity_target - self.robot.data.root_lin_vel_b[:, 1])
        # Angular velocity error
        angular_velocity_distance = torch.abs(self._angular_velocity_target - self.robot.data.root_ang_vel_b[:, 2])

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self._logs["state"]["linear_velocity_distance"] = (
            linear_velocity_distance * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["linear_velocity_distance"] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["lateral_velocity_distance"] = (
            lateral_velocity_distance * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["lateral_velocity_distance"] * self._task_cfg.ema_coeff
        )
        self._logs["state"]["angular_velocity_distance"] = (
            angular_velocity_distance * (1 - self._task_cfg.ema_coeff)
            + self._logs["state"]["angular_velocity_distance"] * self._task_cfg.ema_coeff
        )

        # linear velocity reward
        linear_velocity_rew = torch.exp(
            -linear_velocity_distance / self._task_cfg.lin_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_linear_velocity)
        # lateral velocity reward
        lateral_velocity_rew = torch.exp(
            -lateral_velocity_distance / self._task_cfg.lat_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_lateral_velocity)
        # angular velocity reward
        angular_velocity_rew = torch.exp(
            -angular_velocity_distance / self._task_cfg.ang_vel_exponential_reward_coeff
        ) * int(self._task_cfg.enable_angular_velocity)

        # Checks if the goal is reached
        if self._task_cfg.enable_linear_velocity:
            linear_goal_is_reached = (linear_velocity_distance < self._task_cfg.linear_velocity_tolerance).int()
        else:
            linear_goal_is_reached = torch.ones_like(linear_velocity_distance, dtype=torch.int32)
        if self._task_cfg.enable_lateral_velocity:
            lateral_goal_is_reached = (lateral_velocity_distance < self._task_cfg.lateral_velocity_tolerance).int()
        else:
            lateral_goal_is_reached = torch.ones_like(lateral_velocity_distance, dtype=torch.int32)
        if self._task_cfg.enable_angular_velocity:
            angular_vel_goal_is_reached = (angular_velocity_distance < self._task_cfg.angular_velocity_tolerance).int()
        else:
            angular_vel_goal_is_reached = torch.ones_like(angular_velocity_distance, dtype=torch.int32)

        goal_is_reached = linear_goal_is_reached * lateral_goal_is_reached * angular_vel_goal_is_reached
        self._goal_reached += goal_is_reached

        # Update logs (exponential moving average to see the performance at the end of the episode)
        self._logs["reward"]["linear_velocity"] = (
            linear_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["linear_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["lateral_velocity"] = (
            lateral_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["lateral_velocity"] * self._task_cfg.ema_coeff
        )
        self._logs["reward"]["angular_velocity"] = (
            angular_velocity_rew * (1 - self._task_cfg.ema_coeff)
            + self._logs["reward"]["angular_velocity"] * self._task_cfg.ema_coeff
        )
        return (
            linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + lateral_velocity_rew * self._task_cfg.lateral_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
        )

    def reset(self, task_actions: torch.Tensor, env_seeds: torch.Tensor, env_ids: torch.Tensor) -> None:
        """
        Resets the task to its initial state.

        The environment actions for this task are the following all belong to the [0,1] range:
        - env_actions[0]: The value used to sample the target linear velocity.
        - env_actions[1]: The value used to sample the target lateral velocity.
        - env_actions[2]: The value used to sample the target angular velocity.
        - env_actions[4]: The value used to sample the linear velocity of the robot at spawn.
        - env_actions[5]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            task_actions (torch.Tensor): The actions for the task.
            env_seeds (torch.Tensor): The seeds for the environments.
            env_ids (torch.Tensor): The ids of the environments."""

        super().reset(task_actions, env_seeds, env_ids)

        self._num_steps[env_ids] = 0
        self.update_goals()

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not."""

        # Kill the robot if it goes too far, but don't count it as an early termination.
        position_distance = torch.norm(
            self._env_origins[:, :2] - self.robot.data.root_pos_w[self._env_ids, :2], dim=-1
        )
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(position_distance > self._task_cfg.maximum_robot_distance, ones, task_completed)

        # This task cannot be failed.
        zeros = torch.zeros_like(self._goal_reached, dtype=torch.long)
        return zeros, task_completed

    def set_goals(self, env_ids: torch.Tensor) -> None:
        """
        Generates a random goal for the task.
        These goals are generated in a way allowing to precisely control the difficulty of the task through the
        environment action. In this task, the environment actions control 3 different elements:
        - env_actions[0]: The value used to sample the target linear velocity.
        - env_actions[1]: The value used to sample the target lateral velocity.
        - env_actions[2]: The value used to sample the target angular velocity.

        In this tasks goals are constantly updated. The target velocities are updated at regular intervals, and
        an EMA is used to generate target velocities that smoothly change over time. The EMA rate and the interval
        at which the goals are updated are controlled by the task configuration and randomly sampled.
        These cannot be controlled through environment actions.

        Args:
            env_ids (torch.Tensor): The ids of the environments."""

        num_goals = len(env_ids)

        # Set velocity targets
        if self._task_cfg.enable_linear_velocity:
            self._linear_velocity_target[env_ids] = (
                self._env_actions[env_ids, 0] * (self._task_cfg.goal_max_lin_vel - self._task_cfg.goal_min_lin_vel)
                + self._task_cfg.goal_min_lin_vel
            ) * sample_random_sign(num_goals, device=self._device)
            self._linear_velocity_desired[env_ids] = self._linear_velocity_target.clone()
        if self._task_cfg.enable_lateral_velocity:
            self._lateral_velocity_target[env_ids] = (
                self._env_actions[env_ids, 1] * (self._task_cfg.goal_max_lat_vel - self._task_cfg.goal_min_lat_vel)
                + self._task_cfg.goal_min_lat_vel
            ) * sample_random_sign(num_goals, device=self._device)
            self._lateral_velocity_desired[env_ids] = self._lateral_velocity_target.clone()
        if self._task_cfg.enable_angular_velocity:
            self._angular_velocity_target[env_ids] = (
                self._env_actions[env_ids, 2] * (self._task_cfg.goal_max_ang_vel - self._task_cfg.goal_min_ang_vel)
                + self._task_cfg.goal_min_ang_vel
            ) * sample_random_sign(num_goals, device=self._device)
            self._angular_velocity_desired[env_ids] = self._angular_velocity_target.clone()

        # Pick a random smoothing factor
        self._smoothing_factor[env_ids] = (
            torch.rand((num_goals,), device=self._device)
            * (self._task_cfg.smoothing_factor[1] - self._task_cfg.smoothing_factor[0])
            + self._task_cfg.smoothing_factor[0]
        )
        # Pick a random number of steps to update the goals
        self._update_after_n_steps[env_ids] = torch.randint(
            self._task_cfg.interval[0],
            self._task_cfg.interval[1],
            (num_goals,),
            dtype=torch.int32,
            device=self._device,
        )

    def update_goals(self) -> None:
        """
        Updates the goals for the task."""

        # Update the number of steps
        self._num_steps += 1

        # Use EMA to update the target velocities
        if self._task_cfg.enable_linear_velocity:
            self._linear_velocity_target = (
                self._linear_velocity_desired * (1 - self._smoothing_factor)
                + self._linear_velocity_target * self._smoothing_factor
            )
        if self._task_cfg.enable_lateral_velocity:
            self._lateral_velocity_target = (
                self._lateral_velocity_desired * (1 - self._smoothing_factor)
                + self._lateral_velocity_target * self._smoothing_factor
            )
        if self._task_cfg.enable_angular_velocity:
            self._angular_velocity_target = (
                self._angular_velocity_desired * (1 - self._smoothing_factor)
                + self._angular_velocity_target * self._smoothing_factor
            )

        # Check if the goals should be updated
        idx_to_update = torch.where(self._update_after_n_steps < self._num_steps)[0]
        num_updates = len(idx_to_update)
        if num_updates > 0:
            # Update the desired velocities
            if self._task_cfg.enable_linear_velocity:
                self._linear_velocity_desired[idx_to_update] = (
                    self._env_actions[idx_to_update, 0]
                    * (self._task_cfg.goal_max_lin_vel - self._task_cfg.goal_min_lin_vel)
                    + self._task_cfg.goal_min_lin_vel
                ) * torch.sign(torch.rand((num_updates,), device=self._device) - 0.5)
            if self._task_cfg.enable_lateral_velocity:
                self._lateral_velocity_desired[idx_to_update] = (
                    self._env_actions[idx_to_update, 1]
                    * (self._task_cfg.goal_max_lat_vel - self._task_cfg.goal_min_lat_vel)
                    + self._task_cfg.goal_min_lat_vel
                ) * torch.sign(torch.rand((num_updates,), device=self._device) - 0.5)
            if self._task_cfg.enable_angular_velocity:
                self._angular_velocity_desired[idx_to_update] = (
                    self._env_actions[idx_to_update, 2]
                    * (self._task_cfg.goal_max_ang_vel - self._task_cfg.goal_min_ang_vel)
                    + self._task_cfg.goal_min_ang_vel
                ) * torch.sign(torch.rand((num_updates,), device=self._device) - 0.5)
            # Pick a random smoothing factor
            self._smoothing_factor[idx_to_update] = (
                torch.rand((num_updates,), device=self._device)
                * (self._task_cfg.smoothing_factor[1] - self._task_cfg.smoothing_factor[0])
                + self._task_cfg.smoothing_factor[0]
            )
            # Pick a random number of steps to update the goals
            self._update_after_n_steps[idx_to_update] = torch.randint(
                self._task_cfg.interval[0],
                self._task_cfg.interval[1],
                (num_updates,),
                dtype=torch.int32,
                device=self._device,
            )
            self._num_steps[idx_to_update] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Generates the initial conditions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """

        num_resets = len(env_ids)

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)
        # The position is not randomized no point in doing it
        initial_pose[:, :2] = self._env_origins[env_ids, :2]
        # The orientation is not randomized no point in doing it
        initial_pose[:, 3] = 1.0

        # Randomizes the velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        velocity_norm = (
            self._env_actions[env_ids, 3] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = velocity_norm * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._env_actions[env_ids, 4] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        initial_velocity[:, 5] = angular_velocity

        # Apply to articulation
        self.robot.write_root_pose_to_sim(initial_pose, env_ids)  # That's going to break
        self.robot.write_root_velocity_to_sim(initial_velocity, env_ids)

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
        """Updates the visual marker to the scene."""

        # Update the target linear velocity marker
        marker_pos = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        marker_orientation = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        marker_scale = torch.ones((self._num_envs, 3), dtype=torch.float32, device=self._device)
        marker_pos[:, :2] = self.robot.data.root_pos_w[:, :2]
        marker_pos[:, 2] = 0.5
        marker_heading = self.robot.data.heading_w + torch.atan2(
            self._lateral_velocity_target, self._linear_velocity_target
        )
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = (
            torch.norm(torch.stack((self._linear_velocity_target, self._lateral_velocity_target), dim=-1), dim=-1)
            * self._task_cfg.visualization_linear_velocity_scale
        )
        self.goal_linvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)
        # Update the target angular velocity marker
        marker_pos[:, 2] = 0.5
        marker_heading = self.robot.data.heading_w + math.pi / 2.0
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = self._angular_velocity_target * self._task_cfg.visualization_angular_velocity_scale
        self.goal_angvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)

        # Update the robot velocity marker
        marker_pos[:, 2] = 0.7
        if self._task_cfg.enable_lateral_velocity and self._task_cfg.enable_linear_velocity:
            marker_heading = self.robot.data.heading_w + torch.atan2(
                self.robot.data.root_lin_vel_b[:, 1], self.robot.data.root_lin_vel_b[:, 0]
            )
        elif self._task_cfg.enable_linear_velocity:
            marker_heading = self.robot.data.heading_w + math.pi * (self.robot.data.root_lin_vel_b[:, 0] < 0)
        else:
            marker_heading = self.robot.data.heading_w + math.pi / 2.0

        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = (
            torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)
            * self._task_cfg.visualization_linear_velocity_scale
        )
        self.robot_linvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)
        # Update the robot angular velocity marker
        marker_pos[:, 2] = 0.7
        marker_heading = self.robot.data.heading_w + math.pi / 2.0
        marker_orientation[:, 0] = torch.cos(marker_heading * 0.5)
        marker_orientation[:, 3] = torch.sin(marker_heading * 0.5)
        marker_scale[:, 0] = (
            self.robot.data.root_ang_vel_b[:, -1] * self._task_cfg.visualization_angular_velocity_scale
        )
        self.robot_angvel_visualizer.visualize(marker_pos, marker_orientation, marker_scale)
