# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import BICOLOR_DIAMOND_CFG, CYLINDER_MARKER_CFG, VisualizationMarkers
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import CollisionPropertiesCfg, CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

from omni.isaac.lab_tasks.rans import PushBlockCfg

from .task_core import TaskCore

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class PushBlockTask(TaskCore):
    """
    Implements the GoToPosition task. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        task_cfg: PushBlockCfg = PushBlockCfg(),
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

        # Task and reward parameters
        self._task_cfg = task_cfg

        super().__init__(task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers()
        self.design_scene()

    def design_scene(self) -> None:
        """Setups the cube to be pushed in the environment."""
        # Block
        block_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            spawn=CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                mass_props=MassPropertiesCfg(mass=1.0),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.block = RigidObject(block_cfg)

    def register_rigid_objects(self, scene: InteractiveScene) -> None:
        """Registers the rigid objects in the scene."""
        scene.rigid_objects["block"] = self.block

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_robot_block_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_block_target_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_robot_target_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)

        self._position_robot_block_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._position_block_target_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_block_target_dist = torch.zeros(
            (self._num_envs,), device=self._device, dtype=torch.float32
        )
        self._position_robot_target_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)

        self._target_positions = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._block_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""

        super().create_logs()

        self.scalar_logger.add_log("task_state", "AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "EMA/position_distance", "mean")
        self.scalar_logger.add_log("task_state", "EMA/boundary_distance", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/robot_block_position_rew", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/block_target_position_rew", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/boundary", "mean")

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            robot_data: The current state of the robot.

        self._task_data[:, 0] = The distance between the robot and the block position
        self._task_data[:, 1] = The distance between the block and the target position.
        self._task_data[:, 2] = The cosine of the angle between the robot heading and the block position.
        self._task_data[:, 3] = The sine of the angle between the robot heading and the block position.
        self._task_data[:, 4] = The cosine of the angle between the robot heading and the target position.
        self._task_data[:, 5] = The sine of the angle between the robot heading and the target position.
        self._task_data[:, 6] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 7] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 8] = The angular velocity of the robot.

        Returns:
            torch.Tensor: The observation tensor."""

        # Position error between robot and block
        self._position_robot_block_error = self._block_positions[:, :2] - self._robot.root_pos_w[self._env_ids, :2]
        self._position_robot_block_dist = torch.norm(self._position_robot_block_error, dim=-1)

        # Position error between block and target
        self._position_block_target_error = self._target_positions[:, :2] - self._block_positions[:, :2]
        self._previous_position_block_target_dist = self._position_block_target_dist
        self._position_block_target_dist = torch.norm(self._position_block_target_error, dim=-1)

        # Position error between robot and target
        self._position_robot_target_error = self._target_positions[:, :2] - self._robot.root_pos_w[self._env_ids, :2]
        self._position_robot_target_dist = torch.norm(self._position_robot_target_error, dim=-1)

        heading = self._robot.heading_w[self._env_ids]
        # Position error between robot and block expressed as distance and angular error (robot to block)
        block_heading_w = torch.atan2(
            self._block_positions[:, 1] - self._robot.root_pos_w[self._env_ids, 1],
            self._block_positions[:, 0] - self._robot.root_pos_w[self._env_ids, 0],
        )
        block_heading_error = torch.atan2(torch.sin(block_heading_w - heading), torch.cos(block_heading_w - heading))
        # Position error between robot and target expressed as distance and angular error (robot to target)
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Store in buffer
        self._task_data[:, 0] = self._position_robot_block_dist
        self._task_data[:, 1] = self._position_block_target_dist
        self._task_data[:, 2] = torch.cos(block_heading_error)
        self._task_data[:, 3] = torch.sin(block_heading_error)
        self._task_data[:, 4] = torch.cos(target_heading_error)
        self._task_data[:, 5] = torch.sin(target_heading_error)
        self._task_data[:, 6:8] = self._robot.root_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 8] = self._robot.root_ang_vel_w[self._env_ids, -1]

        # Concatenate the task observations with the robot observations
        return torch.concat((self._task_data, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        The observation is given in the robot's frame. The task provides 4 elements:
        - The position of the block in the robot's frame. It is expressed as the distance between the robot and
            the block position, and the angle between the robot's heading and the block position.
        - The position of the target in the robot's frame. It is expressed as the distance between the block and
            the target position, and the angle between the robot's heading and the target position.
        - The linear velocity of the robot in the robot's frame.
        - The angular velocity of the robot in the robot's frame.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        The observation tensor is composed of the following elements:
        - self._task_data[:, 0] = The distance between the robot and the block position
        - self._task_data[:, 1] = The distance between the block and the target position.
        - self._task_data[:, 2] = The cosine of the angle between the robot heading and the block position.
        - self._task_data[:, 3] = The sine of the angle between the robot heading and the block position.
        - self._task_data[:, 4] = The cosine of the angle between the robot heading and the target position.
        - self._task_data[:, 5] = The sine of the angle between the robot heading and the target position.
        - self._task_data[:, 6] = The linear velocity of the robot along the x-axis.
        - self._task_data[:, 7] = The linear velocity of the robot along the y-axis.
        - self._task_data[:, 8] = The angular velocity of the robot.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # boundary rew
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_robot_target_dist)
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # linear velocity reward
        linear_velocity = torch.norm(self._robot.root_vel_w[self._env_ids, :2], dim=-1)
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)

        # angular velocity reward
        angular_velocity = torch.abs(self._robot.root_vel_w[self._env_ids, -1])
        angular_velocity_rew = angular_velocity - self._task_cfg.angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew
            > (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        ] = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)

        # position reward
        robot_block_position_rew = 1 - torch.clamp(
            torch.linalg.norm(self._block_positions[:, :2] - self._robot.root_pos_w[self._env_ids, :2], dim=-1)
            / (self._task_cfg.max_delta_robot_position + EPS),
            min=0.0,
            max=1.0,
        )
        # Normalized position robot block distance

        block_target_position_rew = 1 - torch.clamp(
            torch.linalg.norm(self._block_positions[:, :2] - self._target_positions[:, :2], dim=-1)
            / (self._task_cfg.cube_dist_from_origin + EPS),
            min=0.0,
            max=1.0,
        )
        # Normalized position block target distance (aligned with the env_origins)
        goal_is_reached = (self._position_block_target_dist < self._task_cfg.position_tolerance).int()

        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        goal_is_reached_rew = (self._goal_reached == self._task_cfg.reset_after_n_steps_in_tolerance).int()

        # # Update logs (exponential moving average to see the performance at the end of the episode)
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)
        self.scalar_logger.log("task_reward", "AVG/robot_block_position_rew", robot_block_position_rew)
        self.scalar_logger.log("task_reward", "AVG/block_target_position_rew", block_target_position_rew)
        self.scalar_logger.log("task_reward", "AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew)

        # Return the reward by combining the different components and adding the robot rewards
        return (
            robot_block_position_rew * self._task_cfg.robot_block_position_weight
            + block_target_position_rew * self._task_cfg.block_target_position_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + goal_is_reached_rew * self._task_cfg.goal_reached_weight
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
        - gen_actions[0]: The value used to sample the distance between the spawn position of the robot and the goal.
        - gen_actions[1]: The value used to sample the angle between the spawn heading and the heading required to be looking at the goal.
        - gen_actions[2]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[3]: The value used to sample the angular velocity of the robot at spawn.
        - gen_actions[4]: The value used to sample the theta of the spawn position distance of the block.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            gen_actions (torch.Tensor | None): The actions for the task. Defaults to None.
            env_seeds (torch.Tensor | None): The seeds for the environments. Defaults to None.
        """

        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # Reset the block
        self.block.reset(env_ids)
        self._block_positions = self.block.data.root_state_w[:, :3]

        # Make sure the position error and position dist are up to date after the reset
        # Position error between robot and block
        self._position_robot_block_error[env_ids] = (
            self._block_positions[env_ids, :2] - self._robot.root_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_robot_block_dist[env_ids] = torch.norm(self._position_robot_block_error[env_ids], dim=-1)

        # Position error between block and target
        self._position_block_target_error[env_ids] = (
            self._target_positions[env_ids, :2] - self._block_positions[env_ids, :2]
        )
        self._position_block_target_dist[env_ids] = torch.norm(self._position_block_target_error[env_ids], dim=-1)

        # Position error between robot and target
        self._position_robot_target_error[env_ids] = (
            self._target_positions[env_ids, :2] - self._robot.root_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_robot_target_dist[env_ids] = torch.norm(self._position_robot_target_error[env_ids], dim=-1)

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""

        self._position_error = self._target_positions[:, :2] - self._robot.root_pos_w[self._env_ids, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self._task_cfg.maximum_robot_distance,
            ones,
            task_failed,
        )

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(
            self._goal_reached >= self._task_cfg.reset_after_n_steps_in_tolerance,
            ones,
            task_completed,
        )

        # update block position
        self._block_positions = self.block.data.root_state_w[:, :3]

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

        # The target position is picked randomly in a square centered on the origin
        self._target_positions[env_ids] = (
            self._rng.sample_uniform_torch(
                -self._task_cfg.goal_max_dist_from_origin, self._task_cfg.goal_max_dist_from_origin, 2, ids=env_ids
            )
            + self._env_origins[env_ids, :2]
        )
        # Update the visual markers
        self._markers_pos[env_ids, :2] = self._target_positions[env_ids]

        # Cube position. Picked randomly thetha and R from the origin
        theta = self._gen_actions[env_ids, 4] * 2 * math.pi
        x = self._task_cfg.cube_dist_from_origin * torch.cos(theta)
        y = self._task_cfg.cube_dist_from_origin * torch.sin(theta)
        z = torch.full((num_goals,), 0.1, device=self._device)
        self._block_positions[env_ids] = torch.stack((x, y, z), dim=-1) + self._env_origins[env_ids]

        block_pos = self.block.data.root_state_w[env_ids, :7]
        block_pos[:, :3] = self._block_positions[env_ids]
        self.block.write_root_pose_to_sim(block_pos, env_ids)

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

        # Position
        r = (
            self._gen_actions[env_ids, 0] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        theta = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)
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
            self._gen_actions[env_ids, 1]
            * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            + self._task_cfg.spawn_min_heading_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        # The spawn heading is the delta heading + the target heading
        theta = delta_heading + target_heading
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        velocity_norm = (
            self._gen_actions[env_ids, 2] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = velocity_norm * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._gen_actions[env_ids, 3] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        initial_velocity[:, 5] = angular_velocity

        # Apply to articulation
        self._robot.set_pose(initial_pose, env_ids)
        self._robot.set_velocity(initial_velocity, env_ids)

        block_pos = self.block.data.root_state_w[env_ids, :7]
        block_pos[:, :3] = self._block_positions[env_ids]
        self.block.write_root_pose_to_sim(block_pos, env_ids)

        block_pos = self.block.data.root_state_w[env_ids, :7]
        block_pos[:, :3] = self._block_positions[env_ids]
        self.block.write_root_pose_to_sim(block_pos, env_ids)

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are two markers: one for the goal and one for the robot.

        The goal marker is a pin with a red sphere on top. The pin is here to precisely show the position of the goal.

        The robot is represented by a diamond with two colors. The colors are used to represent the orientation of the
        robot. The green color represents the front of the robot, and the red color represents the back of the robot.
        """

        # Define the visual markers and edit their properties
        goal_marker_cfg = CYLINDER_MARKER_CFG.copy()
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_pose"
        goal_marker_cfg.markers["cylinder"].radius = self._task_cfg.position_tolerance
        goal_marker_cfg.markers["cylinder"].height = 0.01

        robot_marker_cfg = BICOLOR_DIAMOND_CFG.copy()
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"

        # We should create only one of them.
        self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene."""

        self.goal_pos_visualizer.visualize(self._markers_pos)
        self.robot_pos_visualizer.visualize(self._robot.root_pos_w, self._robot.root_quat_w)
