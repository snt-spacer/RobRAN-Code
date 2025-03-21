# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene

from isaaclab_tasks.rans import GoToPositionWithObstaclesCfg
from isaaclab_tasks.rans.utils import ObjectStorage

from .go_to_position import GoToPositionTask

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPositionWithObstaclesTask(GoToPositionTask):
    """
    Implements the GoToPosition task. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoToPositionWithObstaclesCfg = GoToPositionWithObstaclesCfg(),
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

        self._task_cfg = task_cfg

        super().__init__(
            scene=scene, task_cfg=task_cfg, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids
        )

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers()

        # Obstacles
        self.obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=self._task_cfg.max_num_vis_obstacles,
            store_height=self._task_cfg.obstacles_storage_height_pos,
            rng=self._rng,
            device=device,
        )
        self.batch_indices = (
            torch.arange(num_envs, device=self._device).unsqueeze(1).expand(-1, self._task_cfg.max_num_vis_obstacles)
        )

        self._num_cells = int(1.0 / (self._task_cfg.minimum_point_distance * 2))

        self.design_scene()

    def register_robot(self, robot) -> None:
        self._robot = robot

    def register_sensors(self) -> None:
        filters = [f"/World/envs/env_.*/Obstacles/cylinder_{i}" for i in range(self._task_cfg.max_num_vis_obstacles)]
        self._robot.activateSensors("contacts", filters)
        self._robot.register_sensors()

    def create_logs(self) -> None:
        super().create_logs()
        self.scalar_logger.add_log("task_reward", "SUM/num_collisions", "sum")
        self.scalar_logger.add_log("task_reward", "AVG/progress", "mean")

    def design_scene(self) -> None:
        """
        Initializes the obstacles for the task.
        """

        prim_utils.create_prim("/World/envs/env_0/Obstacles", "Xform")

        rigid_objects = {}
        low, high = 5, 15

        for i in range(self._task_cfg.max_num_vis_obstacles):

            position = low + (high - low) * torch.rand(3)
            position[2] = 0.5

            rigid_objects[f"obstacle_{i}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacles/cylinder_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self._task_cfg.obstacle_radius,
                    height=self._task_cfg.obstacles_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            )

        obstacles_cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        self.obstacles = RigidObjectCollection(obstacles_cfg)

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""
        super().initialize_buffers(env_ids)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)

    def run_setup(self, robot, envs_origin):
        super().run_setup(robot, envs_origin)
        self.obstacles_generator.create_storage_buffer(env_origin=self._env_origins)

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            robot_data: The current state of the robot.

        self._task_data[:, 0] = The distance between the robot and the target position.
        self._task_data[:, 1] = The cosine of the angle between the robot heading and the target position.
        self._task_data[:, 2] = The sine of the angle between the robot heading and the target position.
        self._task_data[:, 3] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 4] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 5] = The angular velocity of the robot.
        self._task_data[:, 6:6 + self._task_cfg.num_obstacles] = The distance between the robot and the obstacles.
        self.task_data[:, 6 + self._task_cfg.num_obstacles: 6 + 2 * self._task_cfg.num_obstacles] = The cosine of the angle between the robot and the obstacles.
        self.task_data[:, 6 + 2 * self._task_cfg.num_obstacles: 6 + 3 * self._task_cfg.num_obstacles] = The sine of the angle between the robot and the obstacles.

        Returns:
            torch.Tensor: The observation tensor."""
        # position error
        self._position_error = self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Obstacles positions
        # Filter obstacles by height
        obstacles_positions = self.obstacles.data.object_link_pos_w[self._env_ids]
        filtered_obstacles = obstacles_positions.clone()

        mask = obstacles_positions[:, :, 2] < 0
        filtered_obstacles[mask] = 2 * self._task_cfg.max_obstacle_distance_from_target

        # Calculate distances and angles for the filtered obstacles
        obstacles_error = filtered_obstacles[:, :, :2] - self._robot.root_link_pos_w[self._env_ids, :2].unsqueeze(1)
        obstacles_dist = torch.norm(obstacles_error, dim=-1)

        # Get the 3 closest obstacles
        closest_distances, closest_indices = torch.topk(obstacles_dist, k=3, dim=1, largest=False)
        closest_obstacles = torch.gather(
            filtered_obstacles, 1, closest_indices.unsqueeze(-1).expand(-1, -1, filtered_obstacles.size(-1))
        )

        obstacles_heading = torch.atan2(
            closest_obstacles[self._env_ids, :, 1] - self._robot.root_link_pos_w[self._env_ids, 1].unsqueeze(1),
            closest_obstacles[self._env_ids, :, 0] - self._robot.root_link_pos_w[self._env_ids, 0].unsqueeze(1),
        )
        obstacles_heading_error = torch.atan2(
            torch.sin(obstacles_heading - heading.unsqueeze(1)), torch.cos(obstacles_heading - heading.unsqueeze(1))
        )

        # Store in buffer [distance, cos(angle), sin(angle), lin_vel_x, lin_vel_y, ang_vel, obstacles_dist, obstacles_cos_angle, obstacles_sin_angle]
        self._task_data[:, 0] = self._position_dist
        self._task_data[:, 1] = torch.cos(target_heading_error)
        self._task_data[:, 2] = torch.sin(target_heading_error)
        self._task_data[:, 3:5] = self._robot.root_com_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 5] = self._robot.root_com_ang_vel_w[self._env_ids, -1]
        self._task_data[:, 6:9] = closest_distances
        self._task_data[:, 9:12] = torch.cos(obstacles_heading_error)
        self._task_data[:, 12:15] = torch.sin(obstacles_heading_error)

        # Concatenate the task observations with the robot observations
        return torch.concat((self._task_data, self._robot.get_observations()), dim=-1)

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        The observation is given in the robot's frame. The task provides 3 elements:
        - The position of the object in the robot's frame. It is expressed as the distance between the robot and
            the target position, and the angle between the robot's heading and the target position.
        - The linear velocity of the robot in the robot's frame.
        - The angular velocity of the robot in the robot's frame.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        The observation tensor is composed of the following elements:
        - self._task_data[:, 0]: The distance between the robot and the target position.
        - self._task_data[:, 1]: The cosine of the angle between the robot's heading and the target position.
        - self._task_data[:, 2]: The sine of the angle between the robot's heading and the target position.
        - self._task_data[:, 3]: The linear velocity of the robot along the x-axis.
        - self._task_data[:, 4]: The linear velocity of the robot along the y-axis.
        - self._task_data[:, 5]: The angular velocity of the robot.
        - self._task_data[:, 6:10] = The distance between the robot and the obstacles.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self._robot.root_com_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])

        # Update logs
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", linear_velocity)

        # position reward
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)
        # progress
        progress_rew = self._previous_position_dist - self._position_dist
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
        reached_ids = goal_is_reached.nonzero(as_tuple=False).squeeze(-1)
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # Check for collision with obstacles
        collisions = torch.squeeze(
            torch.max(torch.norm(self._robot.contacts.data.force_matrix_w, dim=-1), dim=-1)[0], dim=-1
        )  # first max is for the 3 forces (x,y,z), second max is for obstacles

        num_collisions = 1 * (collisions > self._task_cfg.collision_threshold)
        collision_penalty_rew = self._task_cfg.collision_penalty * num_collisions

        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Update logs for rewards
        self.scalar_logger.log("task_reward", "AVG/position", position_rew)
        self.scalar_logger.log("task_reward", "AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "AVG/progress", progress_rew)
        self.scalar_logger.log("task_reward", "SUM/num_collisions", num_collisions)

        # Return the reward by combining the different components and adding the robot rewards
        return (
            progress_rew * self._task_cfg.progress_weight
            + position_rew * self._task_cfg.position_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + collision_penalty_rew
        ) + self._robot.compute_rewards()

    def reset(self, env_ids, gen_actions=None, env_seeds=None):
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._previous_position_dist = self._position_dist.clone()
        return super().get_dones()

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        super().set_initial_conditions(env_ids)

        obstacles_positions, mask = self.randomize_obstacles_positions(env_ids)
        pos_obstacles_in_env = self.obstacles_generator.get_positions_with_storage(obstacles_positions, mask, env_ids)
        # pos_obstacles_in_env[:, :, 3:] = self.obstacles.data.object_com_quat_w[env_ids]
        self.obstacles.write_object_link_pose_to_sim(pos_obstacles_in_env, env_ids=env_ids)

    def randomize_obstacles_positions(self, env_ids: torch.tensor) -> tuple:
        """
        This function randomizes the positions of obstacles within a specified environment in a grid-based layout, ensuring that they are not placed too close to the target or robot. It also generates random orientations for the obstacles and creates a mask indicating which obstacles are visible.

        Args:
            env_ids (torch.tensor): The ids of the environments to randomize the obstacles in.

        Returns:
            tuple: A tuple containing the positions of the obstacles and a mask indicating which obstacles are visible.
        """

        # Randomize obstacles positions
        number_obstacles_to_generate = self._task_cfg.max_num_vis_obstacles * 4
        indices_of_obstacles_to_activate = self._rng.sample_unique_integers_torch(
            min=0, max=self._task_cfg.max_num_vis_obstacles**2, num=number_obstacles_to_generate, ids=env_ids
        )

        x = (indices_of_obstacles_to_activate % self._num_cells) - self._num_cells / 2
        y = (indices_of_obstacles_to_activate // self._num_cells) - self._num_cells / 2
        # Scale to world coordinates
        cell_size = self._task_cfg.maximum_robot_distance / self._num_cells  # Calculate the size of a grid cell
        x = self._rng.sample_sign_torch("int", (number_obstacles_to_generate), ids=env_ids) * x * cell_size / 2
        y = self._rng.sample_sign_torch("int", (number_obstacles_to_generate), ids=env_ids) * y * cell_size / 2
        z = torch.ones_like(x) * self._task_cfg.obstacles_height / 2
        xyz = torch.stack((x, y, z), dim=2)
        xyz[..., :2] += self._env_origins[env_ids].unsqueeze(1)[..., :2]

        """Move the obstacle if it is too close to target position or the robot position"""
        # Calculate the distances between the obstacles and the target and robot positions
        distance_obstacle_to_target = torch.norm(xyz[..., :2] - self._target_positions[env_ids].unsqueeze(1), dim=-1)
        distance_obstacle_to_robot = torch.norm(
            xyz[..., :2] - self._robot.root_link_pos_w[env_ids][..., :2].unsqueeze(1), dim=-1
        )
        # Create a mask to filter out obstacles that are too close to the target or robot
        obstacles_mask = (distance_obstacle_to_target < self._task_cfg.min_obstacle_distance_from_target) | (
            distance_obstacle_to_robot < self._task_cfg.min_obstacle_distance_from_robot
        )

        # Create indices for obstacles that are too close to the target or robot
        valid_indices = (~obstacles_mask).nonzero(as_tuple=True)[1]
        invalid_indices = obstacles_mask.nonzero(as_tuple=True)[1]
        valid_obstacles = xyz[env_ids, valid_indices]

        # Swap obstacles that are too close to the target or robot with valid obstacles
        if len(env_ids) == 1:
            valid_obstacles = valid_obstacles.unsqueeze(0)

        if self._task_cfg.max_num_vis_obstacles > valid_obstacles.shape[1]:
            replacement_indices = self._rng.sample_integer_torch(
                low=0, high=valid_obstacles.shape[1], shape=(len(invalid_indices),), ids=env_ids
            )
        else:
            replacement_indices = self._rng.sample_unique_integers_torch(
                min=self._task_cfg.max_num_vis_obstacles,
                max=valid_obstacles.shape[1],
                num=len(invalid_indices),
                ids=env_ids,
            )
        replacements = valid_obstacles[env_ids, replacement_indices]
        xyz[env_ids, invalid_indices] = replacements

        # Generate quats and concatenate with xyz
        xyzw = self.obstacles.data.object_com_quat_w[env_ids].clone()
        obstacles_positions = torch.cat((xyz[:, : self._task_cfg.max_num_vis_obstacles], xyzw), dim=-1)

        # Create visible obstacles
        num_visible_obstacles_per_env = self._rng.sample_integer_torch(
            low=1, high=self._task_cfg.max_num_vis_obstacles, shape=(1,), ids=env_ids
        )
        mask = torch.arange(self._task_cfg.max_num_vis_obstacles, device=self._device).unsqueeze(
            0
        ) < num_visible_obstacles_per_env.unsqueeze(1)

        return obstacles_positions, mask
