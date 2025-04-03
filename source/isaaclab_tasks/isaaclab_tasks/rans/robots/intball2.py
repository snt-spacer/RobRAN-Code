# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from gymnasium import spaces, vector

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import IntBall2RobotCfg

from .robot_core import RobotCore

# from isaaclab.sensors import ContactSensor


class IntBall2Robot(RobotCore):

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: IntBall2RobotCfg = IntBall2RobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        decimation: int = 4,
        device: str = "cuda",
    ) -> None:
        super().__init__(scene=scene, robot_uid=robot_uid, num_envs=num_envs, decimation=decimation, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = self._robot_cfg.observation_space
        self._dim_robot_act = self._robot_cfg.action_space
        self._dim_gen_act = self._robot_cfg.gen_space

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids=None) -> None:
        super().initialize_buffers(env_ids)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act),
            device=self._device,
            dtype=torch.float32,
        )
        self._thruster_action = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters),
            device=self._device,
            dtype=torch.float32,
        )
        self._transforms = torch.zeros(
            (self._num_envs, 3, 4),
            device=self._device,
            dtype=torch.float32,
        )
        self._thrust_forces = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )
        self._thrust_torques = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )
        self._thrust_positions = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )

    def run_setup(self, robot: Articulation) -> None:
        """Loads the robot into the task. After it has been loaded."""
        super().run_setup(robot)

        # Sets the articulation to be our overloaded articulation with improved force application
        self._robot = robot

        self._thruster_ids, _ = self._robot.find_bodies("propeller_.*")
        # Get the index of the root body (used to get the state of the robot)
        self._root_idx = self._robot.find_bodies(self._robot_cfg.root_body_name)[0]
        # Get the thrust generator
        self._thrust_generator = ThrustGenerator(self._robot_cfg, self._num_envs, self._device)

    def create_logs(self) -> None:
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/thrust", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/torque", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/torque", "mean")

    def get_observations(self) -> torch.Tensor:
        """Returns the observation vector (thruster state + velocities)."""
        # return torch.cat([self._thrust_actions, self.root_lin_vel_w, self.root_ang_vel_w], dim=-1)
        # return self._thrust_actions
        return self._unaltered_actions

    def compute_rewards(self) -> torch.Tensor:
        # Compute
        action_rate = torch.sum(torch.square(self._unaltered_actions - self._previous_unaltered_actions), dim=1)
        observed_torque = torch.sum(torch.abs(self._thrust_torques), dim=(1, 2))

        # Log data
        self.scalar_logger.log("robot_state", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_state", "AVG/torque", observed_torque)

        self.scalar_logger.log("robot_reward", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_reward", "AVG/torque", observed_torque)

        return (
            action_rate * self._robot_cfg.rew_action_rate_scale
            + observed_torque * self._robot_cfg.rew_torque_balance_scale
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        task_done = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        return task_failed, task_done

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0
        self._thrust_forces[env_ids] = 0
        self._thrust_positions[env_ids] = 0
        self._thrust_torques[env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None) -> None:
        # IntBall2 has no controllable joints yet
        # Reset root pose (position + quaternion rotation)
        root_pose = torch.zeros((len(env_ids), 7), device=self._device, dtype=torch.float32)
        self._robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        """
        Process the actions for the robot.

        Expects either binary actions 0/1, or continuous actions in the range [-1, 1].

        - First, clip the actions to the action space limits. This is done to avoid voilating the robot's limits.
        - Second, apply the action randomizers to the actions. This is done to add noise to the actions, apply different scaling factors to the actions, etc.
        - Third, format the actions to senf to the actuators.

        Args:
            actions (torch.Tensor): The actions to process.
        """

        # Enforce action limits at the robot level
        actions.clip_(min=0.0, max=1.0)
        # Store the unaltered actions, by default the robot should only observe the unaltered actions.
        self._previous_unaltered_actions = self._unaltered_actions.clone()
        self._unaltered_actions = actions.clone()

        # Apply the action randomizers
        for randomizer in self.randomizers:
            randomizer.actions(dt=self.scene.physics_dt, actions=actions)

        # Clone previous actions
        self._previous_actions = self._actions.clone()
        # Clone current actions
        # self._actions = actions.clone()
        self._actions = actions

        # Normalize the actions to [0, 1] range if in continuous mode
        if self._robot_cfg.action_mode == "continuous":
            self._thrust_actions = (self._actions + 1) / 2.0  # mapping from [-1, 1] to [0, 1]
        else:
            self._thrust_actions = (self._actions > 0).float()  # binary action

        # Compute the scaled torque using the defined thrust scale factors
        self._thrust_actions *= torch.tensor(
            self._robot_cfg.thrust_scale_factors, device=self._device, dtype=torch.float32
        )
        self.scalar_logger.log("robot_state", "AVG/thrust", torch.sum(self._thrust_actions, dim=1))

    def compute_physics(self) -> None:
        self._thrust_positions, self._thrust_forces, self._thrust_torques = (
            self._thrust_generator.cast_actions_to_thrust(self._thrust_actions)
        )
        # print(f"Thrust actions: \n {self._thrust_actions[0]}")

    def apply_actions(self) -> None:
        # self.compute_physics()
        super().apply_actions()
        for randomizer in self.randomizers:
            randomizer.update(dt=self.scene.physics_dt, actions=self._actions)

        self._robot.set_external_force_and_torque(
            self._thrust_forces, self._thrust_torques, positions=self._thrust_positions, body_ids=self._thruster_ids
        )

    def set_pose(
        self,
        pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._robot.write_root_pose_to_sim(pose, env_ids)

    def set_velocity(
        self,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._robot.write_root_velocity_to_sim(velocity, env_ids)

    def configure_gym_env_spaces(self):
        single_action_space = spaces.MultiDiscrete([2] * self._robot_cfg.num_thrusters)
        action_space = vector.utils.batch_space(single_action_space, self._num_envs)

        return single_action_space, action_space

    # def activateSensors(self, sensor_type: str, filter: list):
    #     if sensor_type == "contacts":
    #         self._robot_cfg.contact_sensor_active=True
    #         if len(filter) > 0:
    #             self._robot_cfg.body_contact_forces.filter_prim_paths_expr = filter

    # def register_sensors(self, scene: InteractiveScene) -> None:
    #     # Contact sensor
    #     if self._robot_cfg.contact_sensor_active:
    #         scene.sensors["robot_contacts"] = ContactSensor(
    #             self._robot_cfg.body_contact_forces
    #         )
    #         self.contacts: ContactSensor = scene["robot_contacts"]

    ##
    # Derived base properties
    ##

    # Return the full roll-pitch-yaw orientation of the robot in world frame

    @property
    def euler_angles_w(self):
        """
        Returns the roll, pitch, and yaw angles (XYZ Euler angles) in the world frame.

        Shape: (num_instances, 3), where:
            - [:, 0]: Roll
            - [:, 1]: Pitch
            - [:, 2]: Yaw
        """
        # NOTE: Check for compatibility with RANS

        # Convert the quaternion to Euler angles
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(self.root_quat_w)

        return torch.stack([roll, pitch, yaw], dim=-1)


class ThrustGenerator:
    def __init__(self, robot_cfg: IntBall2RobotCfg, num_envs: int, device: str):

        self._num_envs = num_envs
        self._device = device
        self._robot_cfg = robot_cfg

        self.initialize_buffers()
        self.get_transforms_from_cfg()

    def initialize_buffers(self):
        self._transforms3D = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 4, 4),
            dtype=torch.float,
            device=self._device,
        )
        self._transforms = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 7),
            dtype=torch.float,
            device=self._device,
        )
        self._thrust_force = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters), device=self._device, dtype=torch.float32
        )
        # self._drag_torque_factors = torch.zeros(
        #     (self._num_envs, self._robot_cfg.num_thrusters),
        #     device=self._device,
        #     dtype=torch.float32,
        # )
        self.unit_vector = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters, 3),
            device=self._device,
            dtype=torch.float32,
        )
        # self.unit_vector[:, :, 2] = 1.0  # Default thrust direction is Z

    def get_transforms_from_cfg(self):
        """Extracts thruster positions and orientations from the config."""

        assert (
            len(self._robot_cfg.thruster_transforms) == self._robot_cfg.num_thrusters
        ), "Number of thruster transforms must match the number of thrusters."

        # Convert each thruster's defined position and rotation into transformation matrices
        for i, trsfrm in enumerate(self._robot_cfg.thruster_transforms):
            x, y, z, rot_x, rot_y, rot_z = trsfrm  # extract components

            R = math_utils.matrix_from_euler(
                torch.tensor([rot_x, rot_y, rot_z], device=self._device).unsqueeze(0), convention="XYZ"
            )

            # define the 3D transformation matrix
            self._transforms3D[:, i, :3, :3] = R
            self._transforms3D[:, i, :3, 3] = torch.tensor([x, y, z], device=self._device)
            self._transforms3D[:, i, 3, 3] = 1.0

            # thrust direction (outward normal)
            normal = R[:, 2]  # Z-axis of thruster frame
            self.unit_vector[:, i, :] = normal

            # Assign thrust scaled defined in config
            self._thrust_force[:, i] = self._robot_cfg.thruster_max_thrust[i]

    def cast_actions_to_thrust(self, actions):
        """
        Projects the thrusts for IntBall2 into world coordinates.
        Converts thrust commands into real-world forces and torques.
        """

        # Scale thrust actions by their maximum thrust values
        rand_forces = actions * self._thrust_force
        # Split transforms into translation and rotation
        R = self._transforms3D[:, :, :3, :3].reshape(-1, 3, 3)  # Extract 3x3 rotation matrices
        T = self._transforms3D[:, :, :3, 3].reshape(-1, 3)  # Extract translation (position) vectors
        # Compute thrust force directions using unit vectors (Z-axis of thruster frame)
        force_vector = -self.unit_vector * rand_forces.view(-1, self._robot_cfg.num_thrusters, 1)
        # Rotate the forces from local frame (thruster) to world frame
        rotated_forces = torch.matmul(R, force_vector.view(-1, 3, 1)).squeeze(-1)  # Shape: (num_envs, num_thrusters, 3)

        # Compute torques
        torques = torch.cross(T, rotated_forces, dim=-1)  # Shape: (num_envs, num_thrusters, 3)

        # Return forces and torques applied at thruster positions
        return (
            T.reshape(-1, self._robot_cfg.num_thrusters, 3),
            rotated_forces.reshape(-1, self._robot_cfg.num_thrusters, 3),
            torques.reshape(-1, self._robot_cfg.num_thrusters, 3),
        )

    @property
    def compact_transforms(self):
        """Returns the compact representation of thruster transforms."""
        return self._transforms  # Shape: (num_envs, num_thrusters, 6)

    @property
    def transforms3D(self):
        """Returns the full 3D transformation matrices of thrusters."""
        return self._transforms3D  # Shape: (num_envs, num_thrusters, 4, 4)
