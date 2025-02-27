# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from gymnasium import spaces, vector

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils import math as math_utils

from omni.isaac.lab_tasks.rans import FloatingPlatformRobotCfg

from .robot_core import RobotCore


class FloatingPlatformRobot(RobotCore):

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: FloatingPlatformRobotCfg = FloatingPlatformRobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        decimation: int = 6,
        device: str = "cuda",
    ):
        super().__init__(scene=scene, robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg
        # Available for use robot_cfg.is_reaction_wheel,robot_cfg.split_thrust,robot_cfg.rew_reaction_wheel_scale
        self._dim_robot_obs = self._robot_cfg.observation_space
        self._dim_robot_act = self._robot_cfg.action_space
        self._dim_gen_act = self._robot_cfg.gen_space

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._actions = torch.zeros((self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32)
        self._previous_actions = torch.zeros(
            (self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32
        )
        self._thrust_action = torch.zeros(
            (self._num_envs, self._robot_cfg.num_thrusters), device=self._device, dtype=torch.float32
        )
        if self._robot_cfg.has_reaction_wheel:
            self._reaction_wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._thrusters_dof_idx, _ = self._robot.find_bodies(self._robot_cfg.thrusters_dof_name)
        self._root_idx, _ = self._robot.find_bodies([self._robot_cfg.root_id_name])

        if self._robot_cfg.has_reaction_wheel:
            self._reaction_wheel_dof_idx, _ = self._robot.find_joints(self._robot_cfg.reaction_wheel_dof_name)

    def create_logs(self):
        super().create_logs()

        self.scalar_logger.add_log("robot_state", "AVG/thrusters", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/reaction_wheel", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_state", "AVG/joint_acceleration", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/action_rate", "mean")
        self.scalar_logger.add_log("robot_reward", "AVG/joint_acceleration", "mean")

    def get_observations(self) -> torch.Tensor:
        return self._unaltered_actions

    def compute_rewards(self):
        # TODO: DT should be factored in?

        action_rate = torch.sum(torch.abs(self._unaltered_actions - self._previous_unaltered_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self.scalar_logger.log("robot_state", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/action_rate", action_rate)
        self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations)

        return (
            action_rate * self._robot_cfg.rew_action_rate_scale
            + joint_accelerations * self._robot_cfg.rew_joint_accel_scale
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
    ):
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor):
        thrust_reset = torch.zeros_like(self._thrust_action)
        self._robot.set_external_force_and_torque(
            thrust_reset, thrust_reset, body_ids=self._thrusters_dof_idx, env_ids=env_ids
        )
        locking_joints = torch.zeros((len(env_ids), 3), device=self._device)
        self._robot.set_joint_velocity_target(locking_joints, env_ids=env_ids)
        self._robot.set_joint_position_target(locking_joints, env_ids=env_ids)

        if self._robot_cfg.has_reaction_wheel:
            rw_reset = torch.zeros_like(self._reaction_wheel_action)
            self._robot.set_joint_velocity_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
            self._robot.set_joint_effort_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor):
        """Process the actions for the robot.

        Expects either binary actions: 0 or 1, or continuous actions: [0, 1].

        - First, clip the actions to the action space limits. This is done to avoid violating the robot's limits.
        - Second, apply the action randomizers to the actions. This is done to add noise to the actions, apply
          different scaling factors to the actions, etc.
        - Third, format the actions to send to the actuators.

        Args:
            actions (torch.Tensor): The actions to process."""

        # Enforce action limits at the robot level
        actions = actions.float()
        actions.clip_(min=0.0, max=1.0)
        # Store the unaltered actions, by default the robot should only observe the unaltered actions.
        self._previous_unaltered_actions = self._unaltered_actions.clone()
        self._unaltered_actions = actions.clone()

        # Apply action randomizers
        for randomizer in self.randomizers:
            randomizer.actions(dt=self.scene.physics_dt, actions=actions)

        self._previous_actions = self._actions.clone()
        self._actions = actions

        # Calculate the number of active thrusters (those with a value of 1)
        n_active_thrusters = torch.sum(actions[:, : self._robot_cfg.num_thrusters], dim=1, keepdim=True)
        # Determine thrust scaling factor
        if self._robot_cfg.split_thrust:
            # Calculate thrust scale as max thrust divided by the number of active thrusters
            thrust_scale = torch.where(
                n_active_thrusters > 0,
                self._robot_cfg.max_thrust / n_active_thrusters,
                torch.tensor(0.0, device=actions.device),
            )
        else:
            thrust_scale = self._robot_cfg.max_thrust

        # Apply thrust to thrusters, based on whether reaction wheel is present
        self._thrust_action = actions[:, : self._robot_cfg.num_thrusters].float() * thrust_scale
        # transform the 2D thrust actions into 3D forces and torques with x and y components set to zero and z components based on the thrust actions
        self._thrust_action = self._thrust_action.unsqueeze(2).expand(-1, -1, 3)
        self._thrust_action = torch.cat(
            (torch.zeros_like(self._thrust_action[:, :, :2]), self._thrust_action[:, :, 2:]), dim=2
        )

        if self._robot_cfg.has_reaction_wheel:
            # Separate continuous control for reaction wheel
            self._reaction_wheel_action = (
                actions[:, self._robot_cfg.num_thrusters :] * self._robot_cfg.reaction_wheel_scale
            )
            self._reaction_wheel_action = self._reaction_wheel_action.unsqueeze(2).expand(-1, -1, 3)

        # Log data for monitoring
        self.scalar_logger.log("robot_state", "AVG/thrusters", torch.linalg.norm(self._thrust_action[:, :, 2], dim=-1))
        if self._robot_cfg.has_reaction_wheel:
            self.scalar_logger.log("robot_state", "AVG/reaction_wheel", self._reaction_wheel_action[:, 0])

    def compute_physics(self):
        pass  # Model motor + ackermann steering here

    def apply_actions(self):
        # Compute the physics
        super().apply_actions()
        for randomizer in self.randomizers:
            randomizer.update(dt=self.scene.physics_dt, actions=self._actions)

        self._robot.set_external_force_and_torque(
            self._thrust_action, torch.zeros_like(self._thrust_action), body_ids=self._thrusters_dof_idx
        )
        if self._robot_cfg.has_reaction_wheel:
            self._robot.set_joint_effort_target(self._reaction_wheel_action, joint_ids=self._reaction_wheel_dof_idx)

    def set_velocity(
        self,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        velocity = torch.cat([velocity[:, :2], velocity[:, -1].unsqueeze(-1)], dim=1)
        position = torch.zeros_like(velocity)
        self._robot.write_joint_state_to_sim(position, velocity, env_ids=env_ids)

    def configure_gym_env_spaces(self):
        single_action_space = spaces.MultiDiscrete([2] * self._robot_cfg.num_thrusters)
        action_space = vector.utils.batch_space(single_action_space, self._num_envs)

        return single_action_space, action_space

    ##
    # Derived base properties
    ##

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self._robot.data.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    ##
    # Derived root properties
    ##

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        return self._robot.data.body_state_w[:, self._root_idx]

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.
        """
        return self._robot.data.body_pos_w[:, self._root_idx].squeeze()

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.
        """
        return self._robot.data.body_quat_w[:, self._root_idx].squeeze()

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.
        """
        return self._robot.data.body_vel_w[:, self._root_idx].squeeze()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.body_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.body_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with
        respect to the articulation root's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        articulation root's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)

    ##
    # Derived Root Link Frame properties
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_link_pos_w[:, self._root_idx].squeeze()

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._robot.data.body_link_quat_w[:, self._root_idx].squeeze()

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.body_link_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.body_link_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_link_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    ##
    # Derived CoM frame properties
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_com_pos_w[:, self._root_idx].squeeze()

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.body_com_quat_w[:, self._root_idx].squeeze()

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_lin_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.body_com_ang_vel_w[:, self._root_idx].squeeze()

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_com_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return math_utils.quat_rotate_inverse(self.root_com_quat_w, self.root_com_ang_vel_w)
