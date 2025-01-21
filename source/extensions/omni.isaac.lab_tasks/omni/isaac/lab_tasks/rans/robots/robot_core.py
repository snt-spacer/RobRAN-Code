# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.assets import Articulation

from omni.isaac.lab_tasks.rans.utils import ScalarLogger


class RobotCore:
    def __init__(
        self,
        robot_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        """Initializes the robot core.

        Args:
            robot_cfg: The configuration of the robot.
            robot_uid: The unique id of the robot.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        # Unique task identifier, used to differentiate between tasks with the same name
        self._robot_uid = robot_uid
        # Number of environments and device to be used
        self._num_envs = num_envs
        self._device = device

        # Defines the observation and actions space sizes for this task
        self._dim_robot_obs: int = MISSING
        self._dim_robot_act: int = MISSING
        self._dim_gen_act: int = MISSING

        # Robot
        self._robot: Articulation = MISSING

        # Logs
        self.create_logs()

    @property
    def num_observations(self) -> int:
        """Returns the number of observations for the robot.
        Typically, this would be linked to the joints of the robot or its actions.
        It's what's unique to that robot."""

        return self._dim_robot_obs

    @property
    def num_actions(self) -> int:
        """Returns the number of actions for the robot. This is the actions that the robot can take.
        Not the randomization of the environment."""

        return self._dim_robot_act

    @property
    def num_gen_actions(self) -> int:
        """Returns the number of actions for the robot. This is the actions that the robot can take.
        Not the randomization of the environment."""

        return self._dim_gen_act

    @property
    def logs(self) -> dict:
        """Returns the logs of the robot."""
        return self.scalar_logger.get_episode_logs

    def create_logs(self):
        """Creates the logs for the robot."""
        self.scalar_logger = ScalarLogger(self._num_envs, self._device, "robot")

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        # Buffers
        if env_ids is None:
            self._env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        else:
            self._env_ids = env_ids
        self._seeds = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        self._gen_actions = torch.zeros(
            (self._num_envs, self._dim_gen_act),
            device=self._device,
            dtype=torch.float32,
        )
        self._actions = torch.zeros(
            (self._num_envs, self._dim_robot_act),
            device=self._device,
            dtype=torch.float32,
        )

    def run_setup(self, robot: Articulation) -> None:
        """Loads the robot into the task. After it has been loaded."""
        self._robot = robot

    def get_observations(self):
        """Returns the observations of the robot."""
        raise NotImplementedError

    def compute_rewards(self):
        """Computes the rewards of the robot."""
        raise NotImplementedError

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the dones of the robot."""
        raise NotImplementedError

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ):
        self._actions[env_ids] = 0

        # Updates the task actions
        if gen_actions is None:
            self._gen_actions[env_ids] = torch.rand((len(env_ids), self.num_gen_actions), device=self._device)
        else:
            self._gen_actions[env_ids] = gen_actions

        # Updates the seed
        if env_seeds is None:
            self._seeds[env_ids] = torch.randint(0, 100000, (len(env_ids),), dtype=torch.int32, device=self._device)
        else:
            self._seeds[env_ids] = env_seeds

        self.set_initial_conditions(env_ids)

    def reset_logs(self, env_ids, episode_length_buf) -> None:
        self.scalar_logger.reset(env_ids, episode_length_buf)

    def compute_logs(self) -> dict:
        return self.scalar_logger.compute_extras()

    def set_pose(
        self,
        pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._robot.write_root_link_pose_to_sim(pose, env_ids)

    def set_velocity(
        self,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._robot.write_root_com_velocity_to_sim(velocity, env_ids)

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None) -> None:
        raise NotImplementedError

    def process_actions(self) -> None:
        raise NotImplementedError

    def compute_physics(self) -> None:
        raise NotImplementedError

    def apply_actions(self) -> None:
        raise NotImplementedError

    def updateMass(self) -> None:
        raise NotImplementedError

    def updateInertia(self) -> None:
        raise NotImplementedError

    def updateCoM(self) -> None:
        raise NotImplementedError

    def updateFriction(self) -> None:
        raise NotImplementedError

    # We wrap around the ArticulationData properties to make them modifiable from the
    # class that inherits from RobotCore. This is done so that we can have a unique interface
    # for all the robots that we create.

    # Typical problem:
    # 2D Floating platform:
    # - It uses a set of joints to constrain it into a 2D plane. This means that the position of the root rigid body
    #  is fixed. Hence querying the root position will always return the same value.
    # - Solution, override the root_state_w property to return the position of a different rigid body.
    # - Ideally, this should be avoided by defining the root rigid body to be the one that moves in the 2D plane. But
    #  it's unclear if this is possible.

    # This is not the nicest hack, but performance should be OK.

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation root's center of mass frame.
        """
        return self._robot.data.root_state_w

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links's actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        return self._robot.data.body_state_w

    @property
    def body_acc_w(self):
        """Acceleration of all bodies. Shape is (num_instances, num_bodies, 6).

        This quantity is the acceleration of the articulation links' center of mass frame.
        """
        return self._robot.data.body_acc_w

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        return self._robot.data.projected_gravity_b

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        return self._robot.data.heading_w

    @property
    def joint_pos(self):
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        return self._robot.data.joint_pos

    @property
    def joint_vel(self):
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        return self._robot.data.joint_vel

    @property
    def joint_acc(self):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        return self._robot.data.joint_acc

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the articulation root.
        """
        return self._robot.data.root_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the articulation root.
        """
        return self._robot.data.root_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of
        mass frame.
        """
        return self._robot.data.root_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.root_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Root angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame.
        """
        return self._robot.data.root_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Root linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with
        respect to the articulation root's actor frame.
        """
        return self._robot.data.root_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Root angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        articulation root's actor frame.
        """
        return self._robot.data.root_ang_vel_b

    #
    # Derived Root Link Frame Properties
    #

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.root_link_pos_w

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        return self._robot.data.root_link_quat_w

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        return self._robot.data.root_link_vel_w

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        return self._robot.data.root_link_lin_vel_w

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.root_link_ang_vel_w

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return self._robot.data.root_link_lin_vel_b

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        return self._robot.data.root_link_ang_vel_b

    ##
    # Derived CoM frame properties
    ##

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.root_com_pos_w

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        return self._robot.data.root_com_quat_w

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.root_com_vel_w

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        return self._robot.data.root_com_lin_vel_w

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """

        return self._robot.data.root_com_ang_vel_w

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._robot.data.root_com_lin_vel_b

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        return self._robot.data.root_com_ang_vel_b

    ##
    # Derived Bodies Frame Properties
    ##

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        return self._robot.data.body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the rigid bodies' actor frame.
        """
        return self._robot.data.body_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the rigid bodies' center of mass frame.
        """
        return self._robot.data.body_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        return self._robot.data.body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        return self._robot.data.body_ang_vel_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        return self._robot.data.body_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        return self._robot.data.body_ang_acc_w
