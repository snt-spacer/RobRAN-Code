# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math
import warp as wp

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab_tasks.rans import ModularFreeflyerRobotCfg

from .robot_core import RobotCore


class ModularFreeflyerRobot(RobotCore):
    def __init__(
        self,
        robot_cfg: ModularFreeflyerRobotCfg,
        robot_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
    ) -> None:
        super().__init__(robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = 8
        self._dim_robot_act = 8
        self._dim_gen_act = 0

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
        self._reaction_wheel_actions = torch.zeros(
            (self._num_envs, 1), device=self._device, dtype=torch.float32
        )
        self._transforms = torch.zeros(
            (self._num_envs, 3, 4), device=self._device, dtype=torch.float32
        )

    def run_setup(self, robot: Articulation) -> None:
        super().run_setup(robot)
        self._x_lock_dof_idx, _ = self._robot.find_joints(self._robot_cfg.x_lock_name)
        self._y_lock_dof_idx, _ = self._robot.find_joints(self._robot_cfg.y_lock_name)
        self._z_lock_dof_idx, _ = self._robot.find_joints(self._robot_cfg.z_lock_name)
        self._lock_ids = [
            self._x_lock_dof_idx,
            self._y_lock_dof_idx,
            self._z_lock_dof_idx,
        ]

    def create_logs(self) -> None:
        super().create_logs()
        torch_zeros = lambda: torch.zeros(
            self._num_envs,
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )
        self._logs["state"]["throttle"] = torch_zeros()
        self._logs["state"]["steering"] = torch_zeros()
        self._logs["state"]["action_rate"] = torch_zeros()
        self._logs["state"]["joint_acceleration"] = torch_zeros()
        self._logs["reward"]["action_rate"] = torch_zeros()
        self._logs["reward"]["joint_acceleration"] = torch_zeros()

    def get_observations(self) -> torch.Tensor:
        return self._actions

    def compute_rewards(self) -> torch.Tensor:
        # TODO: DT should be factored in?

        # Compute
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self._logs["state"]["action_rate"] = action_rate
        self._logs["state"]["joint_acceleration"] = joint_accelerations
        self._logs["reward"]["action_rate"] = action_rate
        self._logs["reward"]["joint_acceleration"] = joint_accelerations
        return (
            action_rate * self._robot_cfg.rew_action_rate_scale
            + joint_accelerations * self._robot_cfg.rew_joint_accel_scale
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = torch.zeros(
            self._num_envs, dtype=torch.int32, device=self._device
        )
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

    def reset_logs(self, env_ids: torch.Tensor) -> None:
        # Reset logs
        self._logs["state"]["throttle"][env_ids] = 0
        self._logs["state"]["steering"][env_ids] = 0
        self._logs["state"]["action_rate"][env_ids] = 0
        self._logs["state"]["joint_acceleration"][env_ids] = 0
        self._logs["reward"]["action_rate"][env_ids] = 0
        self._logs["reward"]["joint_acceleration"][env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor | None = None) -> None:
        # Create zero tensor
        zeros = torch.zeros(
            (len(env_ids), len(self._lock_ids)),
            device=self._device,
            dtype=torch.float32,
        )
        # Sets the joints to zero
        self._robot.set_joint_position_target(
            zeros, joint_ids=self._lock_ids, env_ids=env_ids
        )
        self._robot.set_joint_velocity_target(
            zeros, joint_ids=self._lock_ids, env_ids=env_ids
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        # Clone the previous actions
        self._previous_actions = self._actions.clone()
        # Clone the current actions
        self._actions = actions.clone()

        # Assumes the action space is [-1, 1]
        if self._robot_cfg.action_mode == "continuous":
            self._thrust_actions = self._actions[:, : self._robot_cfg.num_thrusters]
            self._thrust_actions = (self._thrust_actions + 1) / 2.0
            self._reaction_wheel_actions = self._actions[:, -1]
        else:
            self._thrust_actions = (
                self._actions[:, : self._robot_cfg.num_thrusters] > 0.0
            ).float()
            self._reaction_wheel_actions = self._actions[:, -1]

        # Log data
        self._logs["state"]["thrust"] = self._throttle_action[:, 0]
        self._logs["state"]["reaction_wheel"] = self._steering_action[:, 0]

    def compute_physics(self) -> None:
        self.
        pass 

    def apply_actions(self) -> None:
        pass

class ThrustGenerator:
    def __init__(self, robot_cfg: ModularFreeflyerRobotCfg, num_envs: int, device: str):

        self._num_envs = num_envs
        self._device = device
        self._robot_cfg = robot_cfg
        if self._robot_cfg.random_thrusters:
            pass
        elif self._robot_cfg.thruster_transforms:
            pass
        else:
            pass

    #def initialize_buffer(self):
    #    transforms2D = wp.zeros((self._num_envs, self._robot_cfg.num_thrusters), dtype=wp.mat33f, device=self._device)
    #    transforms = wp.zeros((self._num_envs, self._robot_cfg.num_thrusters), dtype=wp.vec5f, device=self._device)

    def initialize_buffers(self):
        self._transforms2D = torch.zeros((self._num_envs, self._robot_cfg.num_thrusters, 3,3), dtype=torch.float, device=self._device)
        self._transforms = torch.zeros((self._num_envs, self._robot_cfg.num_thrusters, 5), dtype=torch.float, device=self._device)

    
    def get_transforms_from_cfg(self):
        transforms = torch.zeros((1, self._robot_cfg.num_thrusters, 5), device=self._device, dtype=torch.float32)
        transforms2D = torch.zeros((1, self._robot_cfg.num_thrusters, 3,3), device=self._device, dtype=torch.float32)
        assert len(self._robot_cfg.thruster_transforms) == self._robot_cfg.num_thrusters, ".... TODO"

        # Transforms are stored in [x,y,theta,F] format, they need to be converted to 2D transforms, and a compact representation
        for i, trsfrm in enumerate(transforms):
            # 2D transforms used to project the forces
            self._transforms2D[:, i, 0, 0] = math.cos(trsfrm[2])
            self._transforms2D[:, i, 0, 1] = math.sin(-trsfrm[2])
            self._transforms2D[:, i, 1, 0] = math.sin(trsfrm[2])
            self._transforms2D[:, i, 1, 1] = math.cos(trsfrm[2])
            self._transforms2D[:, i, 2, 0] = trsfrm[0]
            self._transforms2D[:, i, 2, 1] = trsfrm[1]
            self._transforms2D[:, i, 2, 2] = trsfrm[3]
            # Compact transform representation to inform the network
            self._transforms[:, i, 0] = math.cos(trsfrm[2])
            self._transforms[:, i, 1] = math.sin(trsfrm[2])
            self._transforms[:, i, 2] = trsfrm[0]
            self._transforms[:, i, 3] = trsfrm[1]
            self._transforms[:, i, 4] = trsfrm[3]
        
        transforms.repeat_interleave((self._num_envs), dim=0)

    def build_default_transforms(self):
        transforms2D 