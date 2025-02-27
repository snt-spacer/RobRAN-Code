# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_rotate_inverse


@configclass
class HydrodynamicsCfg:
    """Configuration for hydrodynamics."""

    # Damping
    # [u (Forward), v (Lateral), w (Vertical), p (Roll), q (Pitch), r (Yaw)]
    linear_damping: list = MISSING  # [u, v, w, p, q, r]
    quadratic_damping: list = MISSING  # [u, v, w, p, q, r]
    use_drag_randomization: bool = False
    linear_damping_rand: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    quadratic_damping_rand: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    linear_damping_forward_speed: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    offset_linear_damping: float = 0.0
    offset_lin_forward_damping_speed: float = 0.0
    offset_nonlin_damping: float = 0.0
    scaling_damping: float = 1.0
    offset_added_mass: float = 0.0


class Hydrodynamics:
    def __init__(self, num_envs, device, cfg: HydrodynamicsCfg):

        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        # Initialize damping tensors
        self.base_linear_damping = torch.tensor([cfg.linear_damping] * self.num_envs, device=self.device)
        self.base_quadratic_damping = torch.tensor([cfg.quadratic_damping] * self.num_envs, device=self.device)
        self.linear_damping = torch.tensor([cfg.linear_damping] * self.num_envs, device=self.device)
        self.quadratic_damping = torch.tensor([cfg.quadratic_damping] * self.num_envs, device=self.device)
        self.linear_damping_forward_speed = torch.tensor(cfg.linear_damping_forward_speed, device=self.device)

        # Initialize tensor for drag forces and torques
        self.drag = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

        self.linear_rand = torch.tensor(cfg.linear_damping_rand, device=self.device)
        self.quadratic_rand = torch.tensor(cfg.quadratic_damping_rand, device=self.device)

        # TODO: implement coriolis and added mass
        self._Ca = torch.zeros([6, 6], device=self.device)
        self.added_mass = torch.zeros([num_envs, 6], device=self.device)

        return

    def reset_coefficients(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Resets the drag coefficients for the specified environments.
        Args:
            env_ids (torch.Tensor): Indices of the environments to reset. If None, resets all environments.
        """
        if not self.cfg.use_drag_randomization:
            return

        # Handle case for all environments
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Generate scaled random noise
        noise_linear = (torch.rand((len(env_ids), 6), device=self.device) * 2 - 1) * self.linear_rand
        noise_quad = (torch.rand((len(env_ids), 6), device=self.device) * 2 - 1) * self.quadratic_rand

        # Multiply the random noise by the specified damping coefficients
        self.linear_damping[env_ids] = self.base_linear_damping[env_ids] * (1 + noise_linear)
        self.quadratic_damping[env_ids] = self.base_quadratic_damping[env_ids] * (1 + noise_quad)
        return

    def ComputeDampingMatrix(self, vel):
        """
        // From Antonelli 2014: the viscosity of the fluid causes
        // the presence of dissipative drag and lift forces on the
        // body. A common simplification is to consider only linear
        // and quadratic damping terms and group these terms in a
        // matrix Drb
        """
        # print("vel: ", vel)
        lin_damp = (
            self.linear_damping
            + self.cfg.offset_linear_damping
            - (self.linear_damping_forward_speed + self.cfg.offset_lin_forward_damping_speed)
        )
        # print("lin_damp: ", lin_damp)
        quad_damp = ((self.quadratic_damping + self.cfg.offset_nonlin_damping).mT * torch.abs(vel.mT)).mT
        # print("quad_damp: ", quad_damp)
        # scaling and adding both matrices
        damping_matrix = (lin_damp + quad_damp) * self.cfg.scaling_damping
        # print("damping_matrix: ", damping_matrix)
        return damping_matrix

    def ComputeHydrodynamicsEffects(self, quaternions, world_vel):

        self.local_lin_velocities = quat_rotate_inverse(quaternions, world_vel[:, :3])
        self.local_ang_velocities = quat_rotate_inverse(quaternions, world_vel[:, 3:])
        self.local_velocities = torch.hstack([self.local_lin_velocities, self.local_ang_velocities])

        # Update damping matrix
        damping_matrix = self.ComputeDampingMatrix(self.local_velocities)

        # Damping forces and torques
        self.drag = -1 * damping_matrix * self.local_velocities

        return self.drag
