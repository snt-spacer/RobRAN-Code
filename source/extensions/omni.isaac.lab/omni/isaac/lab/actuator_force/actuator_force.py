# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass


@configclass
class PropellerActuatorCfg:

    cmd_lower_range: float = MISSING
    cmd_upper_range: float = MISSING
    command_rate: float = MISSING  # Frequency of command updates in Hz
    # TODO: Actuators should be handled independently for each thruster
    forces_left: list = MISSING
    forces_right: list = MISSING
    interp_resolution: int = 1000


class PropellerActuator:
    def __init__(self, num_envs, device, dt, cfg: PropellerActuatorCfg = PropellerActuatorCfg()):
        """
        Initializes the PropellerActuator class with the given parameters.

        Args:
            num_envs (int): Number of environments.
            device (torch.device): The device (CPU or GPU) on which tensors will be allocated.
            dt (float): Time step duration.
            cfg (PropellerActuatorCfg, optional): Configuration object for the propeller actuator. Defaults to PropellerActuatorCfg().

        Attributes:
            num_envs (int): Number of environments.
            device (torch.device): The device (CPU or GPU) on which tensors are allocated.
            dt (float): Time step duration.
            cfg (PropellerActuatorCfg): Configuration object for the propeller actuator.
            _max_cmd_delta (float): Maximum command delta calculated from command rate and time step.
            _interp_scale (float): Interpolation scale factor.
            _current_cmds (torch.Tensor): Tensor to store current commands for each environment.
            _target_cmds (torch.Tensor): Tensor to store target commands for each environment.
            forces_left (torch.Tensor): Tensor containing the left forces from the configuration.
            forces_right (torch.Tensor): Tensor containing the right forces from the configuration.
            thruster_forces (torch.Tensor): Tensor to store thruster forces for each environment.
            interp_forces_left (torch.Tensor): Tensor to store interpolated left forces.
            interp_forces_right (torch.Tensor): Tensor to store interpolated right forces.
        """

        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.cfg = cfg

        # Constants
        self._max_cmd_delta = cfg.command_rate * dt
        self._interp_scale = (cfg.interp_resolution - 1) / 2

        # Initialize tensors
        self._current_cmds = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._target_cmds = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        # TODO: Actuators should be handled independently for each thruster
        self.forces_left = torch.tensor(self.cfg.forces_left, dtype=torch.float32, device=self.device)
        self.forces_right = torch.tensor(self.cfg.forces_right, dtype=torch.float32, device=self.device)
        self.thruster_forces = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

        self.interp_forces_left = torch.zeros(self.cfg.interp_resolution, dtype=torch.float32, device=self.device)
        self.interp_forces_right = torch.zeros(self.cfg.interp_resolution, dtype=torch.float32, device=self.device)

        # Expand the forces vectors with linear interpolation
        self.interp_forces_left = self.linear_interpolate_1d(self.forces_left, self.cfg.interp_resolution)
        self.interp_forces_right = self.linear_interpolate_1d(self.forces_right, self.cfg.interp_resolution)
        self.reset()

    def linear_interpolate_1d(self, x: torch.Tensor, size: int):
        return torch.nn.functional.interpolate(x.view(1, 1, -1), size=size, mode="linear", align_corners=True).squeeze()

    def get_forces(self):
        """
        Computes the interpolated forces for each thruster based on the current command.

        The method calculates indices for the left and right thrusters by normalizing the current commands
        and scaling them according to the interpolation resolution. It then uses these indices to gather
        the corresponding interpolated forces from precomputed force arrays.

        Returns:
            torch.Tensor: A tensor containing the interpolated forces for the left and right thrusters.
        """

        idx_left = torch.round((self._current_cmds[:, 0] + 1) / 2 * (self.cfg.interp_resolution - 1)).to(torch.long)
        idx_right = torch.round((self._current_cmds[:, 1] + 1) / 2 * (self.cfg.interp_resolution - 1)).to(torch.long)

        # Using indices to gather interpolated forces for each thruster
        return torch.stack((self.interp_forces_left[idx_left], self.interp_forces_right[idx_right]), dim=1)

    def update_forces(self):
        """
        Updates the current commands based on the target commands and maximum delta, and calculates the thruster forces.

        This method performs the following steps:
        1. Computes the difference between the target commands and the current commands.
        2. Clamps the difference to be within the range defined by the maximum command delta.
        3. Updates the current commands by adding the clamped difference.
        4. Calculates the thruster forces and updates the relevant entries in the thruster forces array.

        Returns:
            torch.Tensor: The updated thruster forces.
        """
        # update the current command based on the target command and maximum delta
        delta = torch.clamp(self._target_cmds - self._current_cmds, -self._max_cmd_delta, self._max_cmd_delta)
        self._current_cmds += delta

        self.thruster_forces[:, [0, 3]] = self.get_forces()

        return self.thruster_forces

    def set_target_cmd(self, commands):
        self._target_cmds = commands

    def reset(self):
        self._current_cmds[:, :] = 0.0
        self._target_cmds[:, :] = 0.0
