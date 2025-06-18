# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class PropellerActuatorCfg:

    cmd_lower_range: float = MISSING
    cmd_upper_range: float = MISSING
    command_rate: float = MISSING  # Frequency of command updates in Hz
    forces: list = MISSING  # Forces for the thruster
    interp_resolution: int = 1001
    enable_randomization: bool = True
    randomization_range: float = 0.1  # Percentage of randomization on the forces


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
            forces (torch.Tensor): Tensor containing the forces from the configuration.
            thruster_forces (torch.Tensor): Tensor to store thruster forces for each environment.
            interp_forces (torch.Tensor): Tensor to store interpolated forces.
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.cfg = cfg

        self._ALL_INDICES = torch.arange(self.num_envs, dtype=torch.long, device=device)
        self._max_cmd_delta = cfg.command_rate * dt
        self._interp_scale = cfg.interp_resolution / (cfg.cmd_upper_range - cfg.cmd_lower_range)

        self._current_cmds = torch.zeros(num_envs, device=device)
        self._target_cmds = torch.zeros(num_envs, device=device)
        self.forces = torch.tensor(cfg.forces, device=device)
        self.thruster_forces = torch.zeros((num_envs, 3), device=device)
        self.interp_forces = self.linear_interpolate_1d(self.forces, cfg.interp_resolution)
        self.randomization_factor = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset(self._ALL_INDICES)

    def linear_interpolate_1d(self, x: torch.Tensor, size: int):
        return torch.nn.functional.interpolate(x.view(1, 1, -1), size=size, mode="linear", align_corners=True).squeeze()

    def get_forces(self):
        """
        Get the thruster forces for each environment.

        Returns:
            torch.Tensor: Tensor containing the thruster forces for each environment.
        """

        idx = torch.round((self._current_cmds[:] + 1) / 2 * (self.cfg.interp_resolution - 1)).to(torch.long)

        # Use the index to gather interpolated foce for the thurster
        return self.interp_forces[idx]

    def update_forces(self):
        """
        Update the thruster forces based on the target commands.
        """
        delta = torch.clamp(self._target_cmds - self._current_cmds, -self._max_cmd_delta, self._max_cmd_delta)
        self._current_cmds += delta

        self.thruster_forces[:, 0] = self.get_forces() * self.randomization_factor
        return self.thruster_forces

    def set_target_cmd(self, commands):
        """
        Set the target commands for the thruster.

        Args:
            commands (torch.Tensor): Tensor containing the target commands for each environment.
        """
        self._target_cmds = torch.clamp(commands, -1, 1)

    def reset(self, env_ids: torch.Tensor | None):
        """
        Reset the propeller actuator.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ALL_INDICES

        if self.cfg.enable_randomization:
            randomization = torch.rand(len(env_ids), device=self.device) * 2 - 1
            self.randomization_factor[env_ids] = randomization * self.cfg.randomization_range + 1

        self._current_cmds[env_ids] = 0.0
        self._target_cmds[env_ids] = 0.0
