# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, matrix_from_quat

"""
Following Fossen's Equation,
Fossen, T. I. (1991). Nonlinear modeling and control of Underwater Vehicles. Doctoral thesis, Department of Engineering Cybernetics, Norwegian Institute of Technology (NTH), June 1991.
"""


@configclass
class HydrostaticsCfg:
    """Configuration for default hydrostatics."""

    gravity: float = -9.81
    water_density: float = 997.0  # Kg/m^3

    mass: float = MISSING  # Kg
    width: float = MISSING  # m
    length: float = MISSING  # m
    waterplane_area: float = MISSING  # m^2
    draught_offset: float = MISSING  # m (distance from base_link to bottom of the hull)
    max_draught: float = MISSING  # m (max draught of the vessel)
    average_hydrostatics_force: float = MISSING
    amplify_torque: float = 1.0


class Hydrostatics:
    def __init__(self, num_envs, device, cfg: HydrostaticsCfg):

        self.cfg = cfg

        self.num_envs = num_envs
        self.device = device
        self.drag = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

        # Buoyancy
        self.archimedes_force_global = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_torque_global = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_force_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.archimedes_torque_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        return

    def compute_archimedes_metacentric_global(self, submerged_volume, rpy):
        roll, pitch = rpy[:, 0], rpy[:, 1]  # roll and pitch are given in global frame

        # compute buoyancy force
        self.archimedes_force_global[:, 2] = -self.cfg.water_density * self.cfg.gravity * submerged_volume

        # torques expressed in global frame, size is (num_envs,3)
        self.archimedes_torque_global[:, 0] = (
            -1 * self.cfg.width / 2.0 * (torch.sin(roll) * self.archimedes_force_global[:, 2])
        )
        self.archimedes_torque_global[:, 1] = (
            -1 * self.cfg.length / 2.0 * (torch.sin(pitch) * self.archimedes_force_global[:, 2])
        )

        self.archimedes_torque_global[:, 0] = (
            -1 * self.cfg.width / 2.0 * (torch.sin(roll) * self.cfg.average_hydrostatics_force)
        )  # cannot multiply by the hydrostatics force in isaac sim because of the simulation rate (low then high value)
        self.archimedes_torque_global[:, 1] = (
            -1 * self.cfg.length / 2.0 * (torch.sin(pitch) * self.cfg.average_hydrostatics_force)
        )

        return self.archimedes_force_global, self.archimedes_torque_global

    def compute_submerged_volume(self, position):

        draught = torch.clamp(self.cfg.draught_offset - position[:, 2], 0, self.cfg.max_draught)

        submerged_volume = draught * self.cfg.waterplane_area

        return submerged_volume

    def compute_archimedes_metacentric_local(self, position, quaternions):

        roll, pitch, yaw = euler_xyz_from_quat(quaternions)
        euler = torch.stack((roll, pitch, yaw), dim=1)

        # get archimedes global force
        submerged_volume = self.compute_submerged_volume(position)
        self.compute_archimedes_metacentric_global(submerged_volume, euler)

        # get rotation matrix from quaternions in world frame, size is (3*num_envs, 3)
        R = matrix_from_quat(quaternions)

        # Arobot = Rworld * Aworld. Resulting matrix should be size (3*num_envs, 3) * (num_envs,3) =(num_envs,3)
        self.archimedes_force_local = torch.bmm(
            R.mT, torch.unsqueeze(self.archimedes_force_global, 1).mT
        )  # add batch dimension to tensor and transpose it
        self.archimedes_force_local = self.archimedes_force_local.mT.squeeze(1)  # remove batch dimension to tensor

        self.archimedes_torque_local = torch.bmm(R.mT, torch.unsqueeze(self.archimedes_torque_global, 1).mT)
        self.archimedes_torque_local = self.archimedes_torque_local.mT.squeeze(1)

        # not sure if torque have to be multiply by the rotation matrix also.
        self.archimedes_torque_local = self.archimedes_torque_global

        return torch.hstack([
            self.archimedes_force_local,
            self.archimedes_torque_local * self.cfg.amplify_torque,
        ])
