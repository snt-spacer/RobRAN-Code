# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab_assets.floating_platform import FLOATING_PLATFORM_CFG

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils import configclass

from .robot_core_cfg import RobotCoreCfg


@configclass
class FloatingPlatformRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = FLOATING_PLATFORM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker_height = 0.75
    has_reaction_wheel = False
    num_thrusters = 8

    thrusters_dof_name = [f"thruster_{i}" for i in range(1, num_thrusters + 1)]
    root_id_name = "Cylinder"
    rew_action_rate_scale = -0.12 / 8
    rew_joint_accel_scale = -2.5e-6

    max_thrust = 1.0
    """Maximum thrust of the thrusters in Newtons"""
    split_thrust = True
    """Split the thrust between the thrusters"""

    if has_reaction_wheel:
        reaction_wheel_dof_name = [
            "reaction_wheel",
        ]
        reaction_wheel_scale = 0.1  # [Nm]

    # Spaces
    observation_space: int = num_thrusters + 1 * has_reaction_wheel
    state_space: int = 0
    action_space: int = num_thrusters + 1 * has_reaction_wheel
    gen_space: int = 0
