# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab_assets.modular_freeflyer import MODULAR_FREEFLYER_2D_CFG

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils import configclass

from .robot_core_cfg import RobotCoreCfg


@configclass
class ModularFreeflyerRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = MODULAR_FREEFLYER_2D_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker_height = 0.75
    is_reaction_wheel = False
    num_thrusters = 8

    root_body_name = "body"
    x_lock_name = "x_lock_joint"
    y_lock_name = "y_lock_joint"
    z_lock_name = "z_lock_joint"

    rew_action_rate_scale = -0.12 / 8
    rew_joint_accel_scale = -2.5e-6

    action_mode = "continuous"

    thruster_transforms = [
        [0.2192031, 0.2192031, -math.pi / 4.0],
        [0.2192031, 0.2192031, math.pi * 3.0 / 4.0],
        [-0.2192031, 0.2192031, math.pi / 4.0],
        [-0.2192031, 0.2192031, math.pi * 5.0 / 4.0],
        [-0.2192031, -0.2192031, math.pi * 3.0 / 4.0],
        [-0.2192031, -0.2192031, -math.pi / 4.0],
        [0.2192031, -0.2192031, math.pi * 5.0 / 4.0],
        [0.2192031, -0.2192031, math.pi / 4.0],
    ]
    thruster_max_thrust = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    split_thrust = True  # Split max thrust force among thrusters

    if is_reaction_wheel:
        reaction_wheel_dof_name = [
            "reaction_wheel",
        ]
        reaction_wheel_scale = 0.1  # [Nm]

    action_space = num_thrusters + int(is_reaction_wheel)
    observation_space = num_thrusters + int(is_reaction_wheel)
    state_space = 0
    gen_space = 0

    def __post_init__(self):
        assert len(self.thruster_transforms) == self.num_thrusters, (
            "Invalid number of thruster transforms. The number of thruster transforms must match the number of"
            " thrusters."
        )
        assert len(self.thruster_max_thrust) == self.num_thrusters, (
            "Invalid number of thruster max thrust values. The number of thruster max thrust values must match the"
            " number of thrusters."
        )
