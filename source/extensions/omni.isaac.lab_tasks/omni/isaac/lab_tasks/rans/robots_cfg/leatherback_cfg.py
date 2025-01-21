# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils import configclass

from .robot_core_cfg import RobotCoreCfg

from omni.isaac.lab_assets import LEATHERBACK_CFG  # isort: skip


@configclass
class LeatherbackRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker_height = 0.5

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left",
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    rew_action_rate_scale = -0.12
    rew_joint_accel_scale = -2.5e-6

    steering_scale = math.pi / 4.0
    """Multiplier for the steering position. The action is in the range [-1, 1]"""
    throttle_scale = 60.0
    """Multiplier for the throttle velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.06m"""

    # Spaces
    observation_space: int = 2
    state_space: int = 0
    action_space: int = 2
    gen_space: int = 0
