# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils import configclass

from .robot_core_cfg import RobotCoreCfg

from omni.isaac.lab_assets import JETBOT_CFG  # isort: skip


@configclass
class JetbotRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    wheels_dof_names = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]

    rew_action_rate_scale = -0.12
    rew_joint_accel_scale = -2.5e-6

    wheel_scale = 1 / 0.0500
    """Multiplier for the wheel velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.05m"""

    # Spaces
    observation_space: int = 2
    state_space: int = 0
    action_space: int = 2
    gen_space: int = 0
