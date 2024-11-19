# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
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
    is_reaction_wheel = False
    num_thrusters = 8

    thrusters_dof_name = [f"thruster_{i}" for i in range(1, num_thrusters + 1)]
    root_id_name = "Cylinder"
    rew_action_rate_scale = -0.12
    rew_joint_accel_scale = -2.5e-6

    max_thrust = 1.0  # [N]
    split_thrust = True  # Split max thrust force among thrusters

    if is_reaction_wheel:
        reaction_wheel_dof_name = [
            "reaction_wheel",
        ]
        reaction_wheel_scale = 0.1  # [Nm]

    # action_space = spaces.Tuple([spaces.Discrete(2)] * num_thrusters)

    # spaces.MultiBinary(num_thrusters) if not is_reaction_wheel else spaces.Tuple((
    #     spaces.MultiBinary(num_thrusters),
    #     spaces.Box(low=-1.0, high=1.0, shape=(1,))
    # ))
