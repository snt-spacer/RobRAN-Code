# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import (
    ActionsRescalerCfg,
    CoMRandomizationCfg,
    MassRandomizationCfg,
    NoisyActionsCfg,
)

from .robot_core_cfg import RobotCoreCfg

from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG  # isort: skip


@configclass
class LeatherbackRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_name: str = "leatherback"

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

    rew_energy_penalty = 0  # -0.025
    rew_action_rate_scale = -0.12
    rew_joint_accel_scale = 0  # -2.5e-6

    steering_scale = math.pi / 4.0
    """Multiplier for the steering position. The action is in the range [-1, 1]"""
    throttle_scale = 60.0
    """Multiplier for the throttle velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.06m"""

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=False,
        randomization_modes=["normal", "constant_time_decay"],
        body_name="chassis",
        max_delta=0.1,
        mass_change_rate=-0.025,
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["normal"], body_name="chassis", max_delta=0.05, std=0.01
    )
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=False,
        randomization_modes=["normal"],
        slices=[0, 1],
        max_delta=[0.025, 0.1],
        std=[0.01, 0.05],
        clip_actions=[(-1, 1), (-1, 1)],
    )
    action_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[0, 1],
        rescaling_ranges=[(0.8, 1.0), (0.8, 1.0)],
        clip_actions=[(-1, 1), (-1, 1)],
    )

    chassis_contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Rigid_Bodies/Chassis",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )
    # Spaces
    observation_space: int = 2
    state_space: int = 0
    action_space: int = 2
    gen_space: int = 0  # TODO: Add the generative space from the randomization
