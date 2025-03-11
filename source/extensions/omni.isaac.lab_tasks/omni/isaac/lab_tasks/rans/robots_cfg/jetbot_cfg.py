# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.rans.domain_randomization import (
    ActionsRescalerCfg,
    CoMRandomizationCfg,
    MassRandomizationCfg,
    NoisyActionsCfg,
)

from .robot_core_cfg import RobotCoreCfg

from omni.isaac.lab_assets import JETBOT_CFG  # isort: skip


@configclass
class JetbotRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_name: str = "jetbot"

    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker_height = 0.5

    wheels_dof_names = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]

    wheel_scale = 1 / 0.05  # Vmax = 1.0m/s, wheel radius is 0.05m
    """Multiplier for the wheel velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.05m"""

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=False, randomization_modes=["normal"], body_name="chassis", max_delta=0.1, mass_change_rate=-0.025
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["normal"], body_name="chassis", max_delta=0.05, std=0.01
    )
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 2)],
        max_delta=[0.025],
        std=[0.01],
        clip_actions=[(-1, 1)],
    )
    actions_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
        enable=False,
        randomization_modes=["uniform"],
        rescaling_ranges=[(0.8, 1.0)],
        slices=[(0, 2)],
        clip_actions=[(-1, 1)],
    )

    # Spaces
    observation_space: int = 2
    state_space: int = 0
    action_space: int = 2
    gen_space: int = 0

    # Sensors
    chassis_contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/chassis",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )

    # Reward
    rew_energy_penalty = -0.025
    rew_action_rate_scale = -0.12
    rew_joint_accel_scale = -2.5e-6
