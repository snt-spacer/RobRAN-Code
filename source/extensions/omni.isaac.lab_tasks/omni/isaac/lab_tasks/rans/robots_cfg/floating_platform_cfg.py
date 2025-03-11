# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab_assets.floating_platform import FLOATING_PLATFORM_CFG

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.rans.domain_randomization import (
    ActionsRescalerCfg,
    CoMRandomizationCfg,
    MassRandomizationCfg,
    NoisyActionsCfg,
    WrenchRandomizationCfg,
)

from .robot_core_cfg import RobotCoreCfg


@configclass
class FloatingPlatformRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_name: str = "FloatingPlatform"

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

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name="Cylinder", max_delta=0.25
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name="Cylinder", max_delta=0.05
    )
    wrench_rand_cfg = WrenchRandomizationCfg(
        enable=False,
        randomization_modes=["constant_uniform"],
        body_name="Cylinder",
        uniform_force=(0, 0.25),
        uniform_torque=(0, 0.05),
        normal_force=(0, 0.25),
        normal_torque=(0, 0.025),
    )
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 8)],
        max_delta=[0.1],
        std=[0.025],
        clip_actions=[(0, 1)],
    )
    action_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
        enable=False,
        randomization_modes=["uniform"],
        slices=[(0, 8)],
        rescaling_ranges=[(0.8, 1.0)],
        clip_actions=[(0, 1)],
    )

    if has_reaction_wheel:
        reaction_wheel_dof_name = [
            "reaction_wheel",
        ]
        reaction_wheel_scale = 0.1  # [Nm]

    # Sensors
    body_contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/Cylinder",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )

    # Spaces
    observation_space: int = num_thrusters + 1 * has_reaction_wheel
    state_space: int = 0
    action_space: int = num_thrusters + 1 * has_reaction_wheel
    gen_space: int = 0  # TODO: Add the generative space from the randomization
