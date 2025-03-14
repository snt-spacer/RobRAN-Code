# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import (
    ActionsRescalerCfg,
    CoMRandomizationCfg,
    MassRandomizationCfg,
    NoisyActionsCfg,
)

from .robot_core_cfg import RobotCoreCfg

from isaaclab_assets.robots.turtlebot2 import TURTLEBOT2_CFG  # isort: skip


@configclass
class TurtleBot2RobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_name: str = "Turtlebot2"

    robot_cfg: ArticulationCfg = TURTLEBOT2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    marker_height = 0.5

    wheels_dof_names = [
        "left_wheel",
        "right_wheel",
    ]

    rew_action_rate_scale = -0.12
    rew_joint_accel_scale = -2.5e-6

    wheel_radius = 0.038  # m
    offset_wheel_space_radius = 0.12
    max_speed = 0.45  # m/s
    max_rotational_speed = 0.9  # rad/s
    max_rotation_speed_wheels = 17.1  # rad/s

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=True, randomization_modes=["normal"], body_name="core", max_delta=0.1, mass_change_rate=-0.025
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["normal"], body_name="core", max_delta=0.05, std=0.01
    )
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=True,
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
