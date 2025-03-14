# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_assets.robots.modular_freeflyer import MODULAR_FREEFLYER_2D_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import (
    ActionsRescalerCfg,
    CoMRandomizationCfg,
    MassRandomizationCfg,
    NoisyActionsCfg,
    WrenchRandomizationCfg,
)

from .robot_core_cfg import RobotCoreCfg


@configclass
class ModularFreeflyerRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_name: str = "modular_freeflyer"

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

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name="body", max_delta=0.25
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name="body", max_delta=0.05
    )
    wrench_rand_cfg = WrenchRandomizationCfg(
        enable=False,
        randomization_modes=["constant_uniform"],
        body_name="body",
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

    if is_reaction_wheel:
        reaction_wheel_dof_name = [
            "reaction_wheel",
        ]
        reaction_wheel_scale = 0.1  # [Nm]

    observation_space = num_thrusters + int(is_reaction_wheel)
    state_space = 0
    action_space = num_thrusters + int(is_reaction_wheel)
    gen_space = 0  # TODO: Add the generative space from the randomization

    def __post_init__(self):
        assert len(self.thruster_transforms) == self.num_thrusters, (
            "Invalid number of thruster transforms. The number of thruster transforms must match the number of"
            " thrusters."
        )
        assert len(self.thruster_max_thrust) == self.num_thrusters, (
            "Invalid number of thruster max thrust values. The number of thruster max thrust values must match the"
            " number of thrusters."
        )
