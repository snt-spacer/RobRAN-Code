# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_assets.robots.intball2_local import INTBALL2_LOCAL_CFG

from isaaclab.assets import ArticulationCfg

# from omni.isaac.lab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import ActionsRescalerCfg, CoMRandomizationCfg, NoisyActionsCfg

from .robot_core_cfg import RobotCoreCfg


@configclass
class IntBall2RobotCfg(RobotCoreCfg):
    """Configuration for the JAXA Int-Ball2 robot in RANS tasks."""

    robot_name: str = "intball2"

    robot_cfg: ArticulationCfg = INTBALL2_LOCAL_CFG.replace(prim_path="/World/envs/env_.*/IntBall2")

    A = 0.035  # Distance from center along X-axis
    B = 0.045  # Distance from center along Y-axis
    C = 0.035  # Distance from center along Z-axis

    marker_height = 0.5
    is_reaction_wheel = False
    num_thrusters = 8  # IntBall2 has 8 thrusters, arranged in a 3D cuboid pattern

    root_body_name = "body"
    rew_action_rate_scale = -0.12 / 8
    rew_torque_balance_scale = -0.05  # penalize torque balance
    # rew_drag_torque_scale = -0.02 # penalize excessive drag torque effects - not yet used since propellers physics is not yet implemented

    action_mode = "discrete"  # "continuous" or "discrete"

    # Thruster transforms (pos, rot: 3D): [X, Y, Z, RotX, RotY, RotZ]
    thruster_transforms = [
        [2 * A, 2 * B, 2 * C, 0, 0, 0],  # Thruster 1
        [2 * A, -2 * B, 2 * C, 0, 0, math.pi],  # Thruster 2
        [2 * A, -2 * B, -2 * C, 0, math.pi, 0],  # Thruster 3
        [2 * A, 2 * B, -2 * C, 0, math.pi, math.pi],  # Thruster 4
        [-2 * A, 2 * B, -2 * C, math.pi, 0, 0],  # Thruster 5
        [-2 * A, 2 * B, 2 * C, math.pi, 0, math.pi],  # Thruster 6
        [-2 * A, -2 * B, 2 * C, math.pi, math.pi, 0],  # Thruster 7
        [-2 * A, -2 * B, -2 * C, math.pi, math.pi, math.pi],  # Thruster 8
    ]
    # Structure: [X, Y, Z, Rotation around X, Rotation around Y, Rotation around Z]

    thruster_max_thrust = [1.0] * 8
    # Thrust scaling factors (from the paper)
    thrust_scale_factors = [0.8] * num_thrusters  # Placeholder
    # drag_torque_factors = [0.1] * num_thrusters  # Placeholder, needs tuning
    split_thrust = True  # Thrusters work in coordinated pairs

    # # Sensors
    # body_contact_forces: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/IntBall2/body",
    #     update_period=0.0,
    #     history_length=3,
    #     debug_vis=True,
    # )

    # Randomization
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name="body", max_delta=0.05
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

    action_space = num_thrusters
    observation_space = num_thrusters
    # + drag on each propeller
    state_space = 0
    gen_space = 0

    def __post_init__(self):
        assert (
            len(self.thruster_transforms) == self.num_thrusters
        ), "Number of thruster transforms must match the number of thrusters."
        assert (
            len(self.thruster_max_thrust) == self.num_thrusters
        ), "Number of max thrust values must match the number of thrusters."
        assert (
            len(self.thrust_scale_factors) == self.num_thrusters
        ), "Thrust scaling factors must match number of thrusters."
        # assert len(self.drag_torque_factors) == self.num_thrusters, (
        #     "Drag torque factors must match number of thrusters."
        # )
