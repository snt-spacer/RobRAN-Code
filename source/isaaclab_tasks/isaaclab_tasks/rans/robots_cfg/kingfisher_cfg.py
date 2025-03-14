# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.kingfisher import KINGFISHER_CFG

from isaaclab.actuator_force.actuator_force import PropellerActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.physics.hydrodynamics import HydrodynamicsCfg
from isaaclab.physics.hydrostatics import HydrostaticsCfg
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
class KingfisherRobotCfg(RobotCoreCfg):
    """Core configuration for a RANS task."""

    robot_cfg: ArticulationCfg = KINGFISHER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    num_thrusters: int = 2
    robot_name: str = "kingfisher"
    thrusters_dof_name: list[str] = ["thruster_left", "thruster_right"]
    root_id_name: str = "base_link"
    marker_height: float = 0.5

    rew_action_rate_scale = -0.05
    rew_joint_accel_scale = -2.5e-6
    energy_reward_scale = -0.001
    max_energy = 2.0

    # Hydrostatics
    hydrostatics_cfg: HydrostaticsCfg = HydrostaticsCfg(
        mass=35.0,
        width=1.0,
        length=1.3,
        waterplane_area=0.33,
        draught_offset=0.21986,
        max_draught=0.20,
        average_hydrostatics_force=275.0,
    )

    # Hydrodynamics
    hydrodynamics_cfg: HydrodynamicsCfg = HydrodynamicsCfg(
        linear_damping=[0.0, 99.99, 99.99, 13.0, 13.0, 5.83],
        quadratic_damping=[17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724],
        use_drag_randomization=False,
        linear_damping_rand=[0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
        quadratic_damping_rand=[0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
    )

    # Thruster dynamics
    propeller_cfg: PropellerActuatorCfg = PropellerActuatorCfg()
    propeller_cfg.cmd_lower_range = -1.0
    propeller_cfg.cmd_upper_range = 1.0
    propeller_cfg.command_rate = (propeller_cfg.cmd_upper_range - propeller_cfg.cmd_lower_range) / 2.0
    propeller_cfg.forces_left = [
        -4.0,  # -1.0
        -4.0,  # -0.9
        -4.0,  # -0.8
        -4.0,  # -0.7
        -2.0,  # -0.6
        -1.0,  # -0.5
        0.0,  # -0.4
        0.0,  # -0.3
        0.0,  # -0.2
        0.0,  # -0.1
        0.0,  # 0.0
        0.0,  # 0.1
        0.0,  # 0.2
        0.5,  # 0.3
        1.5,  # 0.4
        4.75,  # 0.5
        8.25,  # 0.6
        16.0,  # 0.7
        19.5,  # 0.8
        19.5,  # 0.9
        19.5,  # 1.0
    ]
    propeller_cfg.forces_right = propeller_cfg.forces_left

    # Randomization
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=True, randomization_modes=["uniform"], body_name="base_link", max_delta=2.0
    )
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=False, randomization_modes=["uniform"], body_name="base_link", max_delta=0.05
    )
    wrench_rand_cfg = WrenchRandomizationCfg(
        enable=False,
        randomization_modes=["constant_uniform"],
        body_name="disturbance_body",
        uniform_force=(0, 0.25),
        uniform_torque=(0, 0.05),
        normal_force=(0, 0.25),
        normal_torque=(0, 0.025),
    )
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 2)],
        max_delta=[0.025],
        std=[0.001],
        clip_actions=[(-1, 1)],
    )
    action_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 2)],
        rescaling_ranges=[(0.85, 1.0)],
        clip_actions=[(-1, 1)],
    )

    # Spaces
    observation_space: int = 2
    state_space: int = 0
    action_space: int = 2
    gen_space: int = 0
