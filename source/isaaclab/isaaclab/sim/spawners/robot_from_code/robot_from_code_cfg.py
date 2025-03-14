# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from . import robot_from_code


@configclass
class RobotFromCodeCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning a robot using the :attr:`robot_gen_func` and :attr:`robot_gen_props`.

    This class is a base class used to generate robots from code. It includes the common parameters
    for spawning the robots, such as the path to the file and the function to use for spawning
    the robot.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the robot outside of the properties available by default when spawning the robot.

        If they are set to a value, then the properties are modified on the spawned robot in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    robot_gen_func: Callable = MISSING
    robot_gen_props: configclass = MISSING

    func: Callable = robot_from_code.spawn_robot_from_code

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Properties to apply to the articulation root."""

    fixed_tendons_props: schemas.FixedTendonsPropertiesCfg | None = None
    """Properties to apply to the fixed tendons (if any)."""

    joint_drive_props: schemas.JointDrivePropertiesCfg | None = None
    """Properties to apply to a joint."""
