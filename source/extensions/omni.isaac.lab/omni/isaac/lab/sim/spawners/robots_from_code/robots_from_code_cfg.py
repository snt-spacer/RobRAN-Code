# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.lab.sim import schemas
from omni.isaac.lab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from omni.isaac.lab.utils import configclass

from . import robots_from_code


@configclass
class FromCodeCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    This class is a base class for spawning assets from files. It includes the common parameters
    for spawning assets from files, such as the path to the file and the function to use for spawning
    the asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    robot_gen_func: Callable = MISSING
    robot_gen_props: configclass = MISSING

    func: Callable = robots_from_code.spawn_from_code

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Properties to apply to the articulation root."""

    fixed_tendons_props: schemas.FixedTendonsPropertiesCfg | None = None
    """Properties to apply to the fixed tendons (if any)."""

    joint_drive_props: schemas.JointDrivePropertiesCfg | None = None
    """Properties to apply to a joint."""
