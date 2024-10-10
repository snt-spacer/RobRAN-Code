# Copyright (c) 2024, Antoine Richard
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.lab.sim import schemas
from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from omni.isaac.lab.utils import configclass

from . import racing_shapes


@configclass
class RacingShapeCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim."""

    pass


@configclass
class Gate3DCfg(RacingShapeCfg):
    """Configuration parameters for a 3d gate prim.

    See :meth:`spawn_gate_3d` for more information.
    """

    func: Callable = racing_shapes.spawn_gate_3d

    gate_width: float = MISSING
    """Width of the gate (in m)."""
    gate_height: float = MISSING
    """Height of the gate (in m)."""
    gate_depth: float = MISSING
    """Depth of the gate (in m)."""
    gate_thickness: float = MISSING
    """Thickness of the gate (in m)."""
    gate_corner_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    gate_front_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    gate_back_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """


class Gate2DCfg(RacingShapeCfg):
    """Configuration parameters for a 2d gate prim.

    See :meth:`spawn_gate_2d` for more information.
    """

    func: Callable = racing_shapes.spawn_gate_2d

    gate_width: float = MISSING
    """Width of the gate (in m)."""
    gate_height: float = MISSING
    """Height of the gate (in m)."""
    gate_depth: float = MISSING
    """Depth of the gate (in m)."""
    gate_thickness: float = MISSING
    """Thickness of the gate (in m)."""
    gate_corner_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    gate_front_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    gate_back_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """


class GatePylonsCfg(RacingShapeCfg):
    """Configuration parameters for a gate prim.

    See :meth:`spawn_gate_2_poles` for more information.
    """

    func: Callable = racing_shapes.spawn_gate_pylons

    gate_width: float = MISSING
    """Width of the gate (in m)."""
    pole_height: float = MISSING
    """Height of the gate (in m)."""
    pole_thickness: float = MISSING
    """Thickness of the gate (in m)."""
    left_pole_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    right_pole_color: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
