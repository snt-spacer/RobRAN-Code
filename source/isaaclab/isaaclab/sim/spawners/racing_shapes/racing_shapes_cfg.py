# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.utils import configclass

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

    width: float = MISSING
    """Width of the gate (in m)."""
    height: float = MISSING
    """Height of the gate (in m)."""
    depth: float = MISSING
    """Depth of the gate (in m)."""
    thickness: float = MISSING
    """Thickness of the gate (in m)."""
    corner_material_path: str = "material_corner"
    """Path to the visual material to use for the corners of the prim. Defaults to "material_corner".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `corner_material` is not None.
    """
    front_material_path: str = "material_front"
    """Path to the visual material to use for the front of the prim. Defaults to "material_front".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    back_material_path: str = "material_back"
    """Path to the visual material to use for the back of the prim. Defaults to "material_back".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `back_material` is not None.
    """
    corner_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    front_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    back_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class Gate2DCfg(RacingShapeCfg):
    """Configuration parameters for a 2d gate prim.

    See :meth:`spawn_gate_2d` for more information.
    """

    func: Callable = racing_shapes.spawn_gate_2d

    width: float = MISSING
    """Width of the gate (in m)."""
    height: float = MISSING
    """Height of the gate (in m)."""
    depth: float = MISSING
    """Depth of the gate (in m)."""
    thickness: float = MISSING
    """Thickness of the gate (in m)."""
    corner_material_path: str = "material_corner"
    """Path to the visual material to use for the corners of the prim. Defaults to "material_corner.

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `corner_material` is not None.
    """
    front_material_path: str = "material_front"
    """Path to the visual material to use for the front of the prim. Defaults to "material_front".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    back_material_path: str = "material_back"
    """Path to the visual material to use for the back of the prim. Defaults to "material_back".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `back_material` is not None.
    """
    corner_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    front_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    back_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class GatePylonsCfg(RacingShapeCfg):
    """Configuration parameters for a gate prim.

    See :meth:`spawn_gate_2_poles` for more information.
    """

    func: Callable = racing_shapes.spawn_gate_pylons

    width: float = MISSING
    """Width of the gate (in m)."""
    height: float = MISSING
    """Height of the gate (in m)."""
    radius: float = MISSING
    """Thickness of the gate (in m)."""
    left_pole_material_path: str = "material_left"
    """Path to the visual material to use for the left pole. Defaults to "material_left".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    right_pole_material_path: str = "material_right"
    """Path to the visual material to use for the right pole. Defaults to "material_right".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    left_pole_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    right_pole_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
