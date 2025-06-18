# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg

# from isaaclab.sim import schemas
from isaaclab.utils import configclass

from . import navigation_shapes


@configclass
class NavigationShapeCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """
    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class PinSphereCfg(NavigationShapeCfg):
    """Configuration parameters for a a pin with a sphere prim on top.

    See :meth:`spawn_pin_sphere` for more information.
    """

    func: Callable = navigation_shapes.spawn_pin_with_sphere

    sphere_radius: float = MISSING
    """Radius of the sphere on top of the pin (in m)."""
    pin_radius: float = MISSING
    """Radius of the pin (in m)."""
    pin_length: float = MISSING
    """Length of the pin (in m)."""


@configclass
class DiamondCfg(NavigationShapeCfg):
    """Configuration parameters for a diamond mesh (similar to the sims).

    See :meth:`spawn_diamond` for more information.
    """

    func: Callable = navigation_shapes.spawn_diamond

    diamond_height: float = MISSING
    """Height of the diamond (in m)."""
    diamond_width: float = MISSING
    """Width of the diamond (in m)."""


@configclass
class BiColorDiamondCfg(NavigationShapeCfg):
    """Configuration parameters for a diamond mesh (similar to the sims).

    See :meth:`spawn_diamond` for more information.
    """

    func: Callable = navigation_shapes.spawn_bicolor_diamond

    diamond_height: float = MISSING
    """Height of the diamond (in m)."""
    diamond_width: float = MISSING
    """Width of the diamond (in m)."""
    front_material_path: str = "material_front"
    """Path to the visual material to use for the front of the prim. Defaults to "front_material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    back_material_path: str = "material_back"
    """Path to the visual material to use for the back of the prim. Defaults to "back_material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `back_material` is not None.
    """
    front_material: materials.VisualMaterialCfg | None = None
    """Back material properties.

    Note:
        If None, then no visual material will be added.
    """
    back_material: materials.VisualMaterialCfg | None = None
    """Back material properties.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class PinDiamondCfg(NavigationShapeCfg):
    """Configuration parameters for a pin with a diamond mesh on top (similar to the sims).

    See :meth:`spawn_pin_diamond` for more information.
    """

    func: Callable = navigation_shapes.spawn_pin_with_diamond

    diamond_height: float = MISSING
    """Height of the diamond (in m)."""
    diamond_width: float = MISSING
    """Width of the diamond (in m)."""
    pin_radius: float = MISSING
    """Radius of the pin (in m)."""
    pin_length: float = MISSING
    """Length of the pin (in m)."""


@configclass
class ArrowCfg(NavigationShapeCfg):
    """Configuration parameters for a pin with an arrow mesh on top.

    See :meth:`spawn_pin_arrow` for more information.
    """

    func: Callable = navigation_shapes.spawn_arrow

    arrow_body_radius: float = MISSING
    """Radius of the arrow's body (in m)."""
    arrow_body_length: float = MISSING
    """Length of the arrow's body (in m)."""
    arrow_head_radius: float = MISSING
    """Radius of the arrow's head (in m)."""
    arrow_head_length: float = MISSING
    """Length of the arrow's head (in m)."""


@configclass
class PinArrowCfg(NavigationShapeCfg):
    """Configuration parameters for a pin with an arrow mesh on top.

    See :meth:`spawn_pin_arrow` for more information.
    """

    func: Callable = navigation_shapes.spawn_pin_with_arrow

    arrow_body_radius: float = MISSING
    """Radius of the arrow's body (in m)."""
    arrow_body_length: float = MISSING
    """Length of the arrow's body (in m)."""
    arrow_head_radius: float = MISSING
    """Radius of the arrow's head (in m)."""
    arrow_head_length: float = MISSING
    """Length of the arrow's head (in m)."""
    pin_radius: float = MISSING
    """Radius of the pin (in m)."""
    pin_length: float = MISSING
    """Length of the pin (in m)."""


@configclass
class PositionMarker3DCfg(NavigationShapeCfg):
    """Configuration parameters for a position marker.

    See :meth:`spawn_position_marker_3d` for more information.
    """

    func: Callable = navigation_shapes.spawn_position_marker_3d

    pin_radius: float = MISSING
    """Radius of the marker's pins (in m)."""
    pin_length: float = MISSING
    """Length of the marker's pins (in m)."""
    sphere_radius: float = MISSING
    """Radius of the marker's spheres (in m)."""
    x_material_path: str = "material_x"
    """Path to the visual material to use for the x axis of the prim. Defaults to "material_x".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    y_material_path: str = "material_y"
    """Path to the visual material to use for the y axis of the prim. Defaults to "material_y".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    z_material_path: str = "material_z"
    """Path to the visual material to use for the z axis of the prim. Defaults to "material_z".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    x_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    y_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    z_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class PoseMarker3DCfg(NavigationShapeCfg):
    """Configuration parameters for a pose marker.

    See :meth:`spawn_pose_marker_3d` for more information.
    """

    func: Callable = navigation_shapes.spawn_pose_marker_3d

    arrow_body_radius: float = MISSING
    """Radius of the marker's arrows body (in m)."""
    arrow_body_length: float = MISSING
    """Length of the marker's arrows body (in m)."""
    arrow_head_radius: float = MISSING
    """Radius of the marker's arrows head (in m)."""
    arrow_head_length: float = MISSING
    """Length of the marker's arrows head (in m)."""
    x_material_path: str = "material_x"
    """Path to the visual material to use for the x axis of the prim. Defaults to "material_x".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    y_material_path: str = "material_y"
    """Path to the visual material to use for the y axis of the prim. Defaults to "material_y".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    z_material_path: str = "material_z"
    """Path to the visual material to use for the z axis of the prim. Defaults to "material_z".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `front_material` is not None.
    """
    x_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    y_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    z_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
