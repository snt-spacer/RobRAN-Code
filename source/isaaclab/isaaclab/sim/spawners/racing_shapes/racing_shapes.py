# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_visual_material, clone

if TYPE_CHECKING:
    from . import racing_shapes_cfg


@clone
def spawn_gate_3d(
    prim_path: str,
    cfg: racing_shapes_cfg.Gate3DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 3D gate prim using the given attributes.

    It uses a set of cubes to create a Gate.
    This is meant to be used in racing style tasks for drones or other 6DoF vehicles.
    The width and height define the available space INSIDE the gate. The thickness defines the thickness of the gate,
    that is how much it extends beyond the width and height. The depth defines the depth of the gate.

    Cut on the Z axis:

    -Y                      0                        -Y
    <-------><-------------------------------><------->
    Thickness             Width               Thickness

    Cut on the Y axis:

    -Z                       0                       +Z
    <-------><-------------------------------><------->
    Thickness             Height              Thickness

    Cut on the X axis:

    -Y  0  +Y
    <------->
      Depth

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    make_gate_3d(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_gate_2d(
    prim_path: str,
    cfg: racing_shapes_cfg.Gate2DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 2D gate prim using the given attributes.

    It uses a set of cubes to create a Gate.
    This is meant to be used in racing style tasks for cars or other 3DoF vehicles.
    The width and height define the available space INSIDE the gate. The thickness defines the thickness of the gate,
    that is how much it extends beyond the width and height. The depth defines the depth of the gate.

    Cut on the Z axis:

    -Y                      0                      -Y
    <-------><-------------------------------><------->
    Thickness             Width               Thickness

    Cut on the Y axis:

    0                                       +Z
    <-------------------------------><------->
                  Height             Thickness

    Cut on the X axis:

    -Y  0  +Y
    <------->
      Depth


    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    make_gate_2d(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_gate_pylons(
    prim_path: str,
    cfg: racing_shapes_cfg.GatePylonsCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 2D gate prim using the given attributes.

    It uses a set of cubes to create a Gate.
    This is meant to be used in racing style tasks for boats, or planes the gate is composed of two pylons with
    different colors to differentiate the front and back of the gate.
    The width define the available space INSIDE the gate. The Radius defines the radius of the pylons making up the gate,
    that is how much it extends beyond the width. The depth defines the depth of the gate.
    The height defines the height of the pylons making up the gate.

    Cut on the Z axis:

    -Y                      0                        -Y
    <-------><-------------------------------><------->
     Radius               Width                Radius

    Cut on the Y axis:

    -Z                       0                       +Z
    <------------------------------------------------->
                           Height

    Cut on the X axis:

    -Y  0  +Y
    <------->
      Depth


    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    make_gate_pylons(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


def make_gate_3d(
    prim_path: str,
    cfg: racing_shapes_cfg.Gate3DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 3D gate prim with using the given attributes.
    The front, back and corners of the gate have different colors, which can be set using the
    `gate_front_color`, `gate_back_color` and `gate_corner_color` attributes in the config.
    The width and height define the available space INSIDE the gate. The thickness defines the thickness of the gate,
    that is how much it extends beyond the width and height. The depth defines the depth of the gate.

    Cut on the Z axis:

    -Y                      0                        -Y
    <-------><-------------------------------><------->
    Thickness             Width               Thickness

    Cut on the Y axis:

    -Z                       0                       +Z
    <-------><-------------------------------><------->
    Thickness             Height              Thickness

    Cut on the X axis:

    -Y  0  +Y
    <------->
      Depth

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        attributes: The attributes to apply to the prim.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created root prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    container = prim_utils.create_prim(mesh_prim_path, "Xform")

    # Correct the dims
    cfg.width = cfg.width + cfg.thickness
    cfg.height = cfg.height + cfg.thickness

    # Corners
    corner_size = min([cfg.thickness, cfg.depth])
    corner_scale = [dim / corner_size for dim in [cfg.depth, cfg.thickness, cfg.thickness]]
    corner_attribute = {"size": corner_size}
    bottom_left_corner_position = [0, -cfg.width / 2, -cfg.height / 2]
    top_left_corner_position = [0, -cfg.width / 2, cfg.height / 2]
    top_right_corner_position = [0, cfg.width / 2, cfg.height / 2]
    bottom_right_corner_position = [0, cfg.width / 2, -cfg.height / 2]
    bottom_left_corner = prim_utils.create_prim(
        str(container.GetPath().AppendChild("bottom_left_corner")),
        "Cube",
        attributes=corner_attribute,
        translation=bottom_left_corner_position,
        scale=corner_scale,
    )
    top_left_corner = prim_utils.create_prim(
        str(container.GetPath().AppendChild("top_left_corner")),
        "Cube",
        attributes=corner_attribute,
        translation=top_left_corner_position,
        scale=corner_scale,
    )
    top_right_corner = prim_utils.create_prim(
        str(container.GetPath().AppendChild("top_right_corner")),
        "Cube",
        attributes=corner_attribute,
        translation=top_right_corner_position,
        scale=corner_scale,
    )
    bottom_right_corner = prim_utils.create_prim(
        str(container.GetPath().AppendChild("bottom_right_corner")),
        "Cube",
        attributes=corner_attribute,
        translation=bottom_right_corner_position,
        scale=corner_scale,
    )
    # Front
    front_hzt_size = min([cfg.depth / 2, cfg.width - cfg.thickness, cfg.thickness])
    front_vrt_size = min([cfg.depth / 2, cfg.thickness, cfg.height - cfg.thickness])
    front_hzt_attribute = {"size": front_hzt_size}
    front_vrt_attribute = {"size": front_vrt_size}
    front_hzt_scale = [dim / front_hzt_size for dim in [cfg.depth / 2, cfg.width - cfg.thickness, cfg.thickness]]
    front_vrt_scale = [dim / front_vrt_size for dim in [cfg.depth / 2, cfg.thickness, cfg.height - cfg.thickness]]
    front_left_position = [-cfg.depth / 4, -cfg.width / 2, 0]
    front_right_position = [-cfg.depth / 4, cfg.width / 2, 0]
    front_top_position = [-cfg.depth / 4, 0, cfg.height / 2]
    front_bottom_position = [-cfg.depth / 4, 0, -cfg.height / 2]
    front_left = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_left")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=front_left_position,
        scale=front_vrt_scale,
    )
    front_right = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_right")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=front_right_position,
        scale=front_vrt_scale,
    )
    front_top = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_top")),
        "Cube",
        attributes=front_hzt_attribute,
        translation=front_top_position,
        scale=front_hzt_scale,
    )
    front_bottom = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_bottom")),
        "Cube",
        attributes=front_hzt_attribute,
        translation=front_bottom_position,
        scale=front_hzt_scale,
    )
    # Back
    back_left_position = [cfg.depth / 4, -cfg.width / 2, 0]
    back_right_position = [cfg.depth / 4, cfg.width / 2, 0]
    back_top_position = [cfg.depth / 4, 0, cfg.height / 2]
    back_bottom_position = [cfg.depth / 4, 0, -cfg.height / 2]
    back_left = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_left")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=back_left_position,
        scale=front_vrt_scale,
    )
    back_right = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_right")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=back_right_position,
        scale=front_vrt_scale,
    )
    back_top = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_top")),
        "Cube",
        attributes=front_hzt_attribute,
        translation=back_top_position,
        scale=front_hzt_scale,
    )
    back_bottom = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_bottom")),
        "Cube",
        attributes=front_hzt_attribute,
        translation=back_bottom_position,
        scale=front_hzt_scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(bottom_left_corner.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(top_right_corner.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(top_left_corner.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(bottom_right_corner.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_left.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_right.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_top.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_bottom.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_left.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_right.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_top.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_bottom.GetPath()), cfg.collision_props)

    if cfg.front_material is not None:
        if not cfg.front_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.front_material_path}"
        else:
            material_path = cfg.front_material_path
        # create material
        cfg.front_material.func(material_path, cfg.front_material)
        # apply material
        bind_visual_material(str(front_left.GetPath()), material_path)
        bind_visual_material(str(front_right.GetPath()), material_path)
        bind_visual_material(str(front_top.GetPath()), material_path)
        bind_visual_material(str(front_bottom.GetPath()), material_path)
    if cfg.back_material is not None:
        if not cfg.back_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.back_material_path}"
        else:
            material_path = cfg.back_material_path
        # create material
        cfg.back_material.func(material_path, cfg.back_material)
        # apply material
        bind_visual_material(str(back_left.GetPath()), material_path)
        bind_visual_material(str(back_right.GetPath()), material_path)
        bind_visual_material(str(back_top.GetPath()), material_path)
        bind_visual_material(str(back_bottom.GetPath()), material_path)
    if cfg.corner_material is not None:
        if not cfg.corner_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.corner_material_path}"
        else:
            material_path = cfg.corner_material_path
        # create material
        cfg.corner_material.func(material_path, cfg.corner_material)
        # apply material
        bind_visual_material(str(bottom_left_corner.GetPath()), material_path)
        bind_visual_material(str(top_left_corner.GetPath()), material_path)
        bind_visual_material(str(top_right_corner.GetPath()), material_path)
        bind_visual_material(str(bottom_right_corner.GetPath()), material_path)

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)

    return prim


def make_gate_2d(
    prim_path: str,
    cfg: racing_shapes_cfg.Gate2DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 2D gate prim using the given attributes.
    The front, back and corners of the gate have different colors, which can be set using the
    `gate_front_color`, `gate_back_color` and `gate_corner_color` attributes in the config.
    The width and height define the available space INSIDE the gate. The thickness defines the thickness of the gate,
    that is how much it extends beyond the width and height. The depth defines the depth of the gate.

    Cut on the Z axis:

    -Y                      0                        -Y
    <-------><-------------------------------><------->
    Thickness             Width               Thickness

    Cut on the Y axis:

    0                                       +Z
    <-------------------------------><------->
                  Height             Thickness

    Cut on the X axis:

    -Y  0  +Y
    <------->
      Depth

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        attributes: The attributes to apply to the prim.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created root prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    container = prim_utils.create_prim(mesh_prim_path, "Xform")

    # Correct the dims
    cfg.width = cfg.width + cfg.thickness
    cfg.height = cfg.height + cfg.thickness / 2

    # Corners
    corner_size = min([cfg.thickness, cfg.depth])
    corner_scale = [dim / corner_size for dim in [cfg.depth, cfg.thickness, cfg.thickness]]
    corner_attribute = {"size": corner_size}
    top_left_corner_position = [0, -cfg.width / 2, cfg.height]
    top_right_corner_position = [0, cfg.width / 2, cfg.height]
    top_left_corner = prim_utils.create_prim(
        str(container.GetPath().AppendChild("top_left_corner")),
        "Cube",
        attributes=corner_attribute,
        translation=top_left_corner_position,
        scale=corner_scale,
    )
    top_right_corner = prim_utils.create_prim(
        str(container.GetPath().AppendChild("top_right_corner")),
        "Cube",
        attributes=corner_attribute,
        translation=top_right_corner_position,
        scale=corner_scale,
    )
    # Front
    front_hzt_size = min([cfg.depth / 2, cfg.width - cfg.thickness, cfg.thickness])
    front_vrt_size = min([cfg.depth / 2, cfg.thickness, cfg.height])
    front_hzt_attribute = {"size": front_hzt_size}
    front_vrt_attribute = {"size": front_vrt_size}
    front_hzt_scale = [dim / front_hzt_size for dim in [cfg.depth / 2, cfg.width - cfg.thickness, cfg.thickness]]
    front_vrt_scale = [dim / front_vrt_size for dim in [cfg.depth / 2, cfg.thickness, cfg.height - cfg.thickness]]
    front_left_position = [-cfg.depth / 4, -cfg.width / 2, cfg.height / 2]
    front_right_position = [-cfg.depth / 4, cfg.width / 2, cfg.height / 2]
    front_top_position = [-cfg.depth / 4, 0, cfg.height]
    front_left = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_left")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=front_left_position,
        scale=front_vrt_scale,
    )
    front_right = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_right")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=front_right_position,
        scale=front_vrt_scale,
    )
    front_top = prim_utils.create_prim(
        str(container.GetPath().AppendChild("front_top")),
        "Cube",
        attributes=front_hzt_attribute,
        translation=front_top_position,
        scale=front_hzt_scale,
    )
    # Back
    back_left_position = [cfg.depth / 4, -cfg.width / 2, cfg.height / 2]
    back_right_position = [cfg.depth / 4, cfg.width / 2, cfg.height / 2]
    back_top_position = [cfg.depth / 4, 0, cfg.height]
    back_left = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_left")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=back_left_position,
        scale=front_vrt_scale,
    )
    back_right = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_right")),
        "Cube",
        attributes=front_vrt_attribute,
        translation=back_right_position,
        scale=front_vrt_scale,
    )
    back_top = prim_utils.create_prim(
        str(container.GetPath().AppendChild("back_top")),
        "Cube",
        attributes=front_hzt_attribute,
        translation=back_top_position,
        scale=front_hzt_scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(top_right_corner.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(top_left_corner.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_left.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_right.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(front_top.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_left.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_right.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back_top.GetPath()), cfg.collision_props)

    if cfg.front_material is not None:
        if not cfg.front_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.front_material_path}"
        else:
            material_path = cfg.front_material_path
        # create material
        cfg.front_material.func(material_path, cfg.front_material)
        # apply material
        bind_visual_material(str(front_left.GetPath()), material_path)
        bind_visual_material(str(front_right.GetPath()), material_path)
        bind_visual_material(str(front_top.GetPath()), material_path)
    if cfg.back_material is not None:
        if not cfg.back_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.back_material_path}"
        else:
            material_path = cfg.back_material_path
        # create material
        cfg.back_material.func(material_path, cfg.back_material)
        # apply material
        bind_visual_material(str(back_left.GetPath()), material_path)
        bind_visual_material(str(back_right.GetPath()), material_path)
        bind_visual_material(str(back_top.GetPath()), material_path)
    if cfg.corner_material is not None:
        if not cfg.corner_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.corner_material_path}"
        else:
            material_path = cfg.corner_material_path
        # create material
        cfg.corner_material.func(material_path, cfg.corner_material)
        # apply material
        bind_visual_material(str(top_left_corner.GetPath()), material_path)
        bind_visual_material(str(top_right_corner.GetPath()), material_path)

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)

    return prim


def make_gate_pylons(
    prim_path: str,
    cfg: racing_shapes_cfg.GatePylonsCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 2D pylon gate prim using the given attributes.
    The front, back and corners of the gate have different colors, which can be set using the
    `gate_front_color`, `gate_back_color` and `gate_corner_color` attributes in the config.
    The width define the available space INSIDE the gate. The Radius defines the radius of the pylons making up the gate,
    that is how much it extends beyond the width. The depth defines the depth of the gate.
    The height defines the height of the pylons making up the gate.

    Cut on the Z axis:

    -Y                      0                        -Y
    <-------><-------------------------------><------->
     Radius               Width                Radius

    Cut on the Y axis:

    -Z                       0                       +Z
    <------------------------------------------------->
                           Height

    Cut on the X axis:

    -Y  0  +Y
    <------->
      Depth

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        attributes: The attributes to apply to the prim.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.

    Returns:
        The created root prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    container = prim_utils.create_prim(mesh_prim_path, "Xform")

    # Correct the dims
    cfg.width = cfg.width + cfg.radius * 2
    attributes = {"height": cfg.height, "radius": cfg.radius, "axis": "Z"}
    left_position = [0, -cfg.width / 2, 0]
    right_position = [0, cfg.width / 2, 0]
    left_pylon = prim_utils.create_prim(
        str(container.GetPath().AppendChild("left_pylon")),
        "Capsule",
        attributes=attributes,
        translation=left_position,
    )
    right_pylon = prim_utils.create_prim(
        str(container.GetPath().AppendChild("right_pylon")),
        "Capsule",
        attributes=attributes,
        translation=right_position,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(right_pylon.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(left_pylon.GetPath()), cfg.collision_props)

    if cfg.left_pole_material is not None:
        if not cfg.left_pole_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.left_pole_material_path}"
        else:
            material_path = cfg.left_pole_material_path
        # create material
        cfg.left_pole_material.func(material_path, cfg.left_pole_material)
        # apply material
        bind_visual_material(str(left_pylon.GetPath()), material_path)
    if cfg.right_pole_material is not None:
        if not cfg.right_pole_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.right_pole_material_path}"
        else:
            material_path = cfg.right_pole_material_path
        # create material
        cfg.right_pole_material.func(material_path, cfg.right_pole_material)
        # apply material
        bind_visual_material(str(right_pylon.GetPath()), material_path)

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)

    return prim
