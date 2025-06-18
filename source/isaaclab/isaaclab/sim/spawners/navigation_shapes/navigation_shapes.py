# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_visual_material, clone

if TYPE_CHECKING:
    from . import navigation_shapes_cfg


@clone
def spawn_pin_with_sphere(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinSphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with a sphere on top using the given attributes.

    It uses a combination of cylinder and a sphere to create a shape akin to a lolipop.
    This is meant to be used as a navigation helper for position goals: The slender pin body allows to
    visually evaluate the accuracy of the policy while the sphere on top is used to quickly spot it
    in the scene.

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
    make_pin_with_sphere_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_pin_with_diamond(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with a diamond on top using the given attributes.

    It uses a combination of cylinder and a custom mesh to create a shape where a crystal looking object
    stands on top of a pin. This is meant to be used as a navigation helper. The slender pin body allows to visually
    evaluate the accuracy of the policy while the crystal on top is used to quickly spot it in the scene.

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
    make_pin_with_diamond_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_diamond(
    prim_path: str,
    cfg: navigation_shapes_cfg.DiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based diamond prim using the given attributes.

    It uses a custom mesh to create a diamond shape. This is meant to be used as a navigation helper. It is meant to
    be attached to a robot to better visualize it's position relatively to a goal.

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
    make_diamond_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_bicolor_diamond(
    prim_path: str,
    cfg: navigation_shapes_cfg.BiColorDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based bicolor diamond prim using the given attributes.

    It uses a custom mesh to create a diamond shape. This is meant to be used as a navigation helper.
    The two colors can be used to easily distinguish between the front and back of the diamond. It is meant to
    be attached to a robot to better visualize it's position and orientation relatively to a goal.

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
    make_bicolor_diamond_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_arrow(
    prim_path: str,
    cfg: navigation_shapes_cfg.ArrowCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based arrow using the given attributes.

    It uses a combination of cylinder and a cone to create an arrow shape.
    This is meant to be attached to a robot to better visualize it's orientation in the scene.

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
    make_arrow_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_position_marker_3d(
    prim_path: str,
    cfg: navigation_shapes_cfg.PositionMarker3DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 3D position marker using the given attributes.

    It uses a combination of cylinder and spheres to create an non oriented 3D position marker. This is meant
    to be used in 3D navigation tasks to visualize the position of a goal.

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
    make_3d_position_marker_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_pose_marker_3d(
    prim_path: str,
    cfg: navigation_shapes_cfg.PoseMarker3DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 3D pose marker using the given attributes.

    It uses a combination of cylinders and cones to create an oriented 3D pose marker. This is meant
    to be used in 3D navigation tasks to visualize the position and orientation of a goal.

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
    make_3d_pose_marker_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


"""
Helper functions.
"""


@clone
def spawn_pin_with_arrow(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinArrowCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with an arrow on top using the given attributes.

    It uses a combination of cylinders and a cone to create a shape where an arrow stands on top of a pin.
    This is meant to be used as a navigation helper for pose goals: The slender pin body allows to
    visually evaluate the accuracy of the policy while the sphere on top is used to quickly spot it
    in the scene.

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
    make_pin_with_arrow_prim(prim_path, cfg, translation=translation, orientation=orientation)
    return prim_utils.get_prim_at_path(prim_path)


def make_pin_with_sphere_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinSphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with a sphere on top using the given attributes.

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

    pin_attributes = {"radius": cfg.pin_radius, "height": cfg.pin_length, "axis": "Z"}
    pin_translation = (0, 0, cfg.pin_length / 2.0)
    pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("pin_body")),
        "Cylinder",
        position=pin_translation,
        attributes=pin_attributes,
    )
    sphere_attributes = {"radius": cfg.sphere_radius}
    sphere_translation = (0, 0, cfg.pin_length)
    sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("pin_head")),
        "Sphere",
        position=sphere_translation,
        attributes=sphere_attributes,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(sphere.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(pin.GetPath()), cfg.collision_props)

    add_material(geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_diamond_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.DiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based diamond using the given attributes.

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

    scale = (cfg.diamond_width, cfg.diamond_width, cfg.diamond_height)
    make_diamond(
        mesh_prim_path,
        scale=scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props)

    add_material(geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_pin_with_diamond_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with an diamond on top using the given attributes.

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

    pin_length = cfg.pin_length + cfg.diamond_height / 2.0
    pin_attributes = {"radius": cfg.pin_radius, "height": pin_length, "axis": "Z"}
    pin_translation = (0, 0, pin_length / 2.0)
    pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("pin_body")),
        "Cylinder",
        position=pin_translation,
        attributes=pin_attributes,
    )
    scale = (cfg.diamond_width, cfg.diamond_width, cfg.diamond_height)
    diamond = make_diamond(
        str(container.GetPath().AppendChild("diamond")),
        translation=(0, 0, cfg.pin_length),
        scale=scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(diamond.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(pin.GetPath()), cfg.collision_props)

    add_material(geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_diamond(
    mesh_prim_path: str,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> Usd.Prim:
    """
    Create a diamond shaped mesh prim similar to that of the sims.

    Args:
        front_mesh_prim_path: The prim path to spawn the front mesh at.
        back_mesh_prim_path: The prim path to spawn the back mesh at.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to (1, 1, 1).

    Returns:
        The created root prims.

    Raises:
        ValueError: If a prim already exists at the given paths.
    """
    vertices = np.array([
        [0, 0, 0],  # Bottom (0)
        [0.4330125, 0.25, 0.5],  # Hexagon Vertex 1 (1)
        [0.0, 0.5, 0.5],  # Hexagon Vertex 2 (2)
        [-0.4330125, 0.25, 0.5],  # Hexagon Vertex 3 (3)
        [-0.4330125, -0.25, 0.5],  # Hexagon Vertex 4 (4)
        [0.0, -0.5, 0.5],  # Hexagon Vertex 5 (5)
        [0.4330125, -0.25, 0.5],  # Hexagon Vertex 6 (6)
        [0.0, 0.0, 1.0],  # Top (7)
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],
        [7, 1, 2],
        [7, 2, 3],
        [7, 3, 4],
        [7, 4, 5],
        [7, 5, 6],
        [7, 6, 1],
    ])

    mesh_prim = prim_utils.create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices,
            "faceVertexIndices": faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(faces)),
            "subdivisionScheme": "bilinear",
        },
        translation=translation,
        orientation=orientation,
        scale=scale,
    )
    return mesh_prim


def make_bicolor_diamond_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.BiColorDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with an diamond on top using the given attributes.

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

    prim_utils.create_prim(mesh_prim_path, "Xform")

    scale = (cfg.diamond_width, cfg.diamond_width, cfg.diamond_height)
    front, back = make_bicolor_diamond(
        mesh_prim_path + "/front",
        mesh_prim_path + "/back",
        scale=scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(front.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back.GetPath()), cfg.collision_props)

    add_material(geom_prim_path, str(front.GetPath()), cfg)
    add_material(geom_prim_path, str(back.GetPath()), cfg)
    if cfg.front_material is not None:
        if not cfg.front_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.front_material_path}"
        else:
            material_path = cfg.front_material_path
        # create material
        cfg.front_material.func(material_path, cfg.front_material)
        # apply material
        bind_visual_material(str(front.GetPath()), material_path)
    if cfg.back_material is not None:
        if not cfg.back_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.back_material_path}"
        else:
            material_path = cfg.back_material_path
        # create material
        cfg.back_material.func(material_path, cfg.back_material)
        # apply material
        bind_visual_material(str(back.GetPath()), material_path)
    return prim


def make_bicolor_diamond(
    front_mesh_prim_path: str,
    back_mesh_prim_path: str,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> tuple[Usd.Prim, Usd.Prim]:
    """
    Create a diamond shaped mesh prim similar to that of the sims.
    This shape is made out of two meshes, one for the front and one for the back. This is useful to apply different
    materials to the front and back of the diamond.

    Args:
        front_mesh_prim_path: The prim path to spawn the front mesh at.
        back_mesh_prim_path: The prim path to spawn the back mesh at.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to (1, 1, 1).

    Returns:
        The created root prims.

    Raises:
        ValueError: If a prim already exists at the given paths.
    """
    vertices_1 = np.array([
        [0, 0, 0],  # Bottom (0)
        [0.4330125, 0.25, 0.5],  # Hexagon Vertex 1 (1)
        [0.0, 0.5, 0.5],  # Hexagon Vertex 2 (2)
        [0.0, 0.25, 0.5],  # Hexagon Vertex 3 (3)
        [0.0, -0.25, 0.5],  # Hexagon Vertex 4 (4)
        [0.0, -0.5, 0.5],  # Hexagon Vertex 5 (5)
        [0.4330125, -0.25, 0.5],  # Hexagon Vertex 6 (6)
        [0.0, 0.0, 1.0],  # Top (7)
    ])
    faces_1 = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],
        [7, 1, 2],
        [7, 2, 3],
        [7, 3, 4],
        [7, 4, 5],
        [7, 5, 6],
        [7, 6, 1],
    ])
    vertices_2 = np.array([
        [0, 0, 0],  # Bottom (0)
        [0.0, 0.25, 0.5],  # Hexagon Vertex 1 (1)
        [0.0, 0.5, 0.5],  # Hexagon Vertex 2 (2)
        [-0.4330125, 0.25, 0.5],  # Hexagon Vertex 3 (3)
        [-0.4330125, -0.25, 0.5],  # Hexagon Vertex 4 (4)
        [0.0, -0.5, 0.5],  # Hexagon Vertex 5 (5)
        [0.0, -0.25, 0.5],  # Hexagon Vertex 6 (6)
        [0.0, 0.0, 1.0],  # Top (7)
    ])
    faces_2 = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],
        [7, 1, 2],
        [7, 2, 3],
        [7, 3, 4],
        [7, 4, 5],
        [7, 5, 6],
        [7, 6, 1],
    ])
    mesh_front = prim_utils.create_prim(
        front_mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices_1,
            "faceVertexIndices": faces_1.flatten(),
            "faceVertexCounts": np.asarray([3] * len(faces_1)),
            "subdivisionScheme": "bilinear",
        },
        translation=translation,
        orientation=orientation,
        scale=scale,
    )
    mesh_back = prim_utils.create_prim(
        back_mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices_2,
            "faceVertexIndices": faces_2.flatten(),
            "faceVertexCounts": np.asarray([3] * len(faces_2)),
            "subdivisionScheme": "bilinear",
        },
        translation=translation,
        orientation=orientation,
        scale=scale,
    )
    return mesh_front, mesh_back


def make_arrow_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.ArrowCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based arrow using the given attributes.

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

    arrow_body_attributes = {
        "radius": cfg.arrow_body_radius,
        "height": cfg.arrow_body_length,
        "axis": "X",
    }
    arrow_body_translation = (cfg.arrow_body_length / 2.0, 0, 0)
    arrow_body = prim_utils.create_prim(
        str(container.GetPath().AppendChild("arrow_body")),
        "Cylinder",
        position=arrow_body_translation,
        attributes=arrow_body_attributes,
    )

    arrow_head_attributes = {
        "radius": cfg.arrow_head_radius,
        "height": cfg.arrow_head_length,
        "axis": "X",
    }
    arrow_head_translation = (cfg.arrow_body_length + cfg.arrow_head_length / 2.0, 0, 0)
    arrow_head = prim_utils.create_prim(
        str(container.GetPath().AppendChild("arrow_head")),
        "Cone",
        position=arrow_head_translation,
        attributes=arrow_head_attributes,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(arrow_body.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(arrow_head.GetPath()), cfg.collision_props)

    add_material(geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_pin_with_arrow_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinArrowCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based pin prim with an arrow on top using the given attributes.

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

    pin_attributes = {"radius": cfg.pin_radius, "height": cfg.pin_length, "axis": "Z"}
    pin_translation = (0, 0, cfg.pin_length / 2.0)
    pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("pin_body")),
        "Cylinder",
        position=pin_translation,
        attributes=pin_attributes,
    )

    arrow_body_attributes = {
        "radius": cfg.arrow_body_radius,
        "height": cfg.arrow_body_length,
        "axis": "X",
    }
    arrow_body_translation = (cfg.arrow_body_length / 2.0, 0, cfg.pin_length)
    arrow_body = prim_utils.create_prim(
        str(container.GetPath().AppendChild("arrow_body")),
        "Cylinder",
        position=arrow_body_translation,
        attributes=arrow_body_attributes,
    )

    arrow_head_attributes = {
        "radius": cfg.arrow_head_radius,
        "height": cfg.arrow_head_length,
        "axis": "X",
    }
    arrow_head_translation = (
        cfg.arrow_body_length + cfg.arrow_head_length / 2.0,
        0,
        cfg.pin_length,
    )
    arrow_head = prim_utils.create_prim(
        str(container.GetPath().AppendChild("arrow_head")),
        "Cone",
        position=arrow_head_translation,
        attributes=arrow_head_attributes,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(arrow_body.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(arrow_head.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(pin.GetPath()), cfg.collision_props)

    add_material(geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_3d_position_marker_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PositionMarker3DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 3D position marker using the given attributes.

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

    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    container = prim_utils.create_prim(mesh_prim_path, "Xform")

    x_pin_attributes = {"radius": cfg.pin_radius, "height": cfg.pin_length, "axis": "X"}
    y_pin_attributes = {"radius": cfg.pin_radius, "height": cfg.pin_length, "axis": "Y"}
    z_pin_attributes = {"radius": cfg.pin_radius, "height": cfg.pin_length, "axis": "Z"}
    sphere_attributes = {"radius": cfg.sphere_radius}
    x_pos_sphere_translation = (cfg.pin_length / 2.0, 0, 0)
    x_neg_sphere_translation = (-cfg.pin_length / 2.0, 0, 0)
    y_pos_sphere_translation = (0, cfg.pin_length / 2.0, 0)
    y_neg_sphere_translation = (0, -cfg.pin_length / 2.0, 0)
    z_pos_sphere_translation = (0, 0, cfg.pin_length / 2.0)
    z_neg_sphere_translation = (0, 0, -cfg.pin_length / 2.0)

    x_pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("x_pin")),
        "Cylinder",
        position=(0, 0, 0),
        attributes=x_pin_attributes,
    )
    y_pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("y_pin")),
        "Cylinder",
        position=(0, 0, 0),
        attributes=y_pin_attributes,
    )
    z_pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("z_pin")),
        "Cylinder",
        position=(0, 0, 0),
        attributes=z_pin_attributes,
    )
    x_pos_sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("x_pos_sphere")),
        "Sphere",
        position=x_pos_sphere_translation,
        attributes=sphere_attributes,
    )
    x_neg_sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("x_neg_sphere")),
        "Sphere",
        position=x_neg_sphere_translation,
        attributes=sphere_attributes,
    )
    y_pos_sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("y_pos_sphere")),
        "Sphere",
        position=y_pos_sphere_translation,
        attributes=sphere_attributes,
    )
    y_neg_sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("y_neg_sphere")),
        "Sphere",
        position=y_neg_sphere_translation,
        attributes=sphere_attributes,
    )
    z_pos_sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("z_pos_sphere")),
        "Sphere",
        position=z_pos_sphere_translation,
        attributes=sphere_attributes,
    )
    z_neg_sphere = prim_utils.create_prim(
        str(container.GetPath().AppendChild("z_neg_sphere")),
        "Sphere",
        position=z_neg_sphere_translation,
        attributes=sphere_attributes,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(x_pin.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(y_pin.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(z_pin.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(x_pos_sphere.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(x_neg_sphere.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(y_pos_sphere.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(y_neg_sphere.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(z_pos_sphere.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(z_neg_sphere.GetPath()), cfg.collision_props)

    if cfg.x_material is not None:
        if not cfg.x_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.x_material_path}"
        else:
            material_path = cfg.x_material_path
        # create material
        cfg.x_material.func(material_path, cfg.x_material)
        # apply material
        bind_visual_material(str(x_pin.GetPath()), material_path)
        bind_visual_material(str(x_pos_sphere.GetPath()), material_path)
        bind_visual_material(str(x_neg_sphere.GetPath()), material_path)
    if cfg.y_material is not None:
        if not cfg.y_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.y_material_path}"
        else:
            material_path = cfg.y_material_path
        # create material
        cfg.y_material.func(material_path, cfg.y_material)
        # apply material
        bind_visual_material(str(y_pin.GetPath()), material_path)
        bind_visual_material(str(y_pos_sphere.GetPath()), material_path)
        bind_visual_material(str(y_neg_sphere.GetPath()), material_path)
    if cfg.z_material is not None:
        if not cfg.z_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.z_material_path}"
        else:
            material_path = cfg.z_material_path
        # create material
        cfg.z_material.func(material_path, cfg.z_material)
        # apply material
        bind_visual_material(str(z_pin.GetPath()), material_path)
        bind_visual_material(str(z_pos_sphere.GetPath()), material_path)
        bind_visual_material(str(z_neg_sphere.GetPath()), material_path)


def make_3d_pose_marker_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PoseMarker3DCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USDGeom-based 3D pose marker prim using the given attributes.

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

    prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    container = prim_utils.create_prim(mesh_prim_path, "Xform")

    x_pin_attributes = {
        "radius": cfg.arrow_body_radius,
        "height": cfg.arrow_body_length,
        "axis": "X",
    }
    y_pin_attributes = {
        "radius": cfg.arrow_body_radius,
        "height": cfg.arrow_body_length,
        "axis": "Y",
    }
    z_pin_attributes = {
        "radius": cfg.arrow_body_radius,
        "height": cfg.arrow_body_length,
        "axis": "Z",
    }
    x_head_attributes = {
        "radius": cfg.arrow_head_radius,
        "height": cfg.arrow_head_length,
        "axis": "X",
    }
    y_head_attributes = {
        "radius": cfg.arrow_head_radius,
        "height": cfg.arrow_head_length,
        "axis": "Y",
    }
    z_head_attributes = {
        "radius": cfg.arrow_head_radius,
        "height": cfg.arrow_head_length,
        "axis": "Z",
    }

    x_body_translation = (cfg.arrow_body_length / 2.0, 0, 0)
    x_head_translation = (cfg.arrow_body_length + cfg.arrow_head_length / 2.0, 0, 0)
    y_body_translation = (0, cfg.arrow_body_length / 2.0, 0)
    y_head_translation = (0, cfg.arrow_body_length + cfg.arrow_head_length / 2.0, 0)
    z_body_translation = (0, 0, cfg.arrow_body_length / 2.0)
    z_head_translation = (0, 0, cfg.arrow_body_length + cfg.arrow_head_length / 2.0)

    x_pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("x_pin")),
        "Cylinder",
        position=x_body_translation,
        attributes=x_pin_attributes,
    )
    y_pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("y_pin")),
        "Cylinder",
        position=y_body_translation,
        attributes=y_pin_attributes,
    )
    z_pin = prim_utils.create_prim(
        str(container.GetPath().AppendChild("z_pin")),
        "Cylinder",
        position=z_body_translation,
        attributes=z_pin_attributes,
    )
    x_head = prim_utils.create_prim(
        str(container.GetPath().AppendChild("x_head")),
        "Cone",
        position=x_head_translation,
        attributes=x_head_attributes,
    )
    y_head = prim_utils.create_prim(
        str(container.GetPath().AppendChild("y_head")),
        "Cone",
        position=y_head_translation,
        attributes=y_head_attributes,
    )
    z_head = prim_utils.create_prim(
        str(container.GetPath().AppendChild("z_head")),
        "Cone",
        position=z_head_translation,
        attributes=z_head_attributes,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(x_pin.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(y_pin.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(z_pin.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(x_head.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(y_head.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(z_head.GetPath()), cfg.collision_props)

    if cfg.x_material is not None:
        if not cfg.x_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.x_material_path}"
        else:
            material_path = cfg.x_material_path
        # create material
        cfg.x_material.func(material_path, cfg.x_material)
        # apply material
        bind_visual_material(str(x_pin.GetPath()), material_path)
        bind_visual_material(str(x_head.GetPath()), material_path)
    if cfg.y_material is not None:
        if not cfg.y_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.y_material_path}"
        else:
            material_path = cfg.y_material_path
        # create material
        cfg.y_material.func(material_path, cfg.y_material)
        # apply material
        bind_visual_material(str(y_pin.GetPath()), material_path)
        bind_visual_material(str(y_head.GetPath()), material_path)
    if cfg.z_material is not None:
        if not cfg.z_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.z_material_path}"
        else:
            material_path = cfg.z_material_path
        # create material
        cfg.z_material.func(material_path, cfg.z_material)
        # apply material
        bind_visual_material(str(z_pin.GetPath()), material_path)
        bind_visual_material(str(z_head.GetPath()), material_path)


def add_material(
    geom_prim_path: str,
    mesh_prim_path: str,
    cfg: navigation_shapes_cfg.NavigationShapeCfg,
) -> None:
    """Adds material properties to the given prim.

    Args:
        geom_prim_path: The path to the geometry prim.
        mesh_prim_path: The path to the mesh prim.
        cfg: The configuration instance.
    """
    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(mesh_prim_path, material_path)
