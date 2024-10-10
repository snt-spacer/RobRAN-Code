# Copyright (c) 2024, Antoine Richard
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd
import numpy as np

from omni.isaac.lab.sim import schemas
from omni.isaac.lab.sim.utils import bind_physics_material, bind_visual_material, clone

if TYPE_CHECKING:
    from . import navigation_shapes_cfg


@clone
def spawn_pin_with_sphere(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinSphereCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    make_pin_with_sphere_prim(prim_path, cfg, translation=translation, orientation=rotation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_pin_with_diamond(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    make_pin_with_diamond_prim(prim_path, cfg, translation=translation, orientation=rotation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_diamond(
    prim_path: str,
    cfg: navigation_shapes_cfg.DiamondCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    make_diamond_prim(prim_path, cfg, translation=translation, orientation=rotation)
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_bicolor_diamond(
    prim_path: str,
    cfg: navigation_shapes_cfg.BiColorDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    make_bicolor_diamond_prim(prim_path, cfg, translation=translation, orientation=rotation)
    return prim_utils.get_prim_at_path(prim_path)


def make_pin_with_sphere_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinSphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
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

    add_material(prim_path, geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_diamond_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.DiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    scale = (cfg.diamond_width, cfg.diamond_width, cfg.diamond_height)
    diamond = make_diamond(
        mesh_prim_path,
        scale=scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props)

    add_material(prim_path, geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_pin_with_diamond_prim(
    prim_path: str,
    cfg: navigation_shapes_cfg.PinDiamondCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
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

    add_material(prim_path, geom_prim_path, mesh_prim_path, cfg)
    return prim


def make_diamond(
    mesh_prim_path: str,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> Usd.Prim:
    """
    Create a diamond shaped mesh prim similar to that of the sims.
    """
    vertices = np.array(
        [[0, 0, 0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.0, 0.5], [0.0, -0.5, 0.5], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]])

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
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)

    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    container = prim_utils.create_prim(mesh_prim_path, "Xform")

    scale = (cfg.diamond_width, cfg.diamond_width, cfg.diamond_height)
    front, back = make_bicolor_diamond(
        mesh_prim_path + "/front",
        mesh_prim_path + "/back",
        scale=scale,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(str(front.GetPath()), cfg.collision_props)
        schemas.define_collision_properties(str(back.GetPath()), cfg.collision_props)

    add_material(prim_path + "/front", geom_prim_path, str(front.GetPath()), cfg)
    add_material(prim_path + "/back", geom_prim_path, str(back.GetPath()), cfg)
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
    """
    vertices_1 = np.array(
        [[0, 0, 0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, -0.5, 0.5], [0.0, 0.0, 1.0]]
    )
    faces_1 = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]])
    vertices_2 = np.array(
        [[0, 0, 0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.0, 0.5], [0.0, -0.5, 0.5], [0.0, 0.0, 1.0]]
    )
    faces_2 = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]])

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


def add_material(
    prim_path: str,
    geom_prim_path: str,
    mesh_prim_path: str,
    cfg: navigation_shapes_cfg.NavigationShapeCfg,
) -> None:
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
