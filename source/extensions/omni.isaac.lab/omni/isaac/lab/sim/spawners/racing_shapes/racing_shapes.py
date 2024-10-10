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
    from . import racing_shapes_cfg


@clone
def spawn_gate_3d(
    prim_path: str,
    cfg: racing_shapes_cfg.Gate3DCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    raise NotImplementedError("spawn_gate_3d is not implemented yet")


@clone
def spawn_gate_2d(
    prim_path: str,
    cfg: racing_shapes_cfg.Gate2DCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    raise NotImplementedError("spawn_gate_2d is not implemented yet")


@clone
def spawn_gate_pylons(
    prim_path: str,
    cfg: racing_shapes_cfg.GatePylonsCfg,
    translation: tuple[float, float, float] | None = None,
    rotation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    raise NotImplementedError("spawn_pylon is not implemented yet")
