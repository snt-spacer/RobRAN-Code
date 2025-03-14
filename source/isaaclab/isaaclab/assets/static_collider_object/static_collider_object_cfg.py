# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024, Antoine Richard.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .static_collider_object import StaticColliderObject


@configclass
class StaticColliderObjectCfg(AssetBaseCfg):
    """Configuration parameters for a static collider object.
    For now, this is a highly simplified version of the RigidObject class.
    It doesn't have any tracking capabilities and can only modify the pose of the objects.
    This will change if we can modify the pose of non-rigid objects in the future."""

    @configclass
    class InitialStateCfg:
        """Initial state of the asset.

        This defines the default initial state of the assets when they are spawned into the simulation.
        Unlike RigidObjets, the default state is not saved. This is because the objects are static and
        do not have any tracking capabilities.
        """

        # root position
        pos: list[list[tuple[float, float, float]]] = MISSING
        """List of list of positions of the objects in simulation world frame. Must be specified"""
        rot: list[list[tuple[float, float, float, float]]] = MISSING
        """List of list of quaternions (w, x, y, z) of the objects in the simulation world frame.
        Must be specified.
        """

    ##
    # Initialize configurations.
    ##

    class_type: type = StaticColliderObject

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose."""
