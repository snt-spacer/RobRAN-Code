# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024, AntoineRichard.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils

from ..asset_base import AssetBase

if TYPE_CHECKING:
    from .static_collider_object_cfg import StaticColliderObjectCfg


class StaticColliderObject(AssetBase):
    """A static collider object asset class.

    This is a light-weight revision of the RigidObject class.
    It is meant to be used for static objects only, that is objects that are not Rigid.
    This class has been stripped of all the RigidObject specific code, and can only modify the pose of objects.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: StaticColliderObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: StaticColliderObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> None:
        """Data instance for the asset."""
        return None

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Nothing to reset for static objects."""
        pass

    def write_data_to_sim(self):
        """Not needed our objects don't move."""
        pass

    def update(self, dt: float):
        pass

    """
    Operations - Write to simulation.
    """

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = root_pose[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)
        self.root_physx_view.set_velocities(
            torch.zeros((root_pose.shape[0], 6), device=self.device), indices=physx_env_ids
        )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        self.cfg.prim_path = f"{self.cfg.prim_path}.*"
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD RigidBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one rigid body in the prim path tree."
            )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create rigid body at: {self.cfg.prim_path}. Please check PhysX logs.")

        # log information about the rigid body
        carb.log_info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        carb.log_info(f"Number of instances: {self.num_instances}")
        carb.log_info(f"Number of bodies: {self.num_bodies}")

        # process configuration
        self._process_cfg()
        self._create_buffers()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        pass

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
