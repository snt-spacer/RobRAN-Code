# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple ackermann robot."""


import math

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.physx.scripts.utils as physx_utils
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import schemas, schemas_cfg
from isaaclab.utils import configclass


@configclass
class ModularFreeFlyer2DProps:
    num_thruster_pair: int = 4
    initial_offset: float = math.pi / 4
    shape: str = "Cube"
    radius: float = 0.31
    height: float = 0.5
    width: float = 0.4
    depth: float = 0.4
    mass: float = 5.32
    CoM: tuple = (0, 0, 0)
    refinement: int = 2
    enable_collision: bool = False

    def __post_init__(self):
        assert self.shape in [
            "Cube",
            "Cylinder",
            "Sphere",
            "Asset",
        ], "The shape must be 'cylinder', 'sphere' or 'asset'."
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        assert self.mass > 0, "The mass must be larger than 0."
        assert len(self.CoM) == 3, "The length of the CoM coordinates must be 3."
        assert self.refinement > 0, "The refinement level must be larger than 0."
        assert isinstance(self.enable_collision, bool), "The enable_collision must be a bool."
        self.refinement = int(self.refinement)


def generate_freeflyer(root_path: str, robot_cfg: ModularFreeFlyer2DProps) -> None:
    """
    Builds the platform."""

    art_props = schemas_cfg.ArticulationRootPropertiesCfg(
        articulation_enabled=True,
        enabled_self_collisions=False,
        fix_root_link=False,
    )
    collider_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=robot_cfg.enable_collision)
    mass_props = schemas_cfg.MassPropertiesCfg(mass=robot_cfg.mass, center_of_mass=robot_cfg.CoM)
    no_mass_props = schemas_cfg.MassPropertiesCfg(mass=0.00001)

    # Create the root of the articulation
    stage = stage_utils.get_current_stage()
    prim_utils.create_prim(root_path)
    body_path = root_path + "/body"
    joint_base_path = root_path + "/joints"
    schemas.define_articulation_root_properties(root_path, art_props)

    # Create the main body
    if robot_cfg.shape == "Cube":
        scale = [robot_cfg.width, robot_cfg.depth, robot_cfg.height]
        attributes = {"size": 1.0}
        body_prim = prim_utils.create_prim(
            prim_path=body_path,
            prim_type="Cube",
            attributes=attributes,
            scale=scale,
        )
    elif robot_cfg.shape == "Cylinder":
        attributes = {"height": robot_cfg.height, "radius": robot_cfg.radius}
        body_prim = prim_utils.create_prim(
            prim_path=body_path,
            prim_type="Cylinder",
            attributes=attributes,
        )
    elif robot_cfg.shape == "Sphere":
        attributes = {"radius": robot_cfg.radius}
        body_prim = prim_utils.create_prim(
            prim_path=body_path,
            prim_type="Sphere",
            attributes=attributes,
        )
    body_prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
    body_prim.GetAttribute("refinementLevel").Set(robot_cfg.refinement)
    body_prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool)
    body_prim.GetAttribute("refinementEnableOverride").Set(True)
    physx_utils.setPhysics(prim=body_prim, kinematic=False)
    schemas.define_mass_properties(body_path, mass_props)
    schemas.define_collision_properties(body_path, collider_props)

    # Add thrusters
    for i in range(robot_cfg.num_thruster_pair * 2):
        thruster_path = root_path + "/thruster_" + str(i)
        thruster_prim = prim_utils.create_prim(thruster_path)
        physx_utils.setPhysics(prim=thruster_prim, kinematic=False)
        schemas.define_mass_properties(thruster_path, no_mass_props)
        schemas.createJoint(
            stage=stage,
            joint_type="Fixed",
            from_prim=thruster_prim,
            to_prim=body_prim,
            joint_name="fixed_joint_thruster_" + str(i),
            joint_base_path=joint_base_path,
        )
    # Add a body for the reaction wheel
    reaction_wheel_path = root_path + "/reaction_wheel"
    reaction_wheel_prim = prim_utils.create_prim(reaction_wheel_path)
    physx_utils.setPhysics(prim=reaction_wheel_prim, kinematic=False)
    schemas.define_mass_properties(reaction_wheel_path, no_mass_props)
    schemas.createJoint(
        stage=stage,
        joint_type="Fixed",
        from_prim=reaction_wheel_prim,
        to_prim=body_prim,
        joint_name="fixed_joint_reaction_wheel",
        joint_base_path=joint_base_path,
    )

    # Create three joints to lock the platform in the XY plane. (reduce to 3DoF)
    anchor_path = root_path + "/anchor_body"
    anchor_prim = prim_utils.create_prim(anchor_path)
    physx_utils.setPhysics(prim=anchor_prim, kinematic=False)
    schemas.define_mass_properties(anchor_path, no_mass_props)
    x_lock_path = root_path + "/x_lock_body"
    x_lock_prim = prim_utils.create_prim(x_lock_path)
    physx_utils.setPhysics(prim=x_lock_prim, kinematic=False)
    schemas.define_mass_properties(x_lock_path, no_mass_props)
    y_lock_path = root_path + "/y_lock_body"
    y_lock_prim = prim_utils.create_prim(y_lock_path)
    physx_utils.setPhysics(prim=y_lock_prim, kinematic=False)
    schemas.define_mass_properties(y_lock_path, no_mass_props)
    schemas.createJoint(
        stage=stage,
        joint_type="Fixed",
        from_prim=None,
        to_prim=anchor_prim,
        joint_name="anchor_joint",
        joint_base_path=joint_base_path,
    )
    schemas.createJoint(
        stage=stage,
        joint_type="Prismatic",
        from_prim=anchor_prim,
        to_prim=x_lock_prim,
        joint_name="x_lock_joint",
        joint_base_path=joint_base_path,
    )
    y_joint = schemas.createJoint(
        stage=stage,
        joint_type="Prismatic",
        from_prim=x_lock_prim,
        to_prim=y_lock_prim,
        joint_name="y_lock_joint",
        joint_base_path=joint_base_path,
    )
    y_joint.GetAttribute("physics:axis").Set("Y")
    z_joint = schemas.createJoint(
        stage=stage,
        joint_type="Revolute",
        from_prim=y_lock_prim,
        to_prim=body_prim,
        joint_name="z_lock_joint",
        joint_base_path=joint_base_path,
    )
    z_joint.GetAttribute("physics:axis").Set("Z")


MODULAR_FREEFLYER_2D_CFG = ArticulationCfg(
    spawn=sim_utils.RobotFromCodeCfg(
        robot_gen_func=generate_freeflyer,
        robot_gen_props=ModularFreeFlyer2DProps(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "x_lock_joint": 0.0,
            "y_lock_joint": 0.0,
            "z_lock_joint": 0.0,
        },
    ),
    actuators={},
)
"""Configuration for a simple ackermann robot."""
