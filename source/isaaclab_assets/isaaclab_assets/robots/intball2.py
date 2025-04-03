# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of JAXA Int-Ball2 robot."""


import math

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
import omni.physx.scripts.utils as physx_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.sim import schemas, schemas_cfg
from omni.isaac.lab.utils import configclass
from pxr import Gf, Sdf


@configclass
class IntBall2Props:
    radius: float = 0.1
    mass: float = 3.3
    CoM: tuple = (0, 0, 0)  # TODO: Waiting for info from author
    refinement: int = 2
    enable_collision: bool = True
    propeller_mass: float = 0.05  # TODO: Waiting for info from author
    num_propellers: int = 8
    propeller_positions: list = []  # TODO: Waiting for exact info from authors
    A: float = 0.035  # Example value
    B: float = 0.045  # Example value
    C: float = 0.035  # Example value

    def __post_init__(self):
        assert self.radius > 0, "Radius must be positive."
        assert self.mass > 0, "Mass must be positive."
        assert len(self.CoM) == 3, "CoM must have 3 coordinates."
        assert self.refinement > 0, "Refinement level must be positive."
        assert isinstance(self.enable_collision, bool), "enable_collision must be a bool."

        self.propeller_positions = [
            (2 * self.A, 2 * self.B, 2 * self.C),
            (2 * self.A, -2 * self.B, 2 * self.C),
            (2 * self.A, -2 * self.B, -2 * self.C),
            (2 * self.A, 2 * self.B, -2 * self.C),
            (-2 * self.A, 2 * self.B, -2 * self.C),
            (-2 * self.A, 2 * self.B, 2 * self.C),
            (-2 * self.A, -2 * self.B, 2 * self.C),
            (-2 * self.A, -2 * self.B, -2 * self.C),
        ]


# The current IntBall2 model assumes the propellers
# to be massless - thus essentially there is no
# effect of propeller polarity. Ideally, the
# propellers should be represented with accurate
# physical properties like inertia and mass, and
# appropriate revolute joints to model propeller
# angular velocities and map it to thrusts.
def generate_intball2(root_path: str, robot_cfg: IntBall2Props):
    stage = stage_utils.get_current_stage()

    # Define articulation and collision properties
    art_props = schemas_cfg.ArticulationRootPropertiesCfg(
        articulation_enabled=True,
        enabled_self_collisions=False,
        fix_root_link=False,
    )
    collider_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=robot_cfg.enable_collision)
    mass_props = schemas_cfg.MassPropertiesCfg(mass=robot_cfg.mass, center_of_mass=robot_cfg.CoM)
    no_mass_props = schemas_cfg.MassPropertiesCfg(mass=0.00001)

    # Create the root of the articulation
    prim_utils.create_prim(root_path)
    body_path = root_path + "/body"
    joint_base_path = root_path + "/joints"
    schemas.define_articulation_root_properties(root_path, art_props)
    print(f"[DEBUG]: Articulation root set at {root_path}")

    body_prim = prim_utils.create_prim(
        prim_path=body_path,
        prim_type="Sphere",
        attributes={"radius": robot_cfg.radius},
    )
    body_prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(robot_cfg.refinement)
    body_prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
    physx_utils.setPhysics(prim=body_prim, kinematic=False)
    schemas.define_mass_properties(body_path, mass_props)
    schemas.define_collision_properties(body_path, collider_props)

    # Add a dummy body with a revolute joint to address joint registration or actuator issues / errors [IsaacLab bug]
    # With this we ensure proper initialization of articulations during simulation
    dummy_body_path = root_path + "/dummy_body"
    dummy_body_prim = prim_utils.create_prim(dummy_body_path)
    physx_utils.setPhysics(prim=dummy_body_prim, kinematic=False)
    schemas.define_mass_properties(dummy_body_path, no_mass_props)
    dummy_joint = schemas.createJoint(
        stage=stage,
        joint_type="Revolute",
        from_prim=body_prim,
        to_prim=dummy_body_prim,
        joint_name="dummy_joint",
        joint_base_path=joint_base_path,
    )
    dummy_joint.GetAttribute("physics:axis").Set("Z")

    # Add propellers
    for i, pos in enumerate(robot_cfg.propeller_positions):
        propeller_path = f"{root_path}/propeller_{i}"
        thruster_prim = prim_utils.create_prim(propeller_path)

        # Set position of the propeller
        thruster_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*pos))

        sqrt3 = 1 / math.sqrt(3)
        thrust_normal = Gf.Vec3f(
            pos[0] / abs(2 * robot_cfg.A) * sqrt3,
            pos[1] / abs(2 * robot_cfg.B) * sqrt3,
            pos[2] / abs(2 * robot_cfg.C) * sqrt3,
        ).GetNormalized()

        # comp. rotation to align propeller radially outward
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*thrust_normal))
        quaternion = rotation.GetQuaternion()

        # Propeller (thruster) prim attributes and physics
        thruster_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd).Set(
            Gf.Quatd(quaternion.GetReal(), *quaternion.GetImaginary())
        )

        physx_utils.setPhysics(prim=thruster_prim, kinematic=False)
        schemas.define_mass_properties(propeller_path, no_mass_props)

        # Add a cylinder geometry to represent the propeller
        geometry_path = f"{propeller_path}/Geometry"
        geometry_prim = prim_utils.create_prim(
            prim_path=geometry_path,
            prim_type="Cylinder",
            attributes={
                "radius": 0.017,  # TODO: Waiting for info from author
                "height": 0.04,  # TODO: Waiting for info from author
            },
        )
        geometry_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3f(0, 0, -0.045)
        )  # Center on the propeller's position

        # print(f"Created propeller: {propeller_path}")
        # print(f"Created joint: fixed_joint_propeller_{i}")

        schemas.createJoint(
            stage=stage,
            joint_type="Fixed",
            from_prim=thruster_prim,
            to_prim=body_prim,
            joint_name=f"fixed_joint_propeller_{i}",
            joint_base_path=joint_base_path,
        )
        # print(f"[DEBUG]: Creating joint fixed_joint_propeller_{i} between {thruster_prim.GetPath()} and {body_prim.GetPath()}")

    # for i in range(8):
    #     print(f"[DEBUG]: Propeller {i+1} Position: {robot_cfg.propeller_positions[i]}")


INTBALL2_CFG = ArticulationCfg(
    spawn=sim_utils.RobotFromCodeCfg(
        robot_gen_func=generate_intball2,
        robot_gen_props=IntBall2Props(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={},
    ),
    actuators={},
)
"""Simple Configuration for JAXA Int-Ball2 robot."""
