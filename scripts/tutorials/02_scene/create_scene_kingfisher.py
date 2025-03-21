# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene_heron.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

torch.set_printoptions(precision=2, sci_mode=False)

import isaaclab.sim as sim_utils
from isaaclab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from isaaclab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from isaaclab_assets import KINGFISHER_CFG  # isort:skip


@configclass
class HeronSceneCfg(InteractiveSceneCfg):
    """Configuration for a heron scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    kingfisher: ArticulationCfg = KINGFISHER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    robot = scene["kingfisher"]

    # Initialize the hydrostatics with the necessary parameters
    hydrostatics_cfg = HydrostaticsCfg()
    hydrostatics_cfg.mass = 35.0  # Kg considering added sensors
    hydrostatics_cfg.width = 1.0  # Kingfisher/Heron width 1.0m in Spec Sheet
    hydrostatics_cfg.length = 1.3  # Kingfisher/Heron length 1.3m in Spec Sheet
    hydrostatics_cfg.waterplane_area = 0.33  # 0.15 width * 1.1 length * 2 hulls
    hydrostatics_cfg.draught_offset = 0.21986  # Distance from base_link to bottom of the hull
    hydrostatics_cfg.max_draught = 0.20  # Kingfisher/Heron draught 120mm in Spec Sheet
    hydrostatics_cfg.average_hydrostatics_force = 275.0
    hydrostatics = Hydrostatics(num_envs=scene.num_envs, device=robot.device, cfg=hydrostatics_cfg)

    # Initialize the hydrodynamics with the necessary parameters
    hydrodynamics_cfg = HydrodynamicsCfg()
    hydrodynamics_cfg.linear_damping = [0.0, 99.99, 99.99, 13.0, 13.0, 5.83]
    hydrodynamics_cfg.linear_damping_rand = [0.0, 0.1, 1.0, 0.0, 0.0, 0.1]
    hydrodynamics_cfg.quadratic_damping = [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    hydrodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
    hydrodynamics = Hydrodynamics(num_envs=scene.num_envs, device=robot.device, cfg=hydrodynamics_cfg)

    # Initialize the thrusters
    thruster_cfg = PropellerActuatorCfg()
    thruster_cfg.cmd_lower_range = -1.0
    thruster_cfg.cmd_upper_range = 1.0
    thruster_cfg.command_rate = 1.0
    thruster_cfg.forces_left = [
        -4.0,  # -1.0
        -4.0,  # -0.9
        -4.0,  # -0.8
        -4.0,  # -0.7
        -2.0,  # -0.6
        -1.0,  # -0.5
        0.0,  # -0.4
        0.0,  # -0.3
        0.0,  # -0.2
        0.0,  # -0.1
        0.0,  # 0.0
        0.0,  # 0.1
        0.0,  # 0.2
        0.5,  # 0.3
        1.5,  # 0.4
        4.75,  # 0.5
        8.25,  # 0.6
        16.0,  # 0.7
        19.5,  # 0.8
        19.5,  # 0.9
        19.5,  # 1.0
    ]
    thruster_cfg.forces_right = thruster_cfg.forces_left
    thrusters = PropellerActuator(num_envs=scene.num_envs, device=robot.device, dt=sim_dt, cfg=thruster_cfg)
    thruster_forces = torch.zeros((scene.num_envs, 1, 6), device=robot.device, dtype=torch.float32)

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 3000 == 0:
            # reset counter
            print("==========")
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()

            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # get robot data
        robot_pos = robot.data.root_pos_w.clone()
        robot_quat = robot.data.root_quat_w.clone()
        robot_vel = robot.data.root_vel_w.clone()

        # get hydrostatics data
        hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(robot_pos, robot_quat)
        hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)

        combined_force = hydrostatic_force + hydrodynamic_force

        force = combined_force[:, :3]
        torque = combined_force[:, 3:]

        # Get the link ids
        base_link_id, _ = robot.find_bodies("base_link")
        left_thruster_id, _ = robot.find_bodies("thruster_left")
        right_thruster_id, _ = robot.find_bodies("thruster_right")

        # Expand forces and toques to match the number of bodies
        # E.g, for a [num_envs, 6] force, expand to [num_envs, len(body_ids), 3]
        force = force.unsqueeze(1).expand(-1, len(base_link_id), -1)
        torque = torque.unsqueeze(1).expand(-1, len(base_link_id), -1)
        robot.set_external_force_and_torque(force, torque, body_ids=base_link_id)

        thrust_cmds = torch.tensor([-1.0, 0.5], dtype=torch.float32, device=robot.device)
        # Expand the thrust commands in the first dimension to match the number of environments.
        thrust_cmds = thrust_cmds.unsqueeze(0).expand(scene.num_envs, -1)

        thrusters.set_target_cmd(thrust_cmds)
        thruster_forces[:, 0, :] = thrusters.update_forces()

        torque = torch.zeros_like(torque)
        left_thruster_force = thruster_forces[..., :3]
        right_thruster_force = thruster_forces[..., 3:]
        if left_thruster_force.any():
            robot.set_external_force_and_torque(left_thruster_force, torque, body_ids=left_thruster_id)
        if right_thruster_force.any():
            robot.set_external_force_and_torque(right_thruster_force, torque, body_ids=right_thruster_id)

        scene.write_data_to_sim()

        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = HeronSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
