# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FLOATING_PLATFORM_CFG  # isort:skip


@configclass
class FloatingPlatformSceneCfg(InteractiveSceneCfg):
    """Configuration for a floating-platform scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    floating_platform: ArticulationCfg = FLOATING_PLATFORM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot: Articulation = scene["floating_platform"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Retrieve body ID for thrusters
    body_id = [
        robot.find_bodies(f"thruster_{i}")[0][0] for i in range(1, 9)
    ]  # [0] for the indexes and [1] for the names
    #
    # Define thrusts to test which thruster is pushing on which direction
    thrusts = torch.zeros((scene.num_envs, 8, 3), device=robot.device)
    test_thruster = 0  # Test thruster 0,..,7
    thrusts[:, test_thruster, 2] = 1.0
    thrusts[:, 1, 2] = 1.0
    torques = torch.zeros((scene.num_envs, 8, 3), device=robot.device)

    print(f"[INFO]: Body ID: {body_id}")
    print(f"Thrusts: {thrusts}")

    # Simulation loop
    while simulation_app.is_running():

        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] = 1.0
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            bodies = robot.data.body_names
            # --print(bodies)
            joints = robot.data.joint_names
            print(joints)
            joint_pos[:].zero_()
            joint_vel[:].zero_()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # print(robot.data.root_pos_w)
        # Print position of the robot to check if it is moving in the right direction
        print(f"Position: {robot.data.body_state_w[0, 3, :3]}")

        # -- apply action to the robot
        robot.set_external_force_and_torque(thrusts, torques, body_ids=body_id, env_ids=None)
        # robot.set_external_force_and_torque(torch.tensor([[0,0,0]]), torch.tensor([[0,0,0]]), body_ids=[0], env_ids=None)

        # robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def test_thruster_directions(sim, robot, scene, body_id):
    """Test which thrusters move the platform in specific directions."""
    num_thrusters = 8
    directions = {}

    for i in range(num_thrusters):
        # Reset the platform
        scene.reset()

        # Apply thrust to the current thruster
        thrusts = torch.zeros((scene.num_envs, num_thrusters, 3), device=robot.device)
        thrusts[:, i, 2] = 1.0  # Activate thruster along the z-axis
        torques = torch.zeros_like(thrusts)

        # Apply thrust and simulate for a few steps
        robot.set_external_force_and_torque(thrusts, torques, body_ids=body_id, env_ids=None)
        for _ in range(1000):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        # Check the resulting motion
        displacement = robot.data.body_state_w[0, 3, :2]  # x, y position of the platform
        if displacement.norm() > 1e-3:  # If there is noticeable motion
            directions[i + 1] = displacement.cpu().numpy()
        # reset the scene entities and position
        scene.reset()

    print("[INFO]: Thruster direction mapping:")
    for thruster, direction in directions.items():
        print(f"Thruster {thruster}: Direction {direction}")

    return directions


def calculate_system_energy_and_momentum(robot, platform_mass, max_thrust, radius):
    """Calculate energy and momentum of the system."""
    # Extract current velocities
    linear_velocity = robot.data.body_state_w[0, 7, :3]  # Linear velocity
    angular_velocity = robot.data.body_state_w[0, 10, :3]  # Angular velocity

    # Kinetic energy
    kinetic_energy = 0.5 * platform_mass * (linear_velocity.norm() ** 2)

    # Potential energy (assuming z=0 is the ground level)
    g = 9.81  # Gravitational acceleration
    height = robot.data.body_state_w[0, 3, 2]
    potential_energy = platform_mass * g * height

    # Total energy
    total_energy = kinetic_energy + potential_energy

    # Calculate momentum generated by thrusters
    forces = torch.zeros((8, 3), device=robot.device)
    for i in range(8):
        forces[i, 2] = max_thrust if i < 4 else -max_thrust  # Adjust thrust polarity for opposing thrusters
    momenta = forces[:, :2].norm(dim=1) * radius  # Momentum = Force * Distance (radius)
    total_momentum = momenta.sum().item()

    print("[INFO]: Energy and Momentum calculations:")
    print(f"Kinetic Energy: {kinetic_energy.item():.3f} J")
    print(f"Potential Energy: {potential_energy.item():.3f} J")
    print(f"Total Energy: {total_energy.item():.3f} J")
    print(f"Total Momentum: {total_momentum:.3f} Nm")

    return {
        "kinetic_energy": kinetic_energy.item(),
        "potential_energy": potential_energy.item(),
        "total_energy": total_energy.item(),
        "total_momentum": total_momentum,
    }


def test_thruster_pairs(sim, robot, scene, body_id):
    """
    Test thruster alignment by activating pairs of thrusters.

    Pairs of opposite thrusters (180° apart) should generate motion along their shared axis.
    Pairs of adjacent thrusters (90° apart) with opposite signs should generate diagonal motion.

    Args:
        robot: Articulation object representing the floating platform.
        scene: InteractiveScene object for simulation.
        body_id: List of body IDs for thrusters.
        radius: Radius of the platform.

    Returns:
        Dictionary mapping thruster pairs to observed directions.
    """
    num_thrusters = 8
    directions = {}

    # Reset the platform to neutral state before testing
    def reset_platform():
        scene.reset()
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        root_state[:, 2] = 1.0
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

    for i in range(num_thrusters):
        # Opposite pair test (e.g., 1 and 5, 2 and 6, etc.)
        opposite_thruster = (i + 4) % num_thrusters

        # Activate both thrusters in the pair
        reset_platform()
        thrusts = torch.zeros((scene.num_envs, num_thrusters, 3), device=robot.device)
        thrusts[:, i, 2] = 1.0  # Activate current thruster
        thrusts[:, opposite_thruster, 2] = 1.0  # Activate opposite thruster
        torques = torch.zeros_like(thrusts)

        # Apply forces and simulate
        robot.set_external_force_and_torque(thrusts, torques, body_ids=body_id, env_ids=None)
        for _ in range(1000):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        # Record motion
        displacement = robot.data.body_state_w[0, 3, :2]  # x, y position
        directions[f"{i + 1} & {opposite_thruster + 1}"] = displacement.cpu().numpy()

    # Repeat for adjacent pairs
    for i in range(num_thrusters):
        adjacent_thruster = (i + 1) % num_thrusters

        # Activate one thruster positively and the adjacent one negatively
        reset_platform()
        thrusts = torch.zeros((scene.num_envs, num_thrusters, 3), device=robot.device)
        thrusts[:, i, 2] = 1.0  # Positive thrust
        thrusts[:, adjacent_thruster, 2] = -1.0  # Negative thrust
        torques = torch.zeros_like(thrusts)

        # Apply forces and simulate
        robot.set_external_force_and_torque(thrusts, torques, body_ids=body_id, env_ids=None)
        for _ in range(1000):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        # Record motion
        displacement = robot.data.body_state_w[0, 3, :2]  # x, y position
        directions[f"{i + 1} & {adjacent_thruster + 1} (opposite signs)"] = displacement.cpu().numpy()

    print("[INFO]: Pairwise Thruster Direction Mapping:")
    for pair, direction in directions.items():
        print(f"Thruster Pair {pair}: Direction {direction}")

    return directions


def test_simple(sim, robot, scene, body_id):
    "given 2 thrusters, test if the platform moves in the right direction"
    # Reset the platform
    scene.reset()
    # Apply thrust to the current thruster
    thrusts = torch.zeros((scene.num_envs, 8, 3), device=robot.device)
    t_id_first = 7
    t_id_second = 8
    thrusts[:, t_id_first - 1, 2] = 1.0  # Activate thruster along the z-axis
    thrusts[:, t_id_second - 1, 2] = 1.0  # Activate thruster along the z-axis
    torques = torch.zeros_like(thrusts)
    # Apply thrust and simulate for a few steps
    robot.set_external_force_and_torque(thrusts, torques, body_ids=body_id, env_ids=None)
    for _ in range(1000):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    # Check the resulting motion
    displacement = robot.data.body_state_w[0, 3, :2]  # x, y position of the platform
    print(f"Position: {robot.data.body_state_w[0, 3, :3]}")
    if displacement.norm() > 1e-3:  # If there is noticeable motion
        print(f"Platform moved in the right direction: {displacement}")
    # reset the scene entities and position
    scene.reset()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = FloatingPlatformSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

    robot: Articulation = scene["floating_platform"]
    body_id = [robot.find_bodies(f"thruster_{i}")[0][0] for i in range(1, 9)]
    radius = 0.31  # Distance of thrusters from the center of mass (m)
    platform_mass = 5.0  # Mass of the platform (kg)
    max_thrust = 1.0  # Maximum thrust of the thrusters (N)

    # test_thruster_directions(sim, robot, scene, body_id)
    # calculate_system_energy_and_momentum(robot, platform_mass, max_thrust, radius)
    # test_thruster_pairs(sim, robot, scene, body_id)
    # test_simple(sim, robot, scene, body_id)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
