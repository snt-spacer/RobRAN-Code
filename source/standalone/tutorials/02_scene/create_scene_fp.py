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
<<<<<<< HEAD
    robot: Articulation = scene["cartpole"]
=======
    robot: Articulation = scene["floating_platform"]
>>>>>>> 298f15873aaa02f5306cbc3a963eec554f36dd44
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
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
<<<<<<< HEAD
            root_state[:, 2] = 2.0
=======
            root_state[:, 2] = 1.0
>>>>>>> 298f15873aaa02f5306cbc3a963eec554f36dd44
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            bodies = robot.data.body_names
            print(bodies)
            joints = robot.data.joint_names
            print(joints)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
            body_id = [
                robot.find_bodies(f"thruster_{i}")[0][0] for i in range(1, 9)
            ]  # [0] for the indexes and [1] for the names
            print(f"[INFO]: Body ID: {body_id}")

        # print(robot.data.root_pos_w)
        # Apply random action
<<<<<<< HEAD
        # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
=======

        # -- generate forces and torques
        thrusts = torch.zeros((scene.num_envs, 8, 3), device=robot.device)
        thrusts[:, ::2] = 0.1
        torques = torch.zeros((scene.num_envs, 8, 3), device=robot.device)
        # -- apply action to the robot
        print(f"Thrusts: {thrusts}")
        robot.set_external_force_and_torque(thrusts, torques, body_ids=body_id, env_ids=None)

        print(robot.data.body_state_w[0, 3, :3])

        # robot.set_external_force_and_torque(torch.tensor([[0,0,0]]), torch.tensor([[0,0,0]]), body_ids=[0], env_ids=None)

>>>>>>> 298f15873aaa02f5306cbc3a963eec554f36dd44
        # robot.set_joint_effort_target(efforts)
        # -- write data to sim
        # robot.set_external_force_and_torque()

        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
