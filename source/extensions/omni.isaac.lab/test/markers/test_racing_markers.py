# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

"""Rest everything follows."""

import torch
import unittest

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.simulation_context import SimulationContext

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import GATE_2D_CFG, GATE_3D_CFG, GATE_PYLONS_CFG
from omni.isaac.lab.utils.math import random_orientation
from omni.isaac.lab.utils.timer import Timer


class TestRacingMarkers(unittest.TestCase):
    """Test fixture for the VisualizationMarker class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Simulation time-step
        self.dt = 0.01
        # Open a new stage
        stage_utils.create_new_stage()
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="torch", device="cuda:0")

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # close stage
        stage_utils.close_stage()
        # clear the simulation context
        self.sim.clear_instance()

    def test_racing_marker(self):
        """Test with marker from a USD."""
        # create a marker
        config_gate_3d = GATE_3D_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_gate_2d = GATE_2D_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_gate_pylon = GATE_PYLONS_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        # Try to build the markers
        test_gate_3d = VisualizationMarkers(config_gate_3d)
        test_gate_2d = VisualizationMarkers(config_gate_2d)
        test_gate_pylon = VisualizationMarkers(config_gate_pylon)
        # Try to visualize the markers
        test_gate_3d.visualize(translations=torch.tensor([[0.0, 0.0, 0.0]]))
        test_gate_2d.visualize(translations=torch.tensor([[1.0, 0.0, 0.0]]))
        test_gate_pylon.visualize(translations=torch.tensor([[2.0, 0.0, 0.0]]))

        # play the simulation
        self.sim.reset()
        # create a buffer
        num_frames = 0
        # run for a couple steps to check that everything is working
        for count in range(5):
            self.sim.step()

    def test_racing_marker_color(self):
        """Test with marker from a USD with its color modified."""
        # create a marker
        config_gate_3d = GATE_3D_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_gate_2d = GATE_2D_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_gate_pylon = GATE_PYLONS_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        config_gate_3d.markers["gate_3d"].front_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        config_gate_3d.markers["gate_3d"].back_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        config_gate_3d.markers["gate_3d"].corner_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        config_gate_2d.markers["gate_2d"].front_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        config_gate_2d.markers["gate_2d"].back_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        config_gate_2d.markers["gate_2d"].corner_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        config_gate_pylon.markers["gate_pylons"].left_pole_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)
        )
        config_gate_pylon.markers["gate_pylons"].right_pole_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        # Try to build the markers
        test_gate_3d = VisualizationMarkers(config_gate_3d)
        test_gate_2d = VisualizationMarkers(config_gate_2d)
        test_gate_pylon = VisualizationMarkers(config_gate_pylon)
        # Try to visualize the markers
        test_gate_3d.visualize(translations=torch.tensor([[0.0, 0.0, 0.0]]))
        test_gate_2d.visualize(translations=torch.tensor([[1.0, 0.0, 0.0]]))
        test_gate_pylon.visualize(translations=torch.tensor([[2.0, 0.0, 0.0]]))

        # play the simulation
        self.sim.reset()
        # create a buffer
        num_frames = 0
        # run for a couple steps to check that everything is working
        for count in range(5):
            self.sim.step()

    def test_racing_marker_basic_properties(self):
        """Test with marker from a USD with its base properties."""
        # create a marker
        config_gate_3d = GATE_3D_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_gate_2d = GATE_2D_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_gate_pylon = GATE_PYLONS_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        config_gate_3d.markers["gate_3d"].width = 0.5
        config_gate_3d.markers["gate_3d"].height = 0.5
        config_gate_3d.markers["gate_3d"].depth = 0.05
        config_gate_3d.markers["gate_3d"].thickness = 0.05
        config_gate_2d.markers["gate_2d"].width = 0.5
        config_gate_2d.markers["gate_2d"].height = 0.5
        config_gate_2d.markers["gate_2d"].depth = 0.05
        config_gate_2d.markers["gate_2d"].thickness = 0.05
        config_gate_pylon.markers["gate_pylons"].width = 0.5
        config_gate_pylon.markers["gate_pylons"].radius = 0.05
        config_gate_pylon.markers["gate_pylons"].height = 0.5
        # Try to build the markers
        test_gate_3d = VisualizationMarkers(config_gate_3d)
        test_gate_2d = VisualizationMarkers(config_gate_2d)
        test_gate_pylon = VisualizationMarkers(config_gate_pylon)
        # Try to visualize the markers
        test_gate_3d.visualize(translations=torch.tensor([[0.0, 0.0, 0.0]]))
        test_gate_2d.visualize(translations=torch.tensor([[1.0, 0.0, 0.0]]))
        test_gate_pylon.visualize(translations=torch.tensor([[2.0, 0.0, 0.0]]))

        # play the simulation
        self.sim.reset()
        # create a buffer
        num_frames = 0
        # run for a couple steps to check that everything is working
        for count in range(5):
            self.sim.step()

    def test_racing_marker_collisions(self):
        """Add collisions to markers"""
        # create a marker
        config_gate_3d = GATE_3D_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_gate_2d = GATE_2D_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_gate_pylon = GATE_PYLONS_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        config_gate_3d.markers["gate_3d"].collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        config_gate_2d.markers["gate_2d"].collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        config_gate_pylon.markers["gate_pylons"].collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)


if __name__ == "__main__":
    run_tests()
