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
from omni.isaac.lab.markers.config import (
    PIN_ARROW_CFG,
    PIN_DIAMOND_CFG,
    PIN_SPHERE_CFG,
    ARROW_CFG,
    DIAMOND_CFG,
    BICOLOR_DIAMOND_CFG,
    POSITION_MARKER_3D_CFG,
    POSE_MARKER_3D_CFG,
)
from omni.isaac.lab.utils.math import random_orientation
from omni.isaac.lab.utils.timer import Timer


class TestNavigationMarkers(unittest.TestCase):
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

    def test_nav_marker(self):
        """Test with marker from a USD."""
        # create a marker
        config_pin_sphere = PIN_SPHERE_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_pin_arrow = PIN_ARROW_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_pin_diamond = PIN_DIAMOND_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        config_arrow = ARROW_CFG.replace(prim_path="/World/Visuals/test_arrow")
        config_diamond = DIAMOND_CFG.replace(prim_path="/World/Visuals/test_diamond")
        config_bicolor_diamond = BICOLOR_DIAMOND_CFG.replace(prim_path="/World/Visuals/test_bicolor_diamond")
        config_position_marker_3d = POSITION_MARKER_3D_CFG.replace(prim_path="/World/Visuals/test_position_marker_3d")
        config_pose_marker_3d = POSE_MARKER_3D_CFG.replace(prim_path="/World/Visuals/test_pose_marker_3d")
        # Try to build the markers
        test_pin_sphere = VisualizationMarkers(config_pin_sphere)
        test_pin_arrow = VisualizationMarkers(config_pin_arrow)
        test_pin_diamond = VisualizationMarkers(config_pin_diamond)
        test_arrow = VisualizationMarkers(config_arrow)
        test_diamond = VisualizationMarkers(config_diamond)
        test_bicolor_diamond = VisualizationMarkers(config_bicolor_diamond)
        test_position_marker_3d = VisualizationMarkers(config_position_marker_3d)
        test_pose_marker_3d = VisualizationMarkers(config_pose_marker_3d)
        # Try to visualize the markers
        test_pin_sphere.visualize(translations=torch.tensor([[0.0, 0.0, 0.0]]))
        test_pin_arrow.visualize(translations=torch.tensor([[1.0, 0.0, 0.0]]))
        test_pin_diamond.visualize(translations=torch.tensor([[2.0, 0.0, 0.0]]))
        test_arrow.visualize(translations=torch.tensor([[3.0, 0.0, 0.0]]))
        test_diamond.visualize(translations=torch.tensor([[4.0, 0.0, 0.0]]))
        test_bicolor_diamond.visualize(translations=torch.tensor([[5.0, 0.0, 0.0]]))
        test_position_marker_3d.visualize(translations=torch.tensor([[6.0, 0.0, 0.0]]))
        test_pose_marker_3d.visualize(translations=torch.tensor([[7.0, 0.0, 0.0]]))

        # play the simulation
        self.sim.reset()
        # create a buffer
        num_frames = 0
        # run for a couple steps to check that everything is working
        for count in range(5):
            self.sim.step()

    def test_nav_marker_color(self):
        """Test with marker from a USD with its color modified."""
        # create a marker
        config_pin_sphere = PIN_SPHERE_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_pin_arrow = PIN_ARROW_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_pin_diamond = PIN_DIAMOND_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        config_arrow = ARROW_CFG.replace(prim_path="/World/Visuals/test_arrow")
        config_diamond = DIAMOND_CFG.replace(prim_path="/World/Visuals/test_diamond")
        config_bicolor_diamond = BICOLOR_DIAMOND_CFG.replace(prim_path="/World/Visuals/test_bicolor_diamond")
        config_position_maker_3d = POSITION_MARKER_3D_CFG.replace(prim_path="/World/Visuals/test_position_marker_3d")
        config_pose_marker_3d = POSE_MARKER_3D_CFG.replace(prim_path="/World/Visuals/test_pose_marker_3d")
        config_pin_sphere.markers["pin_sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        config_pin_arrow.markers["pin_arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        config_pin_diamond.markers["pin_diamond"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        config_arrow.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        config_diamond.markers["diamond"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        config_bicolor_diamond.markers["bicolor_diamond"].front_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        config_bicolor_diamond.markers["bicolor_diamond"].back_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)
        )
        config_position_maker_3d.markers["position_marker_3d"].x_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        config_position_maker_3d.markers["position_marker_3d"].y_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)
        )
        config_position_maker_3d.markers["position_marker_3d"].z_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)
        )
        config_pose_marker_3d.markers["pose_marker_3d"].x_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)
        )
        config_pose_marker_3d.markers["pose_marker_3d"].y_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)
        )
        config_pose_marker_3d.markers["pose_marker_3d"].z_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)
        )
        # Try to build the markers
        test_pin_sphere = VisualizationMarkers(config_pin_sphere)
        test_pin_arrow = VisualizationMarkers(config_pin_arrow)
        test_pin_diamond = VisualizationMarkers(config_pin_diamond)
        test_arrow = VisualizationMarkers(config_arrow)
        test_diamond = VisualizationMarkers(config_diamond)
        test_bicolor_diamond = VisualizationMarkers(config_bicolor_diamond)
        test_position_marker_3d = VisualizationMarkers(config_position_maker_3d)
        test_pose_marker_3d = VisualizationMarkers(config_pose_marker_3d)
        # Try to visualize the markers
        test_pin_sphere.visualize(translations=torch.tensor([[0.0, 0.0, 0.0]]))
        test_pin_arrow.visualize(translations=torch.tensor([[1.0, 0.0, 0.0]]))
        test_pin_diamond.visualize(translations=torch.tensor([[2.0, 0.0, 0.0]]))
        test_arrow.visualize(translations=torch.tensor([[3.0, 0.0, 0.0]]))
        test_diamond.visualize(translations=torch.tensor([[4.0, 0.0, 0.0]]))
        test_bicolor_diamond.visualize(translations=torch.tensor([[5.0, 0.0, 0.0]]))
        test_position_marker_3d.visualize(translations=torch.tensor([[6.0, 0.0, 0.0]]))
        test_pose_marker_3d.visualize(translations=torch.tensor([[7.0, 0.0, 0.0]]))

        # play the simulation
        self.sim.reset()
        # create a buffer
        num_frames = 0
        # run for a couple steps to check that everything is working
        for count in range(5):
            self.sim.step()

    def test_nav_marker_basic_properties(self):
        """Test with marker from a USD with its base properties."""
        config_pin_sphere = PIN_SPHERE_CFG.replace(prim_path="/World/Visuals/test_pin_sphere")
        config_pin_arrow = PIN_ARROW_CFG.replace(prim_path="/World/Visuals/test_pin_arrow")
        config_pin_diamond = PIN_DIAMOND_CFG.replace(prim_path="/World/Visuals/test_pin_diamond")
        config_arrow = ARROW_CFG.replace(prim_path="/World/Visuals/test_arrow")
        config_diamond = DIAMOND_CFG.replace(prim_path="/World/Visuals/test_diamond")
        config_bicolor_diamond = BICOLOR_DIAMOND_CFG.replace(prim_path="/World/Visuals/test_bicolor_diamond")
        config_position_marker_3d = POSITION_MARKER_3D_CFG.replace(prim_path="/World/Visuals/test_position_marker_3d")
        config_pose_marker_3d = POSE_MARKER_3D_CFG.replace(prim_path="/World/Visuals/test_pose_marker_3d")
        config_pin_sphere.markers["pin_sphere"].pin_radius = 0.1
        config_pin_sphere.markers["pin_sphere"].pin_length = 2.0
        config_pin_sphere.markers["pin_sphere"].sphere_radius = 0.5
        config_pin_arrow.markers["pin_arrow"].arrow_body_length = 0.6
        config_pin_arrow.markers["pin_arrow"].arrow_body_radius = 0.15
        config_pin_arrow.markers["pin_arrow"].arrow_head_radius = 0.3
        config_pin_arrow.markers["pin_arrow"].arrow_head_length = 0.45
        config_pin_arrow.markers["pin_arrow"].pin_radius = 0.1
        config_pin_arrow.markers["pin_arrow"].pin_length = 2.0
        config_pin_diamond.markers["pin_diamond"].diamond_height = 0.9
        config_pin_diamond.markers["pin_diamond"].diamond_width = 0.6
        config_pin_diamond.markers["pin_diamond"].pin_radius = 0.1
        config_pin_diamond.markers["pin_diamond"].pin_length = 2.0
        config_arrow.markers["arrow"].arrow_body_radius = 0.15
        config_arrow.markers["arrow"].arrow_body_length = 0.6
        config_arrow.markers["arrow"].arrow_head_radius = 0.3
        config_arrow.markers["arrow"].arrow_head_length = 0.45
        config_diamond.markers["diamond"].diamond_height = 0.9
        config_diamond.markers["diamond"].diamond_width = 0.6
        config_bicolor_diamond.markers["bicolor_diamond"].diamond_height = 0.9
        config_bicolor_diamond.markers["bicolor_diamond"].diamond_width = 0.6
        config_position_marker_3d.markers["position_marker_3d"].pin_length = 1.0
        config_position_marker_3d.markers["position_marker_3d"].pin_radius = 0.05
        config_position_marker_3d.markers["position_marker_3d"].sphere_radius = 0.25
        config_pose_marker_3d.markers["pose_marker_3d"].arrow_body_length = 1.0
        config_pose_marker_3d.markers["pose_marker_3d"].arrow_body_radius = 0.05
        config_pose_marker_3d.markers["pose_marker_3d"].arrow_head_radius = 0.1
        config_pose_marker_3d.markers["pose_marker_3d"].arrow_head_length = 0.15
        # Try to build the markers
        test_pin_sphere = VisualizationMarkers(config_pin_sphere)
        test_pin_arrow = VisualizationMarkers(config_pin_arrow)
        test_pin_diamond = VisualizationMarkers(config_pin_diamond)
        test_arrow = VisualizationMarkers(config_arrow)
        test_diamond = VisualizationMarkers(config_diamond)
        test_bicolor_diamond = VisualizationMarkers(config_bicolor_diamond)
        test_position_marker_3d = VisualizationMarkers(config_position_marker_3d)
        test_pose_marker_3d = VisualizationMarkers(config_pose_marker_3d)
        # Try to visualize the markers
        test_pin_sphere.visualize(translations=torch.tensor([[0.0, 0.0, 0.0]]))
        test_pin_arrow.visualize(translations=torch.tensor([[1.0, 0.0, 0.0]]))
        test_pin_diamond.visualize(translations=torch.tensor([[2.0, 0.0, 0.0]]))
        test_arrow.visualize(translations=torch.tensor([[3.0, 0.0, 0.0]]))
        test_diamond.visualize(translations=torch.tensor([[4.0, 0.0, 0.0]]))
        test_bicolor_diamond.visualize(translations=torch.tensor([[5.0, 0.0, 0.0]]))
        test_position_marker_3d.visualize(translations=torch.tensor([[6.0, 0.0, 0.0]]))
        test_pose_marker_3d.visualize(translations=torch.tensor([[7.0, 0.0, 0.0]]))

        # play the simulation
        self.sim.reset()
        # create a buffer
        num_frames = 0
        # run for a couple steps to check that everything is working
        for count in range(5):
            self.sim.step()


if __name__ == "__main__":
    run_tests()
