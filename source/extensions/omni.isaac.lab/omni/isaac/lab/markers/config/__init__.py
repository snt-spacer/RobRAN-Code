# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Sensors.
##

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
"""Configuration for the ray-caster marker."""


CONTACT_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "no_contact": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            visible=False,
        ),
    },
)
"""Configuration for the contact sensor marker."""

DEFORMABLE_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.8)),
        ),
    },
)
"""Configuration for the deformable object's kinematic target marker."""


##
# Frames.
##

FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        )
    }
)
"""Configuration for the frame marker."""


RED_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
"""Configuration for the red arrow marker (along x-direction)."""


BLUE_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)
"""Configuration for the blue arrow marker (along x-direction)."""

GREEN_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)
"""Configuration for the green arrow marker (along x-direction)."""


##
# Goals.
##

CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
"""Configuration for the cuboid marker."""

POSITION_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target_far": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_near": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "target_invisible": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            visible=False,
        ),
    }
)
"""Configuration for the end-effector tracking marker."""

##
# Navigation goal markers.
##

PIN_SPHERE_CFG = VisualizationMarkersCfg(
    markers={
        "pin_sphere": sim_utils.PinSphereCfg(
            sphere_radius=0.125,
            pin_radius=0.01,
            pin_length=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.0)),
        ),
    }
)
"""Configuration for the pin with sphere marker."""

PIN_DIAMOND_CFG = VisualizationMarkersCfg(
    markers={
        "pin_diamond": sim_utils.PinDiamondCfg(
            diamond_height=0.3,
            diamond_width=0.2,
            pin_radius=0.01,
            pin_length=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.333, 1.0, 0.0)),
        ),
    }
)
"""Configuration for the pin with diamond marker."""

PIN_ARROW_CFG = VisualizationMarkersCfg(
    markers={
        "pin_arrow": sim_utils.PinArrowCfg(
            arrow_body_length=0.2,
            arrow_body_radius=0.05,
            arrow_head_radius=0.1,
            arrow_head_length=0.15,
            pin_radius=0.01,
            pin_length=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.0)),
        ),
    }
)
"""Configuration for the pin with arrow marker."""
##
# Racing markers.
##


##
# Robot Pose/Position Markers.
##

DIAMOND_CFG = VisualizationMarkersCfg(
    markers={
        "diamond": sim_utils.DiamondCfg(
            diamond_height=0.15,
            diamond_width=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.333, 1.0, 0.0)),
        ),
    }
)

BICOLOR_DIAMOND_CFG = VisualizationMarkersCfg(
    markers={
        "bicolor_diamond": sim_utils.BiColorDiamondCfg(
            diamond_height=0.15,
            diamond_width=0.1,
            front_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.333, 1.0, 0.0)),
            back_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.0)),
        ),
    }
)

ARROW_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.ArrowCfg(
            arrow_body_length=0.2,
            arrow_body_radius=0.05,
            arrow_head_radius=0.1,
            arrow_head_length=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.0)),
        ),
    }
)
