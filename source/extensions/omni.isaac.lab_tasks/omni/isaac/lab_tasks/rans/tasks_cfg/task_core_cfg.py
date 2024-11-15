# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass


@configclass
class TaskCoreCfg:
    """Core configuration for a RANS task."""

    maximum_robot_distance: float = MISSING
    """Maximal distance between the robot and the target pose."""
    reset_after_n_steps_in_tolerance: int = MISSING
    """Reset the environment after n steps in tolerance."""
    ema_coeff: float = 0.9
    """Exponential moving average coefficient used to update some of the logs."""
