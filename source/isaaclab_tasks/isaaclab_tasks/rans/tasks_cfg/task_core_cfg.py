# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class TaskCoreCfg:
    """Core configuration for a RANS task."""

    maximum_robot_distance: float = MISSING
    """Maximal distance between the robot and the target pose."""
    ema_coeff: float = 0.9
    """Exponential moving average coefficient used to update some of the logs."""
