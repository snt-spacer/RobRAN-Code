# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RobotCoreCfg:
    """Core configuration for a RANS robot."""

    robot_name: str = MISSING

    ema_coeff: float = 0.9
    """Exponential moving average coefficient used to update some of the logs."""

    contact_sensor_active: bool = False
    """Flag to enable the contact sensor."""
