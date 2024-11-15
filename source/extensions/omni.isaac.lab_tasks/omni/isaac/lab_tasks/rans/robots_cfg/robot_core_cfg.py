# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass


@configclass
class RobotCoreCfg:
    """Core configuration for a RANS robot."""

    ema_coeff: float = 0.9
    """Exponential moving average coefficient used to update some of the logs."""
