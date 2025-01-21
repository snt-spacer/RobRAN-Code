# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn robots using a generation_function.

Currently, the following spawners are supported:

* :class:`RobotFromCodeCfg`: Provides the base class to spawn a robot using raw USD code exclusively.
"""

from .robot_from_code import spawn_robot_from_code
from .robot_from_code_cfg import RobotFromCodeCfg
