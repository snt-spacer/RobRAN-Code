# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn assets from files.

Currently, the following spawners are supported:

* :class:`FromCodeCfg`: Provides the base class to spawn a robot using raw USD code exclusively.
"""

from .robots_from_code import spawn_from_code
from .robots_from_code_cfg import FromCodeCfg
