# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# isort: off
from .robot_core_cfg import RobotCoreCfg  # noqa: F401, F403

# # isort: on
from .leatherback_cfg import LeatherbackRobotCfg  # noqa: F401, F403
from .floating_platform_cfg import FloatingPlatformRobotCfg  # noqa: F401, F403
from .jetbot_cfg import JetbotRobotCfg  # noqa: F401, F403
from .modular_freeflyer_cfg import ModularFreeflyerRobotCfg  # noqa: F401, F403

from omni.isaac.lab_tasks.rans.utils.misc import factory

ROBOT_CFG_FACTORY = factory()
ROBOT_CFG_FACTORY.register("Jetbot", JetbotRobotCfg)
ROBOT_CFG_FACTORY.register("Leatherback", LeatherbackRobotCfg)
ROBOT_CFG_FACTORY.register("FloatingPlatform", FloatingPlatformRobotCfg)
ROBOT_CFG_FACTORY.register("ModularFreeflyer", ModularFreeflyerRobotCfg)
