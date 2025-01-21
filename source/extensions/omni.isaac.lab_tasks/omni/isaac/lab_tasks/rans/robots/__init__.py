# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab_tasks.rans.utils.misc import factory

from .floating_platform import FloatingPlatformRobot
from .jetbot import JetbotRobot
from .leatherback import LeatherbackRobot
from .modular_freeflyer import ModularFreeflyerRobot
from .robot_core import RobotCore

ROBOT_FACTORY = factory()
ROBOT_FACTORY.register("Jetbot", JetbotRobot)
ROBOT_FACTORY.register("Leatherback", LeatherbackRobot)
ROBOT_FACTORY.register("FloatingPlatform", FloatingPlatformRobot)
ROBOT_FACTORY.register("ModularFreeflyer", ModularFreeflyerRobot)
