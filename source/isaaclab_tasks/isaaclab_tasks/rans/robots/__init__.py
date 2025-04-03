# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tasks.rans.utils.misc import factory

from .floating_platform import FloatingPlatformRobot
from .intball2 import IntBall2Robot
from .jetbot import JetbotRobot
from .kingfisher import KingfisherRobot
from .leatherback import LeatherbackRobot
from .modular_freeflyer import ModularFreeflyerRobot
from .robot_core import RobotCore
from .turtlebot2 import TurtleBot2Robot

ROBOT_FACTORY = factory()
ROBOT_FACTORY.register("Jetbot", JetbotRobot)
ROBOT_FACTORY.register("Leatherback", LeatherbackRobot)
ROBOT_FACTORY.register("FloatingPlatform", FloatingPlatformRobot)
ROBOT_FACTORY.register("ModularFreeflyer", ModularFreeflyerRobot)
ROBOT_FACTORY.register("Kingfisher", KingfisherRobot)
ROBOT_FACTORY.register("Turtlebot2", TurtleBot2Robot)
ROBOT_FACTORY.register("IntBall2", IntBall2Robot)
