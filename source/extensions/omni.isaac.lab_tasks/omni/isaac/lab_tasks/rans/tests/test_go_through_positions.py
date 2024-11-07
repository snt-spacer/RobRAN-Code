# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from omni.isaac.lab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import unittest
