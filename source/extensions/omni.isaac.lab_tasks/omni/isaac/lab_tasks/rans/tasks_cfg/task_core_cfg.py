from omni.isaac.lab.utils import configclass
from dataclasses import MISSING


@configclass
class TaskCoreCfg:
    """Core configuration for a RANS task."""

    maximum_robot_distance: float = MISSING
    """Maximal distance between the robot and the target pose."""
    reset_after_n_steps_in_tolerance: int = MISSING
    """Reset the environment after n steps in tolerance."""
    ema_coeff: float = 0.9
    """Exponential moving average coefficient used to update some of the logs."""
