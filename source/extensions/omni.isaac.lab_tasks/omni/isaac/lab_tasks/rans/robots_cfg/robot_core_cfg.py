from omni.isaac.lab.utils import configclass
from dataclasses import MISSING


@configclass
class RobotCoreCfg:
    """Core configuration for a RANS robot."""

    ema_coeff: float = 0.9
    """Exponential moving average coefficient used to update some of the logs."""
