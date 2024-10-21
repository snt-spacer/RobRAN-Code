from omni.isaac.lab.assets import ArticulationData, Articulation
from omni.isaac.lab_tasks.rans import RobotCoreCfg

import torch


class RobotCore:
    def __init__(self, robot_cfg: RobotCoreCfg, robot_uid: int = 0, num_envs: int = 1, device: str = "cuda"):
        """Initializes the robot core.

        Args:
            robot_cfg: The configuration of the robot.
            robot_uid: The unique id of the robot.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        # Task parameters
        self._robot_cfg = robot_cfg
        # Unique task identifier, used to differentiate between tasks with the same name
        self._robot_uid = robot_uid
        # Number of environments and device to be used
        self._num_envs = num_envs
        self._device = device

        # The number of observations/actions for the robot & the environment.
        self._dim_robot_obs = 0
        self._dim_robot_act = 0
        self._dim_env_act = 0

        # Buffers
        self._seeds = torch.arange(num_envs, device=self._device, dtype=torch.int32)

        # Logs
        self._logs = {}
        self.create_logs()

    @property
    def num_observations(self):
        """Returns the number of observations for the robot.
        Typically, this would be linked to the joints of the robot or its actions.
        It's what's unique to that robot."""

        return self._dim_robot_obs

    @property
    def num_actions(self):
        """Returns the number of actions for the robot. This is the actions that the robot can take.
        Not the randomization of the environment."""

        return self._dim_robot_act

    @property
    def logs(self) -> dict:
        return self._logs

    @property
    def statistics(self):
        raise NotImplementedError

    def run_setup(self, articulation: Articulation):
        raise NotImplementedError

    def create_logs(self):
        self._logs["state"] = {}
        self._logs["reward"] = {}

    def get_observations(self):
        raise NotImplementedError

    def compute_rewards(self, robot_data: ArticulationData):
        raise NotImplementedError

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(
        self, task_actions: torch.Tensor, env_seeds: torch.Tensor, articulations: Articulation, env_ids: torch.Tensor
    ):
        raise NotImplementedError

    def set_initial_conditions(self, env_ids, articulations: Articulation):
        raise NotImplementedError

    def process_actions(self):
        raise NotImplementedError

    def compute_physics(self):
        raise NotImplementedError

    def apply_actions(self):
        raise NotImplementedError

    def updateMass(self):
        raise NotImplementedError

    def updateInertia(self):
        raise NotImplementedError

    def updateCoM(self):
        raise NotImplementedError

    def updateFriction(self):
        raise NotImplementedError
