from omni.isaac.lab_tasks.rans import TaskCoreCfg
from omni.isaac.lab.assets import Articulation

from dataclasses import MISSING

import torch


class TaskCore:
    """
    The base class that implements the core of the task.
    """

    def __init__(
        self,
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        The base class for the different subtasks.

        Args:
            task_uid: The unique id of the task.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
            env_ids: The ids of the environments used by this task."""

        # Unique task identifier, used to differentiate between tasks with the same name
        self._task_uid = task_uid
        # Number of environments and device to be used
        self._num_envs = num_envs
        self._device = device

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = MISSING
        self._dim_env_act = MISSING

        # Robot
        self._robot: Articulation = MISSING

        # Buffers

        # Logs
        self._logs = {}
        self.create_logs()

    @property
    def num_observations(self) -> int:
        return self._dim_task_obs

    @property
    def num_actions(self) -> int:
        return self._dim_env_act

    @property
    def logs(self) -> dict:
        return self._logs

    def create_logs(self) -> None:
        self._logs["state"] = {}
        self._logs["reward"] = {}

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        # Buffers
        if env_ids is None:
            self._env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        else:
            self._env_ids = env_ids
        self._env_origins = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._robot_origins = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._seeds = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        self._goal_reached = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)
        self._env_actions = torch.zeros((self._num_envs, self._dim_env_act), device=self._device, dtype=torch.float32)
        self._task_data = torch.zeros((self._num_envs, self._dim_task_obs), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: Articulation, envs_origin: torch.Tensor) -> None:
        """
        Sets the default origins of the environments and the robot.

        Args:
            env_origins (torch.Tensor): The origins of the environments.
            robot_origins (torch.Tensor): The origins of the robot.
        """
        self.robot = robot
        self._env_origins = envs_origin.clone()

    def get_observations(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_rewards(self) -> torch.Tensor:
        raise NotImplementedError

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset_logs(self, env_ids) -> None:
        for key in self._logs:
            self._logs[key][env_ids] = 0

    def reset(self, task_actions: torch.Tensor, env_seeds: torch.Tensor, env_ids: torch.Tensor) -> None:
        """
        Args:
            env_ids (torch.Tensor): The ids of the environments.
            task_actions (torch.Tensor): The actions to be taken to generate the env.
            env_seed (torch.Tensor): The seed to used in each environment.
        """
        # Updates the task actions
        self._env_actions[env_ids] = task_actions
        # Updates the seed
        self._seeds[env_ids] = env_seeds

        # Randomizes goals and initial conditions
        self.set_goals(env_ids)
        self.set_initial_conditions(env_ids)

        # Resets the goal reached flag
        self._goal_reached[env_ids] = 0
        self.update_task_visualization()

        # Resets the logs
        self.reset_logs(env_ids)

    def set_goals(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def set_initial_conditions(self, env_ids: torch.Tensor):
        raise NotImplementedError

    def create_task_visualization(self):
        raise NotImplementedError

    def update_task_visualization(self):
        raise NotImplementedError
