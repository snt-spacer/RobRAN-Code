# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from omni.isaac.lab.scene import InteractiveScene

from omni.isaac.lab_tasks.rans import (
    PerEnvSeededRNG,
    RandomizationCore,
    RandomizationCoreCfg,
    RandomizerFactory,
    RobotCore,
    ScalarLogger,
    TaskCoreCfg,
)


class TaskCore:
    """
    The base class that implements the core of the task.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        The base class for the different subtasks.

        Args:
            scene: The scene in which the task is run.
            task_uid: The unique id of the task.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
            env_ids: The ids of the environments used by this task."""

        self.scene = scene

        self._task_cfg = TaskCoreCfg()
        # Unique task identifier, used to differentiate between tasks with the same name
        self._task_uid = task_uid
        # Number of environments and device to be used
        self._num_envs = num_envs
        self._device = device

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs: int = MISSING
        self._dim_gen_act: int = MISSING

        # Robot
        self._robot: RobotCore = MISSING

        # RNG
        seeds = torch.randint(0, 2**31, (self._num_envs,), dtype=torch.int32, device=self._device)
        self._rng = PerEnvSeededRNG(seeds, self._num_envs, self._device)

        # Logs
        self.create_logs()

    @property
    def num_observations(self) -> int:
        return self._dim_task_obs

    @property
    def num_gen_actions(self) -> int:
        return self._dim_gen_act

    @property
    def logs(self) -> dict:
        return self.scalar_logger.get_episode_logs

    def get_randomizers(self) -> None:
        """Collects the randomizers applied to the robot."""

        self.randomizers: list[RandomizationCore] = []
        for attr in self._task_cfg.__dir__():
            if isinstance(getattr(self._task_cfg, attr), RandomizationCoreCfg):
                self.randomizers.append(
                    RandomizerFactory.create(
                        getattr(self._task_cfg, attr),
                        self._rng,
                        self.scene,
                        asset_name=self._robot._robot_cfg.robot_name,
                        num_envs=self._num_envs,
                        device=self._device,
                    )
                )

    def design_scene(self) -> None:
        """Adds objects to the scene."""
        pass

    def create_logs(self) -> None:
        """
        Initializes class for logging.

        -type: Task logs
        """
        self.scalar_logger = ScalarLogger(self._num_envs, self._device, "task")

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        # Buffers
        if env_ids is None:
            self._env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        else:
            self._env_ids = env_ids
        self._env_origins = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._robot_origins = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._seeds = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        self._goal_reached = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)
        self._gen_actions = torch.zeros(
            (self._num_envs, self._dim_gen_act),
            device=self._device,
            dtype=torch.float32,
        )
        self._task_data = torch.zeros(
            (self._num_envs, self._dim_task_obs),
            device=self._device,
            dtype=torch.float32,
        )
        self._robot_marker_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

    def run_setup(self, robot: RobotCore, envs_origin: torch.Tensor) -> None:
        """
        Sets the default origins of the environments and the robot.

        Args:
            env_origins (torch.Tensor): The origins of the environments.
            robot_origins (torch.Tensor): The origins of the robot."""

        self._robot = robot
        self._env_origins = envs_origin.clone()
        self._robot_origins = self._robot._robot.data.default_root_state[:, :3].clone()
        self._robot_marker_pos[:, 2] = self._robot._robot_cfg.marker_height

        self.get_randomizers()
        # Run the setup functions of the randomizers
        for randomizer in self.randomizers:
            randomizer.setup()

    def get_observations(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_rewards(self) -> torch.Tensor:
        raise NotImplementedError

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset_logs(self, env_ids, episode_length_buf) -> None:
        self.scalar_logger.reset(env_ids, episode_length_buf)

    def compute_logs(self) -> dict:
        return self.scalar_logger.compute_extras()

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        """
        Resets the task to its initial state.

        If gen_actions is None, then the environment is generated at random. This is the default mode.
        If env_seeds is None, then the seed is generated at random. This is the default mode.

        Args:
            task_actions (torch.Tensor | None): The actions to be taken to generate the env.
            env_seed (torch.Tensor | None): The seed to used in each environment.
            env_ids (torch.Tensor): The ids of the environments."""

        # Updates the seed
        if env_seeds is None:
            self._seeds[env_ids] = torch.randint(0, 2**31, (len(env_ids),), dtype=torch.int32, device=self._device)
        else:
            self._seeds[env_ids] = env_seeds

        # Update the RNG
        self._rng.set_seeds(self._seeds[env_ids], env_ids)

        # Reset the robot
        self._robot.reset(env_ids)

        # Updates the task actions
        if gen_actions is None:
            self._gen_actions[env_ids] = torch.rand((len(env_ids), self.num_gen_actions), device=self._device)
        else:
            self._gen_actions[env_ids] = gen_actions

        # Reset the randomizers
        for randomizer in self.randomizers:
            randomizer.reset(env_ids)

        # Randomizes goals and initial conditions
        self.set_goals(env_ids)
        self.set_initial_conditions(env_ids)

        # Resets the goal reached flag
        self._goal_reached[env_ids] = 0

    def set_goals(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def create_task_visualization(self) -> None:
        raise NotImplementedError

    def update_task_visualization(self) -> None:
        raise NotImplementedError

    def register_rigid_objects(self) -> None:
        pass
