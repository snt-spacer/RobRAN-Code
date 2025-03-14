# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class ScalarLogger:
    def __init__(self, num_envs: int, device: str, type: str) -> None:
        """
        Class for logging.
        - _step_logs: Logs data on a per-step basis.
        - _episode_logs: Logs data at the end of each episode.
        - _logs_operation: Holds operation to indicate how certain episode-level logs
        should be computed.

        Args:
            num_envs (int): The number of environments.
            device (str): The device to use.
            type (str): The type of log. It's only used for naming purposes. It can be "robot" or "task" for instance.
        """

        self._num_envs = num_envs
        self._device = device
        self._type = type

        self._step_logs = {f"{self._type}_state": {}, f"{self._type}_reward": {}}
        self._episode_logs = {f"{self._type}_state": {}, f"{self._type}_reward": {}}
        self._logs_operation = {f"{self._type}_state": {}, f"{self._type}_reward": {}}

        self._supported_ops = ["sum", "mean", "max", "min", "ema"]
        self._operations_map = {
            "sum": self.sum_logs,
            "mean": self.mean_logs,
            "ema": self.ema_logs,
            "max": self.max_logs,
            "min": self.min_logs,
        }

        self.ema_coeff = 0.9

    def torch_zeros(self) -> torch.Tensor:
        """Create a tensor of zeros with the same shape as the number of environments.

        Returns:
            torch.Tensor: A tensor of zeros with the same shape as the number of environments."""

        return torch.zeros(
            self._num_envs,
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )

    def add_log(self, type: str, name: str, operation: str) -> None:
        """Add a log to the logger.

        Args:
            type (str): The type of log. It's solely used for naming purposes. It can be "robot" or "task" for instance.
            name (str): The name of the log.
            operation (str): The operation to be performed on the log. One of ["sum", "mean", "max", "min", "ema"]."""

        assert type in [f"{self._type}_state", f"{self._type}_reward"], f"Invalid log type: {type}"
        assert operation in self._supported_ops, f"Invalid operation: {operation}"
        self._step_logs[type][name] = self.torch_zeros()
        self._episode_logs[type][name] = self.torch_zeros()
        self._logs_operation[type][name] = operation

    def log(self, type: str, name: str, value: torch.Tensor) -> None:
        """Log a value.

        Args:
            type (str): The type of log. It's solely used for naming purposes, it can be "robot" or "task" for instance.
            name (str): The name of the log.
            value (torch.Tensor): The value to be logged."""

        op = self._logs_operation[type][name]
        self._step_logs[type][name] = self._operations_map[op](type, name, value)

    @property
    def get_step_logs(self) -> dict:
        """Get the step logs."""
        return self._step_logs

    @property
    def get_episode_logs(self) -> dict:
        """Get the episode logs."""
        return self._episode_logs

    def reset(self, env_ids: torch.Tensor, episode_length_buf: torch.Tensor) -> None:
        """Reset the logs of the given environments based on their ids.
        The mean is computed by dividing the sum by the episode length buffer passed as an argument.
        # TODO: Decide if we built-in our own counter.

        Args:
            env_ids (torch.Tensor): The environment IDs.
            episode_length_buf (torch.Tensor): The episode length buffer."""

        for rew_state_key in self._step_logs:
            for key in self._step_logs[rew_state_key]:
                op = self._logs_operation[rew_state_key][key]
                if op == "mean":
                    # Avoid division by zero
                    episode_length = episode_length_buf[env_ids] + (episode_length_buf[env_ids] == 0) * 1e-7
                    self._episode_logs[rew_state_key][key][env_ids] = torch.div(
                        self._step_logs[rew_state_key][key][env_ids], episode_length
                    )
                elif op == "max":
                    self._episode_logs[rew_state_key][key][env_ids] = self._step_logs[rew_state_key][key][env_ids].max()
                elif op == "min":
                    self._episode_logs[rew_state_key][key][env_ids] = self._step_logs[rew_state_key][key][env_ids].min()
                else:
                    self._episode_logs[rew_state_key][key][env_ids] = self._step_logs[rew_state_key][key][env_ids]

                self._step_logs[rew_state_key][key][env_ids] = 0

    def compute_extras(self) -> dict:
        """The function used to format the logs to be returned to the environment and used by tensorboard or
        wandb."""

        extras = dict()
        for rew_state_key in self._episode_logs.keys():
            for key in self._episode_logs[rew_state_key]:
                extras[rew_state_key + "/" + key] = self._episode_logs[rew_state_key][key].mean()

        return extras

    def min_logs(self, type, name, value):
        """Minimum operation when adding a new data point to the logs."""

        return value

    def max_logs(self, type, name, value):
        """Maximum operation when adding a new data point to the logs."""

        return value

    def sum_logs(self, type, name, value):
        """Sum operation when adding a new data point to the logs."""

        return self._step_logs[type][name] + value

    def ema_logs(self, type, name, value):
        """Exponential moving average operation when adding a new data point to the logs."""

        return value * (1 - self.ema_coeff) + self._step_logs[type][name] * self.ema_coeff

    def mean_logs(self, type, name, value):
        """Mean operation when adding a new data point to the logs."""

        return self._step_logs[type][name] + value

    def set_ema_coeff(self, ema_coeff):
        """Set the exponential moving average coefficient."""

        self.ema_coeff = ema_coeff
