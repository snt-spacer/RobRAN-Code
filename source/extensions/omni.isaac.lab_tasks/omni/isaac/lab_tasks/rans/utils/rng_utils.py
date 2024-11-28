# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import warp as wp

from .rng_kernels import integer, normal, poisson, quaternion, rand_sign_fn, set_states, uniform


class PerEnvSeededRNG:
    def __init__(self, seeds: int | torch.Tensor, num_envs: int, device: str):
        """Initialize the random number generator.
        Args:
            seeds: The seeds for each environment.
            num_envs: The number of environments.
            device: The device to use."""

        self._device = device
        self._num_envs = num_envs

        # Instantiate buffers
        if isinstance(seeds, int):
            self._seeds = wp.array(np.ones(num_envs) * seeds, dtype=wp.int32, device=device)
        else:
            self._seeds = wp.from_torch(seeds, dtype=wp.int32, device=device)

        self._states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._new_states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._ALL_INDICES = wp.array(np.arange(num_envs), dtype=wp.int32, device=device)

    @property
    def seeds_warp(self) -> wp.array:
        """Get the seeds for each environment."""
        return self._seeds

    @property
    def seeds_torch(self) -> torch.Tensor:
        """Get the seeds for each environment."""
        return wp.to_torch(self._seeds)

    @property
    def seeds_numpy(self) -> np.ndarray:
        """Get the seeds for each environment."""
        return self._seeds.numpy()

    @property
    def states_warp(self) -> wp.array:
        """Get the states for each environment."""
        return self._states

    @property
    def states_torch(self) -> torch.Tensor:
        """Get the states for each environment."""
        return wp.to_torch(self._states)

    @property
    def states_numpy(self) -> np.ndarray:
        """Get the states for each environment."""
        return self._states.numpy()

    @staticmethod
    def to_tuple(shape: int | tuple[int]) -> tuple:
        """Casts to a tuple."""
        if isinstance(shape, int):
            return (shape,)
        else:
            return shape

    @staticmethod
    def get_offset(shape: tuple[int]) -> int:
        """Get the offset based on the shape."""
        out = 1
        for i in shape:
            out *= i
        return out

    def set_seeds_warp(self, seeds: wp.array, ids: wp.array | None) -> None:
        """Set the seeds for each environment.
        Args:
            seeds: The seeds for each environment.
            ids: The ids of the environments."""

        if ids is None:
            ids = self._ALL_INDICES

        num_instances = len(seeds)
        wp.launch(
            kernel=set_states,
            dim=num_instances,
            inputs=[seeds, self._seeds, self._states, ids],
            device=self._device,
        )

    def set_seeds(self, seeds: torch.Tensor, ids: torch.Tensor | None) -> None:
        """Set the seeds for each environment.
        If ids is None, the seeds are set for all environments. No checks are performed on the input tensors,
        It is the user's responsibility to ensure that the tensors are of the correct shape.
        Args:
            seeds: The seeds for each environment.
            ids: The ids of the environments."""

        if isinstance(ids, torch.Tensor):
            self.set_seeds_warp(
                wp.from_torch(seeds, dtype=wp.int32),
                wp.from_torch(ids, dtype=wp.int32),
            )
        else:
            self.set_seeds_warp(
                wp.from_torch(seeds, dtype=wp.int32),
                None,
            )

    def set_seeds_numpy(self, seeds: np.ndarray, ids: np.ndarray | None) -> None:
        """Set the seeds for each environment.
        Args:
            seeds: The seeds for each environment.
            ids: The ids of the environments."""

        if isinstance(ids, np.ndarray):
            self.set_seeds_warp(
                wp.array(seeds, dtype=wp.int32, device=self._device),
                wp.array(ids, dtype=wp.uint32, device=self._device),
            )
        else:
            self.set_seeds_warp(
                wp.array(seeds, dtype=wp.int32, device=self._device),
                None,
            )

    def sample_uniform_warp(self, low: float, high: float, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample from a uniform distribution. Warp implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return uniform(low, high, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_uniform_torch(
        self, low: float, high: float, shape: tuple | int, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample from a uniform distribution. Torch implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = wp.from_torch(ids, dtype=wp.int32)
        output = self.sample_uniform_warp(low, high, shape, ids)
        return wp.to_torch(output)

    def sample_uniform_numpy(
        self, low: float, high: float, shape: tuple | int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample from a uniform distribution. Numpy implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = wp.array(ids, dtype=wp.int32, device=self._device)
        output = self.sample_uniform_warp(low, high, shape, ids)
        return output.numpy()

    def sample_sign_warp(self, dtype: str, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample a sign. Warp implementation.
        Args:
            dtype: The data type of the output tensor.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return rand_sign_fn(self._states, self._new_states, ids, self.to_tuple(shape), dtype, self._device)

    def sample_sign_torch(self, dtype: str, shape: tuple | int, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Sample a sign. Torch implementation.
        Args:
            dtype: The data type of the output tensor.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = wp.from_torch(ids, dtype=wp.int32)
        assert dtype in ["int", "float"], "The data type must be either 'int' or 'float'."
        output = self.sample_sign_warp(dtype, shape, ids)
        return wp.to_torch(output)

    def sample_sign_numpy(self, dtype: str, shape: tuple | int, ids: np.ndarray | None = None) -> np.ndarray:
        """Sample a sign. Numpy implementation.
        Args:
            dtype: The data type of the output tensor.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = wp.array(ids, dtype=wp.int32, device=self._device)
        assert dtype in ["int", "float"], "The data type must be either 'int' or 'float'."
        output = self.sample_sign_warp(dtype, shape, ids)
        return output.numpy()

    # Rand int
    def sample_integer_warp(self, low: int, high: int, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample for a random integer. Warp implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return integer(low, high, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_integer_torch(
        self, low: int, high: int, shape: tuple | int, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample for a random integer. Torch implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is None:
            ids = wp.from_torch(ids, dtype=wp.int32)
        output = self.sample_integer_warp(low, high, shape, ids)
        return wp.to_torch(output)

    def sample_integer_numpy(
        self, low: int, high: int, shape: tuple | int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample for a random integer. Numpy implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            np.ndarray: The sampled values."""
        if ids is None:
            ids = wp.array(ids, dtype=wp.int32, device=self._device)
        output = self.sample_integer_warp(low, high, shape, ids)
        return output.numpy()

    def sample_normal_warp(self, mean: float, std: float, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample from a normal distribution. Warp implementation.
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return normal(mean, std, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_normal_torch(
        self, mean: float, std: float, shape: tuple | int, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample from a normal distribution. Torch implementation.
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is None:
            ids = wp.from_torch(ids, dtype=wp.int32)
        output = self.sample_normal_warp(mean, std, shape, ids)
        return wp.to_torch(output)

    def sample_normal_numpy(
        self, mean: float, std: float, shape: tuple | int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample from a normal distribution. Numpy implementation.
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            np.ndarray: The sampled values."""
        if ids is None:
            ids = wp.array(ids, dtype=wp.int32, device=self._device)
        output = self.sample_normal_warp(mean, std, shape, ids)
        return output.numpy()

    def sample_poisson_warp(self, lam: float, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample from a poisson distribution. Warp implementation.
        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return poisson(lam, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_poisson_torch(self, lam: float, shape: tuple | int, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Sample from a poisson distribution. Torch implementation.
        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = wp.from_torch(ids, dtype=wp.int32)
        output = self.sample_poisson_warp(lam, shape, ids)
        return wp.to_torch(output)

    def sample_poisson_numpy(self, lam: float, shape: tuple | int, ids: np.ndarray | None) -> np.ndarray:
        """Sample from a poisson distribution. Numpy implementation.
        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = wp.array(ids, dtype=wp.int32, device=self._device)
        output = self.sample_poisson_warp(lam, shape, ids)
        return output.numpy()

    def sample_quaternion_warp(self, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample a quaternion. Warp implementation.
        Args:
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return quaternion(self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_quaternion_torch(self, shape: tuple | int, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Sample a quaternion. Torch implementation.
        Args:
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is None:
            ids = wp.from_torch(ids, dtype=wp.int32)
        output = self.sample_quaternion_warp(shape, ids)
        return wp.to_torch(output)

    def sample_quaternion_numpy(self, shape: tuple | int, ids: np.ndarray | None = None) -> np.ndarray:
        """Sample a quaternion. Numpy implementation.
        Args:
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            np.ndarray: The sampled values."""
        if ids is None:
            ids = wp.array(ids, dtype=wp.int32, device=self._device)
        output = self.sample_quaternion_warp(shape, ids)
        return output.numpy()
