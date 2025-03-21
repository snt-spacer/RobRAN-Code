# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
            self._seeds = wp.from_torch(seeds, dtype=wp.int32)

        self._states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._new_states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._ALL_INDICES = wp.array(np.arange(num_envs), dtype=wp.int32, device=device)

        self._use_numpy_rng = False
        self._numpy_rng_is_initialized = False

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

    def initialize_numpy_rng(self) -> None:
        """Initialize the numpy random number generator."""

        self._numpy_rngs = [np.random.default_rng(seed) for seed in self.seeds_numpy]

        self._use_numpy_rng = True
        self._numpy_rng_is_initialized = True

    def set_numpy_rng_seeds(self, seeds: wp.array, ids: wp.array | None) -> None:
        """When using distributions that are not yet supported by our custom warp kernels, we fall
        back to numpy. This function sets the seeds for each environments."""
        if ids is None:
            ids = self._ALL_INDICES

        ids = ids.numpy()
        seeds = seeds.numpy()

        for i, seed in enumerate(seeds):
            self._numpy_rngs[i] = np.random.default_rng(seed)

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

        if self._use_numpy_rng:
            self.set_numpy_rng_seeds(seeds, ids)

    def set_seeds(self, seeds: torch.Tensor, ids: torch.Tensor | None) -> None:
        """Set the seeds for each environment.
        If ids is None, the seeds are set for all environments. No checks are performed on the input tensors,
        It is the user's responsibility to ensure that the tensors are of the correct shape.
        Args:
            seeds: The seeds for each environment.
            ids: The ids of the environments."""

        if isinstance(ids, torch.Tensor):
            self.set_seeds_warp(
                wp.from_torch(seeds.to(torch.int32), dtype=wp.int32),
                wp.from_torch(ids.to(torch.int32), dtype=wp.int32),
            )
        else:
            self.set_seeds_warp(
                wp.from_torch(seeds.to(torch.int32), dtype=wp.int32),
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

    def sample_uniform_warp(
        self, low: float | wp.array, high: float | wp.array, shape: tuple | int, ids: wp.array | None = None
    ) -> wp.array:
        """Sample from a uniform distribution. Warp implementation.

        If low and high are arrays, their shapes need to match that of the ids.

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
        self,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        shape: tuple | int,
        ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from a uniform distribution. Torch implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is not None:
            ids = wp.from_torch(ids.to(torch.int32), dtype=wp.int32)
        if isinstance(low, torch.Tensor):
            low = wp.from_torch(low)
        else:
            low = float(low)
        if isinstance(high, torch.Tensor):
            high = wp.from_torch(high)
        else:
            high = float(high)
        output = self.sample_uniform_warp(low, high, shape, ids)
        return wp.to_torch(output)

    def sample_uniform_numpy(
        self, low: float | np.ndarray, high: float | np.ndarray, shape: tuple | int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample from a uniform distribution. Numpy implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is not None:
            ids = wp.array(ids.to(torch.int32), dtype=wp.int32, device=self._device)
        if isinstance(low, np.ndarray):
            low = wp.array(low, dtype=float, device=self._device)
        else:
            low = float(low)
        if isinstance(high, np.ndarray):
            high = wp.array(high, dtype=float, device=self._device)
        else:
            high = float(high)
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
        if ids is not None:
            ids = wp.from_torch(ids.to(torch.int32), dtype=wp.int32)
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
        if ids is not None:
            ids = wp.array(ids.to(torch.int32), dtype=wp.int32, device=self._device)
        assert dtype in ["int", "float"], "The data type must be either 'int' or 'float'."
        output = self.sample_sign_warp(dtype, shape, ids)
        return output.numpy()

    # Rand int
    def sample_integer_warp(
        self, low: int | wp.array, high: int | wp.array, shape: tuple | int, ids: wp.array | None = None
    ) -> wp.array:
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
        self, low: int | torch.Tensor, high: int | torch.Tensor, shape: tuple | int, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample for a random integer. Torch implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is not None:
            ids = wp.from_torch(ids.to(torch.int32))
        if isinstance(low, torch.Tensor):
            low = wp.from_torch(low.to(torch.int32))
        else:
            low = int(low)
        if isinstance(high, torch.Tensor):
            high = wp.from_torch(high.to(torch.int32))
        else:
            high = int(high)
        output = self.sample_integer_warp(low, high, shape, ids)
        return wp.to_torch(output)

    def sample_integer_numpy(
        self, low: int | np.ndarray, high: int | np.ndarray, shape: tuple | int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample for a random integer. Numpy implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            np.ndarray: The sampled values."""
        if ids is not None:
            ids = wp.array(ids.to(torch.int32), dtype=wp.int32, device=self._device)
        if isinstance(low, np.ndarray):
            low = wp.array(low, dtype=wp.int32, device=self._device)
        else:
            low = int(low)
        if isinstance(high, np.ndarray):
            high = wp.array(high, dtype=wp.int32, device=self._device)
        else:
            high = int(high)
        output = self.sample_integer_warp(low, high, shape, ids)
        return output.numpy()

    def sample_normal_warp(
        self, mean: float | wp.array, std: float | wp.array, shape: tuple | int, ids: wp.array | None = None
    ) -> wp.array:
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
        self,
        mean: float | torch.Tensor,
        std: float | torch.Tensor,
        shape: tuple | int,
        ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from a normal distribution. Torch implementation.
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values."""
        if ids is not None:
            ids = wp.from_torch(ids.to(torch.int32), dtype=wp.int32)
        if isinstance(mean, torch.Tensor):
            mean = wp.from_torch(mean)
        else:
            mean = float(mean)
        if isinstance(std, torch.Tensor):
            std = wp.from_torch(std)
        else:
            std = float(std)
        output = self.sample_normal_warp(mean, std, shape, ids)
        return wp.to_torch(output)

    def sample_normal_numpy(
        self, mean: float | np.ndarray, std: float | np.ndarray, shape: tuple | int, ids: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample from a normal distribution. Numpy implementation.
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            np.ndarray: The sampled values."""
        if ids is not None:
            ids = wp.array(ids.to(torch.int32), dtype=wp.int32, device=self._device)
        if isinstance(mean, np.ndarray):
            mean = wp.array(mean, dtype=float, device=self._device)
        else:
            mean = float(mean)
        if isinstance(std, np.ndarray):
            std = wp.array(std, dtype=float, device=self._device)
        else:
            std = float(std)
        output = self.sample_normal_warp(mean, std, shape, ids)
        return output.numpy()

    def sample_poisson_warp(self, lam: float | wp.array, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
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

    def sample_poisson_torch(
        self, lam: float | torch.Tensor, shape: tuple | int, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample from a poisson distribution. Torch implementation.
        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is not None:
            ids = wp.from_torch(ids.to(torch.int32), dtype=wp.int32)
        if isinstance(lam, torch.Tensor):
            lam = wp.from_torch(lam)
        else:
            lam = float(lam)
        output = self.sample_poisson_warp(lam, shape, ids)
        return wp.to_torch(output)

    def sample_poisson_numpy(self, lam: float | np.ndarray, shape: tuple | int, ids: np.ndarray | None) -> np.ndarray:
        """Sample from a poisson distribution. Numpy implementation.
        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is not None:
            ids = wp.array(ids.to(torch.int32), dtype=wp.int32, device=self._device)
        if isinstance(lam, np.ndarray):
            lam = wp.array(lam, dtype=float, device=self._device)
        else:
            lam = float(lam)
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
        if ids is not None:
            ids = wp.from_torch(ids.to(torch.int32), dtype=wp.int32)
        output = self.sample_quaternion_warp(shape, ids)
        return wp.to_torch(output)

    def sample_quaternion_numpy(self, shape: tuple | int, ids: np.ndarray | None = None) -> np.ndarray:
        """Sample a quaternion. Numpy implementation.
        Args:
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            np.ndarray: The sampled values."""
        if ids is not None:
            ids = wp.array(ids.to(torch.int32), dtype=wp.int32, device=self._device)
        output = self.sample_quaternion_warp(shape, ids)
        return output.numpy()

    def sample_unique_integers_numpy(
        self, min: int | np.ndarray, max: int | np.ndarray, num: int, ids: np.ndarray | None = None
    ) -> np.array:
        """Sample unique integers. Numpy implementation, numpy backend.
        Args:
            min: The minimum value.
            max: The maximum value.
            num: The number of unique integers to sample.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values. Shape (num_envs, num)."""

        if self._numpy_rng_is_initialized is False:
            self.initialize_numpy_rng()

        if ids is None:
            ids = self._ALL_INDICES.numpy()

        if type(min) is np.ndarray:
            assert type(max) is np.ndarray, "min and max must have the same type"
            assert (min < max).all(), "min must be less than max"
            assert (num <= max - min).all(), "num must be less than or equal to max - min"
            outputs = [
                self._numpy_rngs[id].choice(np.arange(min[i], max[i]), num, replace=False) for i, id in enumerate(ids)
            ]

        elif type(min) is int:
            assert type(max) is int, "min and max must have the same type"
            assert min < max, "min must be less than max"
            assert num <= max - min, "num must be less than or equal to max - min"
            outputs = [self._numpy_rngs[id].choice(np.arange(min, max), num, replace=False) for id in ids]

        else:
            raise ValueError("min and max must be either int or np.ndarray")

        return np.array(outputs, dtype=np.int32)

    def sample_unique_integers_torch(
        self, min: int | torch.Tensor, max: int | torch.Tensor, num: int, ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample unique integers. Torch implementation, numpy backend.
        Args:
            min: The minimum value.
            max: The maximum value.
            num: The number of unique integers to sample.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values. Shape (num_envs, num)."""

        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()

        if isinstance(min, torch.Tensor):
            min = min.detach().cpu().numpy()
        if isinstance(max, torch.Tensor):
            max = max.detach().cpu().numpy()

        outputs = self.sample_unique_integers_numpy(min, max, num, ids)

        return torch.from_numpy(outputs).to(self._device)

    def sample_unique_integers_warp(
        self, min: int | wp.array, max: int | wp.array, num: int, ids: wp.array | None = None
    ) -> wp.array:
        """Sample unique integers. Warp implementation, numpy backend.
        Args:
            min: The minimum value.
            max: The maximum value.
            num: The number of unique integers to sample.
            ids: The ids of the environments.
        Returns:
            torch.Tensor: The sampled values. Shape (num_envs, num)."""

        if isinstance(ids, wp.array):
            ids = ids.numpy()

        if isinstance(min, wp.array):
            min = min.numpy()
        if isinstance(max, wp.array):
            max = max.numpy()

        outputs = self.sample_unique_integers_numpy(min, max, num, ids)

        return wp.array(outputs, dtype=wp.int32, device=self._device)
