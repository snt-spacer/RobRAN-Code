import numpy as np
import warp as wp
import torch


@wp.func
def set_state(seed: wp.int32) -> wp.uint32:
    """Set the seed for the random number generator."""
    return wp.rand_init(seed)


@wp.kernel
def set_states(
    seed: wp.array(dtype=wp.int32),
    old_seeds: wp.array(dtype=wp.int32),
    states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
):
    """Set the seeds for the random number generator."""
    tid = wp.tid()
    states[ids[tid]] = set_state(seed[tid])
    old_seeds[ids[tid]] = seed[tid]


@wp.kernel
def rand_uniform_1D(
    low: wp.float32,
    high: wp.float32,
    states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Sample from a uniform distribution."""
    tid = wp.tid()
    output[tid] = wp.randf(states[ids[tid]], low, high)
    states[ids[tid]] = states[ids[tid]] + wp.uint32(1)


@wp.kernel
def rand_uniform_2D(
    low: wp.float32,
    high: wp.float32,
    states: wp.array(dtype=wp.uint32),
    ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32, ndim=2),
    offset: wp.uint32,
):
    """Sample from a uniform distribution."""
    i, j = wp.tid()
    output[i][j] = wp.randf(states[ids[i]] + wp.uint32(j), low, high)
    states[ids[i]] = states[ids[i]] + wp.uint32(offset)


#
#
# @wp.kernel
# def rand_uniform_3D(
#    low: wp.float32,
#    high: wp.float32,
#    states: wp.array(dtype=wp.uint32),
#    ids: wp.array(dtype=wp.uint32),
#    output: wp.array(dtype=float, ndim=3),
# ):
#    """Sample from a uniform distribution."""
#    i, j, k = wp.tid()
#    output[i][j][k] = wp.randf(states[ids[i]], low, high)


class PerEnvSeededRNG:
    def __init__(self, seeds: int | torch.Tensor, num_envs: int, device: str):
        """Initialize the random number generator."""

        self._device = device
        self._num_envs = num_envs

        # Instantiate buffers
        if isinstance(seeds, int):
            self._seeds = wp.array(np.ones(num_envs) * seeds, dtype=wp.int32, device=device)
        else:
            self._seeds = wp.from_torch(seeds, dtype=wp.int32, device=device)

        self._states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._ALL_INDICES = wp.array(np.arange(num_envs), dtype=wp.uint32, device=device)

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
        return self._seeds.cpu().numpy()

    def set_seeds_warp(self, seeds: wp.array, ids: wp.array | None):
        """Set the seeds for each environment.

        Args:
            seeds (int32): The seeds for each environment.
            ids (uint32): The ids of the environments.
        """

        if ids is None:
            ids = self._ALL_INDICES

        num_instances = len(seeds)
        wp.launch(
            kernel=set_states,
            dim=num_instances,
            inputs=[seeds, self._seeds, self._states, ids],
            device=self._device,
        )

    def set_seeds(self, seeds: torch.Tensor, ids: torch.Tensor | None):
        """Set the seeds for each environment.

        Args:
            seeds (torch.Tensor): The seeds for each environment.
            ids (torch.Tensor): The ids of the environments.
        """

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

    def set_seeds_numpy(self, seeds: np.ndarray, ids: np.ndarray | None):
        """Set the seeds for each environment.

        Args:
            seeds (np.ndarray): The seeds for each environment.
            ids (np.ndarray): The ids of the environments.
        """

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

    def sample_uniform(self, low: float, high: float, shape: tuple | int, ids: torch.Tensor | None) -> torch.Tensor:
        """Sample from a uniform distribution.

        Args:
            low (float): The lower bound of the distribution.
            high (float): The upper bound of the distribution.
            shape (Tuple[int, int]): The shape of the output tensor.

        Returns:
            torch.Tensor: The sampled values.

        Raises:
            RuntimeError: If the shape is greater than 3D.
        """
        ids = wp.from_torch(ids, dtype=wp.int32)
        num_instances = len(ids)

        if isinstance(shape, int):  # Either 1D or 2D
            if shape == 1:  # 1D
                output = wp.zeros((num_instances), dtype=wp.float32, device=self._device)
                wp.launch(
                    kernel=rand_uniform_1D,
                    dim=self._num_envs,
                    inputs=[low, high, self._states, ids, output],
                    device=self._device,
                )
            else:  # 2D
                output = wp.zeros((num_instances, shape), dtype=wp.float32, device=self._device)
                wp.launch(
                    kernel=rand_uniform_2D,
                    dim=(self._num_envs, shape),
                    inputs=[low, high, self._states, ids, output, shape],
                    device=self._device,
                )
        elif len(shape) == 1:  # 2D
            output = wp.zeros((num_instances, shape[0]), dtype=wp.float32, device=self._device)
            wp.launch(
                kernel=rand_uniform_2D,
                dim=(self._num_envs, shape[0]),
                inputs=[low, high, self._states, ids, output, shape[0]],
                device=self._device,
            )
        elif len(shape) == 2:  # 3D
            output = wp.zeros((num_instances, shape[0], shape[1]), dtype=wp.float32, device=self._device)
            wp.launch(
                kernel=rand_uniform_3D,
                dim=(self._num_envs, shape[0], shape[1]),
                inputs=[low, high, self._states, ids, output],
                device=self._device,
            )
        else:
            raise RuntimeError("Cannot generate random numbers with shape greater than 3D.")

        return wp.to_torch(output)

    # Missing stuff:
    # Rand sign (return -1 or 1 with a specific type float or int)
    # Rand int
    # Rand normal
    # Rand poisson
    # test + doc
    # Rand quaternion (sample on sphere)


if __name__ == "__main__":
    wp.init()
    PESRNG = PerEnvSeededRNG(42, 1000, "cuda")
    print("Seeds:")
    print(PESRNG._seeds)
    print("States:")
    print(PESRNG._states)
    PESRNG.set_seeds(
        torch.arange(1000, dtype=torch.int32, device="cuda"),
        torch.arange(1000, dtype=torch.int32, device="cuda"),  # .to(torch.uint32),
    )
    print("Seeds:")
    print(PESRNG._seeds)
    print("States:")
    print(PESRNG._states)
    out = PESRNG.sample_uniform(0.0, 1.0, 1, torch.arange(1000, dtype=torch.int32, device="cuda"))
    print("1D Values:")
    print(out[:10])
    out = PESRNG.sample_uniform(0.0, 1.0, 1, torch.arange(1000, dtype=torch.int32, device="cuda"))
    print("1D Values:")
    print(out[:10])
    out = PESRNG.sample_uniform(0.0, 1.0, (3), torch.arange(1000, dtype=torch.int32, device="cuda"))
    print("2D Values:")
    print(out[:10])
    print("2D shape:", out.shape)
    out = PESRNG.sample_uniform(0.0, 1.0, (3), torch.arange(1000, dtype=torch.int32, device="cuda"))
    print("2D Values:")
    print(out[:10])
    print("2D shape:", out.shape)
