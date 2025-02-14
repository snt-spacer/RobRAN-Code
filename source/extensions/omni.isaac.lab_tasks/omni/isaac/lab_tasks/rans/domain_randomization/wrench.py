# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_rotate_inverse

from omni.isaac.lab_tasks.rans.domain_randomization import RandomizationCore, RandomizationCoreCfg, Registerable
from omni.isaac.lab_tasks.rans.utils import PerEnvSeededRNG


@configclass
class WrenchRandomizationCfg(RandomizationCoreCfg):
    randomization_modes: list = ["none"]
    """The randomization modes to apply to the center of mass of the rigid bodies."""

    body_name: str | None = None
    """The name of the body to apply a force or torque onto. This body must be part of the asset. The asset is set by
    the caller of the randomization."""

    # 2D or 3D randomization
    force_dimension: int = 2
    """The dimension of the randomization. Can be 2D or 3D. If 2D, the randomization is applied in the x and y
    directions. If 3D, the randomization is applied in the x, y, and z directions."""
    torque_dimension: int = 1
    """The dimension of the randomization. Can be 1D or 3D. If 1D, the randomization is applied in the z direction. If
    3D, the randomization is applied in the x, y, and z directions."""

    # Uniform randomization
    uniform_force: tuple[float, float] = (0.0, 0.0)
    """The range within which we sample forces. (min_force, range). max_force = min_force + range. (in Newtons) Default
    is (0.0, 0.0). Note: The values must be positive."""
    uniform_torque: tuple[float, float] = (0.0, 0.0)
    """The range within which we sample torques. (min_torque, range). max_torque = min_torque + range.
    (in Newton-meters) Default is (0.0, 0.0). Note: The values must be positive."""

    # Normal randomization
    normal_force: tuple[float, float] = (0.0, 0.00001)
    """The normal distribution used to sample forces. (mean, std). Default is (0.0, 0.0001). Note: The values must
    be positive."""
    normal_torque: tuple[float, float] = (0.0, 0.00001)
    """The normal distribution used to sample torques. (mean, std). Default is (0.0, 0.0001). Note: The values must
    be positive."""

    # Sporadic wrenches
    kick_force_multiplier: float = 100.0
    """The multiplier for the force applied sporadically. Default is 1.0. This is used to increase or reduce the
    magnitude of the wrenches relative to the maximum force."""
    kick_torque_multiplier: float = 100.0
    """The multiplier for the torque applied sporadically. Default is 1.0. This is used to increase or reduce the
    magnitude of the wrenches relative to the maximum torque."""
    push_interval: int = 5
    """The interval at which the wrench is applied. (in seconds) Default is 5."""
    max_push_interval_variation: int = 1
    """The maximum variation in the interval at which the wrench is applied. (in seconds) Default is 1."""

    # Constant wrenches
    constant_force_multiplier: float = 1.0
    """The multiplier for the constant force applied. Default is 1.0. This is used to increase or reduce the
    magnitude of the wrenches relative to the maximum force."""
    constant_torque_multiplier: float = 1.0
    """The multiplier for the constant torque applied. Default is 1.0. This is used to increase or reduce the
    magnitude of the wrenches relative to the maximum torque."""
    use_sinusoidal_pattern: bool = False
    """Whether to use a sinusoidal pattern for the constant wrenches. Default is False. If True, the wrenches
    vary sinusoidally over time."""
    sine_wave_pattern: tuple[float, float] = (0.0, 2.0)
    """The parameters of the sinusoidal pattern. (frequency max, frequency min). Default is (0.0, 0.5)."""

    # Define the size of the generative space associated with the randomization
    gen_space: int = 2 if body_name is not None else 0  # DO NOT EDIT


class WrenchRandomization(Registerable, RandomizationCore):
    _cfg: WrenchRandomizationCfg

    def __init__(
        self,
        cfg: WrenchRandomizationCfg,
        rng: PerEnvSeededRNG,
        scene: InteractiveScene,
        asset_name: str = "MISSING",
        num_envs: int = 1,
        device: str = "cuda",
        **kwargs
    ) -> None:
        """Random wrench application class.

        Args:
            cfg: The configuration for the randomization.
            rng: The random number generator.
            scene: The scene to get the assets from.
            asset_name: The name of the asset to randomize.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        super().__init__(cfg, rng, scene, num_envs, device)
        self._asset_name = asset_name
        self._cfg.push_interval = int(self._scene.physics_dt * self._cfg.push_interval)
        self._cfg.max_push_interval_variation = int(self._scene.physics_dt * self._cfg.max_push_interval_variation)

    @property
    def data(self) -> dict[str, torch.Tensor]:
        data = {
            "constant_force": self._constant_force,
            "constant_torque": self._constant_torque,
            "kick_force": self._kick_force,
            "kick_torque": self._kick_torque,
        }
        return data

    def data_shape(self) -> dict[str, tuple[int]]:
        shape = {
            "constant_force": (3,),
            "constant_torque": (3,),
            "kick_force": (3,),
            "kick_torque": (3,),
        }
        return shape

    def check_cfg(self) -> None:
        """Check the configuration of the randomization."""

        super().check_cfg()

        if "uniform" in self._cfg.randomization_modes:
            assert "normal" not in self._cfg.randomization_modes, "The 'uniform' and 'normal' modes cannot be combined."

        if "normal" in self._cfg.randomization_modes:
            assert (
                "uniform" not in self._cfg.randomization_modes
            ), "The 'uniform' and 'normal' modes cannot be combined."

    def pre_setup(self) -> None:
        """Setup the wrench randomization."""

        super().pre_setup()

        if self._cfg.enable:
            self._asset: Articulation | RigidObject = self._scene[self._asset_name]
            self._body_id, _ = self._asset.find_bodies(self._cfg.body_name)

            # Initialize the wrenches
            self._kick_force = torch.zeros((self._num_envs, 3), device=self._device)
            self._kick_torque = torch.zeros((self._num_envs, 3), device=self._device)
            self._constant_force = torch.zeros((self._num_envs, 3), device=self._device)
            self._constant_torque = torch.zeros((self._num_envs, 3), device=self._device)
            self._bool_to_kick = torch.zeros((self._num_envs), device=self._device, dtype=torch.bool)

    def default_reset(self, env_ids: torch.Tensor | None, **kwargs) -> None:
        """The default reset function for the randomization."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # By default, the mass is set to the default mass.
        self._constant_force[env_ids].fill_(0.0)
        self._constant_torque[env_ids].fill_(0.0)
        self._kick_force[env_ids].fill_(0.0)
        self._kick_torque[env_ids].fill_(0.0)
        self._bool_to_kick[env_ids].fill_(False)

    def default_update(self, **kwargs) -> None:
        """The default update function for the randomization."""

        # Do nothing
        pass

    def fn_on_setup_kick_uniform(self) -> None:
        """Setup the uniform kick randomization."""

        self._kick_time = torch.zeros((self._num_envs,), dtype=torch.int32, device=self._device)
        self._kick_force_scale = torch.zeros((self._num_envs,), device=self._device)
        self._kick_torque_scale = torch.zeros((self._num_envs,), device=self._device)

    def fn_on_reset_kick_uniform(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Samples a next time step to apply the kick."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Use the gen actions as a scaling factor for the force and torque
        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (2,), env_ids)
        self._kick_force_scale[env_ids] = gen_actions[:, 0] * self._cfg.kick_force_multiplier
        self._kick_torque_scale[env_ids] = gen_actions[:, 1] * self._cfg.kick_torque_multiplier

        # Reset the kick time
        self._kick_time[env_ids] = self._rng.sample_integer_torch(
            self._cfg.push_interval - self._cfg.max_push_interval_variation,
            self._cfg.push_interval + self._cfg.max_push_interval_variation,
            (1,),
            env_ids,
        )

    def fn_on_update_kick_uniform(self, **kwargs) -> None:
        """Updates the kick force and torque."""

        self._kick_time -= 1

        # Collect the indices of the bodies to kick. Saving bool_to_kick as a class variable to avoid setting it to zero
        # every time after the update.
        self._bool_to_kick = self._kick_time <= 0
        idx_to_kick = torch.argwhere(self._bool_to_kick).flatten()

        # If there are any bodies to kick
        if idx_to_kick.shape[0] > 0:
            # Compute the norm of the force and torque
            kick_force = self._kick_force_scale[idx_to_kick] * self._rng.sample_uniform_torch(
                self._cfg.uniform_force[0], self._cfg.uniform_force[1], (1,), idx_to_kick
            )
            kick_torque = self._kick_torque_scale[idx_to_kick] * self._rng.sample_uniform_torch(
                self._cfg.uniform_torque[0], self._cfg.uniform_torque[1], (1,), idx_to_kick
            )

            # Project the force and torque in the 2D or 3D space
            theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=idx_to_kick)
            if self._cfg.force_dimension == 2:  # 2D
                self._kick_force[self._bool_to_kick, 0] = torch.cos(theta) * kick_force
                self._kick_force[self._bool_to_kick, 1] = torch.sin(theta) * kick_force
            else:  # 3D
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=idx_to_kick)
                self._kick_force[self._bool_to_kick, 0] = torch.cos(theta) * torch.sin(phi) * kick_force
                self._kick_force[self._bool_to_kick, 1] = torch.sin(theta) * torch.sin(phi) * kick_force
                self._kick_force[self._bool_to_kick, 2] = torch.cos(phi) * kick_force

            if self._cfg.torque_dimension == 1:  # 2D
                self._kick_torque[self._bool_to_kick, 2] = kick_torque
            else:  # 3D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=idx_to_kick)
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=idx_to_kick)
                self._kick_torque[self._bool_to_kick, 0] = torch.cos(theta) * torch.sin(phi) * kick_torque
                self._kick_torque[self._bool_to_kick, 1] = torch.sin(theta) * torch.sin(phi) * kick_torque
                self._kick_torque[self._bool_to_kick, 2] = torch.cos(phi) * kick_torque

            # Reset the kick time
            self._kick_time[self._bool_to_kick] = self._rng.sample_integer_torch(
                self._cfg.push_interval - self._cfg.max_push_interval_variation,
                self._cfg.push_interval + self._cfg.max_push_interval_variation,
                (1,),
                idx_to_kick,
            )

    def fn_on_setup_kick_normal(self) -> None:
        """Setup the normal kick randomization."""

        self._kick_time = torch.zeros((self._num_envs), dtype=torch.int32, device=self._device)
        self._kick_force_scale = torch.zeros((self._num_envs,), device=self._device)
        self._kick_torque_scale = torch.zeros((self._num_envs,), device=self._device)

    def fn_on_reset_kick_normal(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Samples a next time step to apply the kick."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Use the gen actions as a scaling factor for the force and torque.
        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (2,), env_ids)
        self._kick_force_scale[env_ids] = gen_actions[:, 0] * self._cfg.kick_force_multiplier
        self._kick_torque_scale[env_ids] = gen_actions[:, 1] * self._cfg.kick_torque_multiplier

        # Reset the kick time
        self._kick_time[env_ids] = self._rng.sample_integer_torch(
            self._cfg.push_interval - self._cfg.max_push_interval_variation,
            self._cfg.push_interval + self._cfg.max_push_interval_variation,
            (1,),
            env_ids,
        )

    def fn_on_update_kick_normal(self, **kwargs) -> None:
        """Updates the kick force and torque."""

        self._kick_time -= 1

        # Collect the indices of the bodies to kick. Saving bool_to_kick as a class variable to avoid setting it to zero
        # every time after the update.
        self._bool_to_kick = self._kick_time <= 0
        idx_to_kick = torch.argwhere(self._bool_to_kick).flatten()

        # If there are any bodies to kick
        if idx_to_kick.shape[0] > 0:
            # Compute the norm of the force and torque
            kick_force = self._kick_force_scale[idx_to_kick] * self._rng.sample_normal_torch(
                self._cfg.normal_force[0], self._cfg.normal_force[1], (1,), idx_to_kick
            )
            kick_torque = self._kick_torque_scale[idx_to_kick] * self._rng.sample_normal_torch(
                self._cfg.normal_torque[0], self._cfg.normal_torque[1], (1,), idx_to_kick
            )

            theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=idx_to_kick)
            if self._cfg.force_dimension == 2:  # 2D
                self._kick_force[self._bool_to_kick, 0] = torch.cos(theta) * kick_force
                self._kick_force[self._bool_to_kick, 1] = torch.sin(theta) * kick_force
            else:  # 3D
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=idx_to_kick)
                self._kick_force[self._bool_to_kick, 0] = torch.cos(theta) * torch.sin(phi) * kick_force
                self._kick_force[self._bool_to_kick, 1] = torch.sin(theta) * torch.sin(phi) * kick_force
                self._kick_force[self._bool_to_kick, 2] = torch.cos(phi) * kick_force
            if self._cfg.torque_dimension == 1:  # 2D
                self._kick_torque[self._bool_to_kick, 2] = kick_torque
            else:  # 3D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=idx_to_kick)
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=idx_to_kick)
                self._kick_torque[self._bool_to_kick, 0] = torch.cos(theta) * torch.sin(phi) * kick_torque
                self._kick_torque[self._bool_to_kick, 1] = torch.sin(theta) * torch.sin(phi) * kick_torque
                self._kick_torque[self._bool_to_kick, 2] = torch.cos(phi) * kick_torque

            # Reset the kick time
            self._kick_time[self._bool_to_kick] = self._rng.sample_integer_torch(
                self._cfg.push_interval - self._cfg.max_push_interval_variation,
                self._cfg.push_interval + self._cfg.max_push_interval_variation,
                (1,),
                idx_to_kick,
            )

    def fn_on_setup_constant_uniform(self) -> None:
        """Setup the uniform constant randomization."""

        self._constant_force_scale = torch.zeros((self._num_envs,), device=self._device)
        self._constant_torque_scale = torch.zeros((self._num_envs,), device=self._device)

        if self._cfg.use_sinusoidal_pattern:
            self._force_sin_wave_freq = torch.zeros((self._num_envs, 2), device=self._device)
            self._torque_sin_wave_freq = torch.zeros((self._num_envs, 2), device=self._device)
            self._force_sin_wave_freq_offset = torch.zeros((self._num_envs, 2), device=self._device)
            self._torque_sin_wave_freq_offset = torch.zeros((self._num_envs, 2), device=self._device)

    def fn_on_reset_constant_uniform(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Samples the constant force and torque."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Use the gen actions as a scaling factor for the force and torque.
        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (2,), env_ids)
        # Scale like so F = K_F * U[min, max] * gen_actions, where U[min, max] is a uniform distribution and K_F is a multiplier
        self._constant_force_scale = (
            self._cfg.constant_force_multiplier
            * self._rng.sample_uniform_torch(self._cfg.uniform_force[0], self._cfg.uniform_force[1], (1,), ids=env_ids)
            * gen_actions[:, 0]
        )
        self._constant_torque_scale = (
            self._cfg.constant_torque_multiplier
            * self._rng.sample_uniform_torch(
                self._cfg.uniform_torque[0], self._cfg.uniform_torque[1], (1,), ids=env_ids
            )
            * gen_actions[:, 1]
        )

        # Set the sinusoidal pattern offsets and frequencies
        if self._cfg.use_sinusoidal_pattern:
            # Generate a set of sinusoids with different frequencies
            self._force_sin_wave_freq = self._rng.sample_uniform_torch(
                self._cfg.sine_wave_pattern[0], self._cfg.sine_wave_pattern[1], (2,), env_ids
            )
            self._torque_sin_wave_freq = self._rng.sample_uniform_torch(
                self._cfg.sine_wave_pattern[0], self._cfg.sine_wave_pattern[1], (2,), env_ids
            )
            # From these sinusoids pick a random frequency offset
            self._force_sin_wave_freq_offset = self._rng.sample_uniform_torch(
                torch.zeros_like(self._force_sin_wave_freq), self._force_sin_wave_freq * 2, (2,), env_ids
            )
            self._torque_sin_wave_freq_offset = self._rng.sample_uniform_torch(
                torch.zeros_like(self._torque_sin_wave_freq), self._torque_sin_wave_freq * 2, (2,), env_ids
            )
        else:
            # If not using a sinusoidal pattern, generate a random force and torque. They are constant across the
            # episode so they can be computed on reset.
            theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)

            if self._cfg.force_dimension == 2:
                self._constant_force[env_ids, 0] = torch.cos(theta) * self._constant_force_scale
                self._constant_force[env_ids, 1] = torch.sin(theta) * self._constant_force_scale
            else:
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=env_ids)
                self._constant_force[env_ids, 0] = self._constant_force_scale * torch.cos(theta) * torch.sin(phi)
                self._constant_force[env_ids, 1] = self._constant_force_scale * torch.sin(theta) * torch.sin(phi)
                self._constant_force[env_ids, 2] = self._constant_force_scale * torch.cos(phi)
            if self._cfg.torque_dimension == 1:
                self._constant_torque[env_ids, 2] = self._constant_torque_scale
            else:
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=env_ids)
                self._constant_torque[env_ids, 0] = self._constant_torque_scale * torch.cos(theta) * torch.sin(phi)
                self._constant_torque[env_ids, 1] = self._constant_torque_scale * torch.sin(theta) * torch.sin(phi)
                self._constant_torque[env_ids, 2] = self._constant_torque_scale * torch.cos(phi)

    def fn_on_update_constant_uniform(self, dt: float = 0.0, **kwargs) -> None:
        """Updates the constant force and torque."""

        if self._cfg.use_sinusoidal_pattern:
            # Apply the sinusoidal pattern to the force and torque
            force = self._constant_force_scale * torch.sin(
                self._force_sin_wave_freq_offset[..., 0] + self._force_sin_wave_freq[..., 0] * dt
            )
            torque = self._constant_torque_scale * torch.sin(
                self._torque_sin_wave_freq_offset[..., 1] + self._torque_sin_wave_freq[..., 1] * dt
            )
            # Project the force and torque in the 2D or 3D space
            if self._cfg.force_dimension == 2:  # 2D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,))
                self._constant_force[:, 0] = force * torch.cos(theta)
                self._constant_force[:, 1] = force * torch.sin(theta)
            else:  # 3D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,))
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,))
                self._constant_force[:, 0] = force * torch.cos(theta) * torch.sin(phi)
                self._constant_force[:, 1] = force * torch.sin(theta) * torch.sin(phi)
                self._constant_force[:, 2] = force * torch.cos(phi)
            if self._cfg.torque_dimension == 1:  # 2D
                self._constant_torque[:, 2] = torque
            else:  # 3D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,))
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,))
                self._constant_torque[:, 0] = torque * torch.cos(theta) * torch.sin(phi)
                self._constant_torque[:, 1] = torque * torch.sin(theta) * torch.sin(phi)
                self._constant_torque[:, 2] = torque * torch.cos(phi)

    def fn_on_setup_constant_normal(self) -> None:
        """Setup the uniform constant randomization.
        This mode is useful to emulate a constant force and or torque applied to an object. If the use_sinuoidal_pattern
        is set to True, the force and torque will vary sinusoidally over time. This can be used to emulate a periodic
        disturbance like a wind gust or a vibrations, or current in a fluid."""

        self._constant_force_scale = torch.zeros((self._num_envs,), device=self._device)
        self._constant_torque_scale = torch.zeros((self._num_envs,), device=self._device)

        if self._cfg.use_sinusoidal_pattern:
            self._force_sin_wave_freq = torch.zeros((self._num_envs, 2), device=self._device)
            self._torque_sin_wave_freq = torch.zeros((self._num_envs, 2), device=self._device)
            self._force_sin_wave_freq_offset = torch.zeros((self._num_envs, 2), device=self._device)
            self._torque_sin_wave_freq_offset = torch.zeros((self._num_envs, 2), device=self._device)

    def fn_on_reset_constant_normal(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Samples the constant force and torque."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # Use the gen actions as a scaling factor for the force and torque.
        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (2,), env_ids)
        # Scale like so F = K_F * N[mean, std] * gen_actions, where U[min, max] is a normal distribution and K_F is a multiplier
        self._constant_force_scale = (
            self._cfg.constant_force_multiplier
            * self._rng.sample_normal_torch(self._cfg.normal_force[0], self._cfg.normal_force[1], (1,), ids=env_ids)
            * gen_actions[:, 0]
        )
        self._constant_torque_scale = (
            self._cfg.constant_torque_multiplier
            * self._rng.sample_normal_torch(self._cfg.normal_torque[0], self._cfg.normal_torque[1], (1,), ids=env_ids)
            * gen_actions[:, 1]
        )

        # Set the sinusoidal pattern offsets and frequencies
        if self._cfg.use_sinusoidal_pattern:
            # Generate a set of sinusoids with different frequencies
            self._force_sin_wave_freq = self._rng.sample_uniform_torch(
                self._cfg.sine_wave_pattern[0], self._cfg.sine_wave_pattern[1], (2,), env_ids
            )
            self._torque_sin_wave_freq = self._rng.sample_uniform_torch(
                self._cfg.sine_wave_pattern[0], self._cfg.sine_wave_pattern[1], (2,), env_ids
            )
            # From these sinusoids pick a random frequency offset
            self._force_sin_wave_freq_offset = self._rng.sample_uniform_torch(
                torch.zeros_like(self._force_sin_wave_freq), self._force_sin_wave_freq * 2, (2,), env_ids
            )
            self._torque_sin_wave_freq_offset = self._rng.sample_uniform_torch(
                torch.zeros_like(self._torque_sin_wave_freq), self._torque_sin_wave_freq * 2, (2,), env_ids
            )
        else:
            # If not using a sinusoidal pattern, generate a random force and torque. They are constant across the
            # episode so they can be computed on reset.
            theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
            if self._cfg.force_dimension == 2:
                self._constant_force[env_ids, 0] = self._constant_force_scale * torch.cos(theta)
                self._constant_force[env_ids, 1] = self._constant_force_scale * torch.sin(theta)
            else:
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=env_ids)
                self._constant_force[env_ids, 0] = self._constant_force_scale * torch.cos(theta) * torch.sin(phi)
                self._constant_force[env_ids, 1] = self._constant_force_scale * torch.sin(theta) * torch.sin(phi)
                self._constant_force[env_ids, 2] = self._constant_force_scale * torch.cos(phi)
            if self._cfg.torque_dimension == 1:
                self._constant_torque[env_ids, 2] = self._constant_torque_scale
            else:
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,), ids=env_ids)
                self._constant_torque[env_ids, 0] = self._constant_torque_scale * torch.cos(theta) * torch.sin(phi)
                self._constant_torque[env_ids, 1] = self._constant_torque_scale * torch.sin(theta) * torch.sin(phi)
                self._constant_torque[env_ids, 2] = self._constant_torque_scale * torch.cos(phi)

    def fn_on_update_constant_normal(self, dt: float = 0.0, **kwargs) -> None:
        """Updates the constant force and torque."""

        if self._cfg.use_sinusoidal_pattern:
            # Apply the sinusoidal pattern to the force and torque
            force = self._constant_force_scale * torch.sin(
                self._force_sin_wave_freq_offset[..., 0] + self._force_sin_wave_freq[..., 0] * dt
            )
            torque = self._constant_torque_scale * torch.sin(
                self._torque_sin_wave_freq_offset[..., 1] + self._torque_sin_wave_freq[..., 1] * dt
            )
            # Project the force and torque in the 2D or 3D space
            if self._cfg.force_dimension == 2:  # 2D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,))
                self._constant_force[:, 0] = force * torch.cos(theta)
                self._constant_force[:, 1] = force * torch.sin(theta)
            else:  # 3D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,))
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,))
                self._constant_force[:, 0] = force * torch.cos(theta) * torch.sin(phi)
                self._constant_force[:, 1] = force * torch.sin(theta) * torch.sin(phi)
                self._constant_force[:, 2] = force * torch.cos(phi)
            if self._cfg.torque_dimension == 1:  # 2D
                self._constant_torque[:, 2] = torque
            else:  # 3D
                theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,))
                phi = self._rng.sample_uniform_torch(-torch.pi / 2, torch.pi / 2, (1,))
                self._constant_torque[:, 0] = torque * torch.cos(theta) * torch.sin(phi)
                self._constant_torque[:, 1] = torque * torch.sin(theta) * torch.sin(phi)
                self._constant_torque[:, 2] = torque * torch.cos(phi)

    def apply_randomization(self, ids: torch.Tensor | None = None) -> None:
        """Updates the mass of the robot. The mass is in kg.

        Args:
            ids: The ids of the robot."""

        if ids is None:
            ids = self._ALL_INDICES

        # Apply the constant force and torque in the global frame
        forces = self._constant_force[ids] + self._kick_force[ids]
        projected_forces = quat_rotate_inverse(self._asset.data.root_com_quat_w[ids], forces).unsqueeze(1)
        torques = self._constant_torque[ids] + self._kick_torque[ids]
        if self._cfg.torque_dimension != 1:
            projected_torques = quat_rotate_inverse(self._asset.data.root_com_quat_w[ids], torques).unsqueeze(1)
        else:
            projected_torques = torques.unsqueeze(1)

        # Apply the forces and torques
        self._asset.set_external_force_and_torque(
            projected_forces, projected_torques, body_ids=self._body_id, env_ids=ids
        )

        # Reset the kick force and torque if necessary
        if self._bool_to_kick.any():
            self._bool_to_kick[self._bool_to_kick] = False
