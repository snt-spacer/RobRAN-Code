# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import RandomizationCore, RandomizationCoreCfg, Registerable
from isaaclab_tasks.rans.utils import PerEnvSeededRNG


@configclass
class InertiaRandomizationCfg(RandomizationCoreCfg):
    enable: bool = False
    """Whether to enable the randomization or not."""

    randomization_modes: list = ["none"]
    """The randomization modes to apply to the inertia of the rigid body."""

    body_name: str | None = None
    """The name of the body to randomize the inertia of. This body must be part of the asset. The asset is set by
    the caller of the randomization."""

    max_delta: float = 0.00001
    """The maximum delta to apply to the inertia of the rigid body. (in meters) Default is 0.0001."""

    std: float = 0.00001
    """The standard deviation of the normal distribution to sample the delta from. (in meters) Default is 0.0001."""

    decay_rate: float = 0.0
    """The decay rate of the mass. Default is 0.0. (no decay)"""

    # Define the size of the generative space associated with the randomization
    gen_space: int = 1 if enable else 0  # DO NOT EDIT


class InertiaRandomization(Registerable, RandomizationCore):
    _cfg: InertiaRandomizationCfg

    def __init__(
        self,
        cfg: InertiaRandomizationCfg,
        rng: PerEnvSeededRNG,
        scene: InteractiveScene,
        asset_name: str = "MISSING",
        num_envs: int = 1,
        device: str = "cuda",
        **kwargs
    ):
        """Inertia randomization class.

        Args:
            cfg: The configuration for the randomization.
            rng: The random number generator.
            scene: The scene to get the assets from.
            asset_name: The name of the asset to randomize.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        super().__init__(cfg, rng, scene, num_envs, device)
        self._asset_name = asset_name

    @property
    def data(self) -> dict[str, torch.Tensor]:
        """The current inertia of the rigid body."""
        super().data
        return {"inertia": self._current_inertia[self._ALL_INDICES, self._body_id]}

    @property
    def data_shape(self) -> dict[str, tuple[int]]:
        super().data
        return {"inertia": (9,)}

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
        """Setup the inertia randomization."""

        super().pre_setup()

        if self._cfg.enable:
            self._asset: Articulation | RigidObject = self._scene[self._asset_name]
            self._body_id, _ = self._asset.find_bodies(self._cfg.body_name)
            self._default_inertia: torch.Tensor = self._asset.root_physx_view.get_inertias().to(self._device)
            self._current_inertia = self._default_inertia.clone()

    def default_reset(self, env_ids: torch.Tensor | None, **kwargs) -> None:
        """The default reset function for the randomization."""

        if env_ids is None:
            env_ids = self._ALL_INDICES

        # By default, the inertia is set to the default inertia.
        self._current_inertia[env_ids, self._body_id] = self._default_inertia[env_ids, self._body_id]

    def default_update(self, **kwargs) -> None:
        """The default update function for the randomization."""

        # Do nothing
        pass

    def fn_on_reset_uniform(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ):
        """Randomize the inertia of the rigid bodies uniformly. The function multiplies the initial inertia by a factor.
        Only the diagonal elements are modified.

        Note: The generative actions are given in the range [0, 1] and are scaled to the range [-1, 1].

        Args:
            env_ids: The ids of the environments.
            gen_actions: The actions taken by the agent."""

        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (1,), env_ids)

        if env_ids is None:
            env_ids = self._ALL_INDICES

        theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
        phi = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)

        # Only the diagonal elements are modified
        self._current_inertia[env_ids, self._body_id, 0] *= 1 + self._cfg.max_delta * gen_actions * torch.sin(
            theta
        ) * torch.cos(phi)
        self._current_inertia[env_ids, self._body_id, 4] *= 1 + self._cfg.max_delta * gen_actions * torch.sin(
            theta
        ) * torch.sin(phi)
        self._current_inertia[env_ids, self._body_id, 8] *= 1 + self._cfg.max_delta * gen_actions * torch.cos(theta)

    def fn_on_reset_normal(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ):
        """Randomize the inertia of the rigid bodies  by following a normal distribution. The function multiplies the
        initial inertia by a factor. Only the diagonal elements are modified.

        Note: The generative actions are given in the range [0, 1] and are scaled to the range [-1, 1].

        Args:
            env_ids: The ids of the environments.
            gen_actions: The actions taken by the agent."""

        if gen_actions is None:
            gen_actions = self._rng.sample_normal_torch(0, 1, (1,), env_ids)

        if env_ids is None:
            env_ids = self._ALL_INDICES

        normal = gen_actions * self._cfg.std
        theta = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)
        phi = self._rng.sample_uniform_torch(-torch.pi, torch.pi, (1,), ids=env_ids)

        # Only the diagonal elements are modified
        self._current_inertia[env_ids, self._body_id, 0] *= 1 + normal * torch.sin(theta) * torch.cos(phi)
        self._current_inertia[env_ids, self._body_id, 4] *= 1 + normal * torch.sin(theta) * torch.sin(phi)
        self._current_inertia[env_ids, self._body_id, 8] *= 1 + normal * torch.cos(theta)

    def apply_randomization(self, ids: torch.Tensor | None = None) -> None:
        """Updates the inertia of the robot.

        Args:
            ids: The ids of the robot."""

        if ids is None:
            ids_cpu = self._ALL_INDICES_CPU
        else:
            ids_cpu = ids.to("cpu")

        self._asset.root_physx_view.set_inertias(self._current_inertia.to("cpu"), ids_cpu)
