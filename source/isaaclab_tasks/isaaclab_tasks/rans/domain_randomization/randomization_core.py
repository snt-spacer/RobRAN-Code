# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass

from isaaclab_tasks.rans.utils import PerEnvSeededRNG


@configclass
class RandomizationCoreCfg:
    enable: bool = False
    """Whether to enable the randomization or not."""

    randomization_modes: list = []
    """The randomization modes to apply to the center of mass of the rigid bodies."""


class RandomizationCore:
    def __init__(
        self,
        cfg: RandomizationCoreCfg,
        rng: PerEnvSeededRNG,
        scene: InteractiveScene,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        """Core randomization class.

        Args:
            cfg: The configuration for the randomization.
            rng: The random number generator.
            scene: The scene to get the assets from."""

        self._rng = rng
        self._cfg = cfg
        self._scene = scene
        self._num_envs = num_envs
        self._device = device

    @property
    def data(self) -> dict:
        """Get the data of the randomization."""

        return {}

    @property
    def data_shape(self) -> dict:
        """Get the shape of the data of the randomization."""

        return {}

    def pre_setup(self) -> None:
        # Automatically collects the methods available for randomization
        self.modes = [fn[12:] for fn in self.__dir__() if (fn[:11] == "fn_on_reset")]
        self.modes += [fn[13:] for fn in self.__dir__() if (fn[:12] == "fn_on_update")]
        self.modes += [fn[14:] for fn in self.__dir__() if (fn[:13] == "fn_on_actions")]
        self.modes += [fn[18:] for fn in self.__dir__() if (fn[:17] == "fn_on_observation")]
        self.modes = list(set(self.modes))  # Remove duplicates
        self.on_reset_fns = {
            mode: getattr(self, "fn_on_reset_" + mode) for mode in self.modes if "fn_on_reset_" + mode in self.__dir__()
        }
        self.on_update_fns = {
            mode: getattr(self, "fn_on_update_" + mode)
            for mode in self.modes
            if "fn_on_update_" + mode in self.__dir__()
        }
        self.on_setup_fns = {
            mode: getattr(self, "fn_on_setup_" + mode) for mode in self.modes if "fn_on_setup_" + mode in self.__dir__()
        }
        self.on_actions_fns = {
            mode: getattr(self, "fn_on_actions_" + mode)
            for mode in self.modes
            if "fn_on_actions_" + mode in self.__dir__()
        }
        self.on_observations_fns = {
            mode: getattr(self, "fn_on_observations_" + mode)
            for mode in self.modes
            if "fn_on_observations_" + mode in self.__dir__()
        }
        # Generate flags to inform the low-level code if the randomization is active
        self.update_on_reset = any([mode in self._cfg.randomization_modes for mode in self.on_reset_fns.keys()])
        self.update_on_update = any([mode in self._cfg.randomization_modes for mode in self.on_update_fns.keys()])
        self.update_on_actions = any([mode in self._cfg.randomization_modes for mode in self.on_actions_fns.keys()])
        self.update_on_observations = any(
            [mode in self._cfg.randomization_modes for mode in self.on_observations_fns.keys()]
        )
        # Run cfg checks
        self.check_cfg()
        # Build all indices
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device)
        self._ALL_INDICES_CPU = torch.arange(self._num_envs, device="cpu")

    def check_cfg(self) -> None:
        """Check the configuration of the randomization."""

        for mode in self._cfg.randomization_modes:
            assert mode in self.modes, f"Invalid randomization mode: {mode}, available modes: {self.modes}"

    def setup(self) -> None:
        """Setup the mass randomization.

        Args:
            mass: The default mass of the rigid bodies."""

        # If the randomization is not enabled then do nothing
        if not self._cfg.enable:
            return

        # Run pre-setup steps
        self.pre_setup()

        # Run the setup functions
        for mode in self._cfg.randomization_modes:
            if mode in self.on_setup_fns:
                self.on_setup_fns[mode]()

    def reset(self, env_ids: torch.Tensor, **kwargs) -> None:
        """Reset the mass of the rigid bodies to their default values.

        Args:
            env_ids: The ids of the environments."""

        # If the randomization is not enabled then do nothing
        if not self._cfg.enable:
            return

        # Apply the default reset, defined in the child class.
        self.default_reset(env_ids)
        if self.update_on_reset:
            # Then apply the randomizations if they exist
            for mode in self._cfg.randomization_modes:
                if mode in self.on_reset_fns:
                    self.on_reset_fns[mode](env_ids)

            self.apply_randomization(env_ids)

    def update(self, **kwargs) -> None:
        """Update the mass of the rigid bodies.

        Args:
            dt: The time step."""

        # If the randomization is not enabled then do nothing
        if not self._cfg.enable:
            return

        # Apply the default update, defined in the child class.
        self.default_update()
        if self.update_on_update:
            # Apply the randomizations if they exist
            for mode in self._cfg.randomization_modes:
                if mode in self.on_update_fns:
                    self.on_update_fns[mode](**kwargs)

            self.apply_randomization()

    def actions(self, **kwargs) -> None:
        """Randomize the actions of the agent."""

        # If the randomization is not enabled then do nothing
        if not self._cfg.enable:
            return

        # Apply the default actions, defined in the child class.
        self.default_actions()
        if self.update_on_actions:
            # Apply the randomizations if they exist
            for mode in self._cfg.randomization_modes:
                if mode in self.on_actions_fns:
                    self.on_actions_fns[mode](**kwargs)

        # Note, only the actions are updated here.

    def observations(self, **kwargs) -> None:
        """Randomize the observations of the agent."""

        if not self._cfg.enable:
            return

        # If the randomization is not enabled then do nothing
        self.default_observations()
        if self.update_on_observations:
            # Apply the randomizations if they exist
            for mode in self._cfg.randomization_modes:
                if mode in self.on_observations_fns:
                    self.on_observations_fns[mode](**kwargs)

        # Note, only the observations are updated here.

    def default_reset(self, env_ids: torch.Tensor, **kwargs) -> None:
        """The default reset function for the randomization."""
        return

    def default_update(self, **kwargs) -> None:
        """The default update function for the randomization."""
        return

    def default_actions(self, **kwargs) -> None:
        """The default actions function for the randomization."""
        return

    def default_observations(self, **kwargs) -> None:
        """The default observations function for the randomization."""
        return

    def apply_randomization(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the randomization to the rigid bodies.

        Args:
            env_ids: The ids of the environments."""

        raise NotImplementedError
