# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import RandomizationCore, RandomizationCoreCfg, Registerable
from isaaclab_tasks.rans.utils import PerEnvSeededRNG


@configclass
class ActionsRescalerCfg(RandomizationCoreCfg):
    randomization_modes: list[str] = ["none"]
    """The randomization modes to apply to the actions."""

    slices: list[tuple[int, int] | int] = []
    """The slices of the actions to apply the noise to."""

    rescaling_ranges: list[tuple[float, float]] = []
    """The range within which the scalers are drawn. Default is [].
    The length of the list must be the same as the number of slices."""

    # Used for clipping
    clip_actions: list[tuple[float, float]] | None = None
    """The minimum and maximum values for the actions. Default is None (no clipping).
    The length of the list must be the same as the number of slices, unless it's set to none."""

    # Define the size of the generative space associated with the randomization
    gen_space: int = 1  # DO NOT EDIT


class ActionsRescaler(Registerable, RandomizationCore):
    _cfg: ActionsRescalerCfg

    def __init__(
        self,
        cfg: ActionsRescalerCfg,
        rng: PerEnvSeededRNG,
        scene: InteractiveScene,
        num_envs: int = 1,
        device: str = "cuda",
        **kwargs,
    ):
        """Random action rescaling class.

        Args:
            cfg: The configuration for the randomization.
            rng: The random number generator.
            scene: The scene to get the assets from.
            asset_name: The name of the asset to randomize.
            num_envs: The number of environments.
            device: The device on which the tensors are stored."""

        super().__init__(cfg, rng, scene, num_envs, device)

    @property
    def data(self) -> dict[str, torch.Tensor]:
        """The maximum amount of noise applicable to that env.

        Returns:
            dict[str, torch.Tensor]: The maximum amount of noise applicable to that env."""

        return {"action_noise": torch.stack(self._rescaling_ranges)}

    @property
    def data_shape(self) -> dict[str, tuple[int]]:
        """The shape of the data.

        Returns:
            dict[str, tuple[int]]: The shape of the data."""

        return {
            "action_scales:",
            (sum([1 if isinstance(slice, int) else slice[1] - slice[0] for slice in self._cfg.slices]),),
        }

    def check_cfg(self) -> None:
        """Check the configuration of the randomization."""

        super().check_cfg()

        # Check the slices
        assert len(self._cfg.slices) > 0, "The slices must be defined."
        # Check the min max values
        if self._cfg.clip_actions is not None:
            assert len(self._cfg.slices) == len(
                self._cfg.clip_actions
            ), "The length of 'clip_actions' must be the same as 'slices'."
        # Check the randomization modes
        if "uniform" in self._cfg.randomization_modes:
            assert "normal" not in self._cfg.randomization_modes, "The 'uniform' and 'normal' modes cannot be combined."
            assert len(self._cfg.rescaling_ranges) == len(
                self._cfg.slices
            ), "The length of 'max_delta' must be the same as 'slices'."

    def pre_setup(self) -> None:
        """Setup the action scaling randomization."""

        super().pre_setup()

    def fn_on_setup_uniform(self, **kwargs) -> None:
        """Setup the uniform rescaling of the actions."""

        # Creates the rescaling ranges buffer
        self._rescaling_ranges = []
        for rescaling_range, slices in zip(self._cfg.rescaling_ranges, self._cfg.slices):
            if isinstance(slices, int):
                self._rescaling_ranges.append(torch.ones((self._num_envs), device=self._device))
            else:
                self._rescaling_ranges.append(torch.ones((self._num_envs, slices[1] - slices[0]), device=self._device))

    def default_reset(self, env_ids: torch.Tensor | None, **kwargs) -> None:
        """The default reset function for the randomization."""
        # Do nothing
        pass

    def default_update(self, **kwargs) -> None:
        """The default update function for the randomization."""
        # Do nothing
        pass

    def fn_on_reset_uniform(
        self, env_ids: torch.Tensor | None = None, gen_actions: torch.Tensor | None = None, **kwargs
    ) -> None:
        """Randomizes the scaling factors used on the actions. The update function applies a unique scaling-factor to
        each actions by multiplying them. The expectations is that the maximum and minimum value of the actions is 1
        and -1 respectively. "rescaling_range" is used to determine the upper bound of the scaling-factor by multiplying
        it by the generative action. For each slice of the actions, the function uses the same generative action to
        determine the upper bound of the noise. This means that the same generative action will be used for all the
        slices.

        Args:
            env_ids: The ids of the environments.
            gen_actions: The actions taken by the agent."""

        if gen_actions is None:
            gen_actions = self._rng.sample_uniform_torch(0, 1, (1,), env_ids)

        if env_ids is None:
            env_ids = self._ALL_INDICES

        for i, (rescaling_range, slice) in enumerate(zip(self._cfg.rescaling_ranges, self._cfg.slices)):
            # Get the min values, the max is 1.
            min_values = (1 - gen_actions) * (rescaling_range[1] - rescaling_range[0]) + rescaling_range[0]
            num_actions = 1 if isinstance(slice, int) else slice[1] - slice[0]
            # Sample a value between the previously computed min and 1.
            scalers = self._rng.sample_uniform_torch(
                min_values, torch.ones_like(min_values), (num_actions,), ids=env_ids
            )
            self._rescaling_ranges[i][env_ids] = scalers

    def fn_on_actions_uniform(self, actions: torch.Tensor | None = None, **kwargs) -> None:
        """Apply the scaling to the actions. This function modifies the actions in place.

        Args:
            actions: The actions taken by the agent."""

        for i, slice in enumerate(self._cfg.slices):
            if isinstance(slice, int):
                actions[:, slice] *= self._rescaling_ranges[i]
                if self._cfg.clip_actions is not None:
                    actions[:, slice].clip_(self._cfg.clip_actions[i][0], self._cfg.clip_actions[i][1])
            else:
                actions[:, slice[0] : slice[1]] *= self._rescaling_ranges[i]
                if self._cfg.clip_actions is not None:
                    actions[:, slice[0] : slice[1]].clip_(self._cfg.clip_actions[i][0], self._cfg.clip_actions[i][1])

    def apply_randomization(self, ids: torch.Tensor | None = None) -> None:
        """Everything is done in place. Nothing to update.

        Args:
            ids: The ids of the environments."""

        return
