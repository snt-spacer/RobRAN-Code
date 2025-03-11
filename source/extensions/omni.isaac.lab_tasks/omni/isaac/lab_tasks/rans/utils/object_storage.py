# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from .rng_utils import PerEnvSeededRNG


class ObjectStorage:
    def __init__(
        self,
        num_envs: int,
        max_num_vis_objects_in_env: int,
        store_height: float,
        rng: PerEnvSeededRNG,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            num_envs (int): The number of environments.
            env_origin (torch.tensor): The origin of the environment.
            max_num_vis_objects_in_env (int): The maximum number of objects visible in the environment.
            store_height (float): The height position to storage the objects.
            device (str): The device to use.
        """

        assert num_envs > 0, "The number of environments must be greater than 0."

        self.num_envs = num_envs
        self._max_num_vis_objects_in_env = max_num_vis_objects_in_env
        self._store_height = store_height
        self._rng = rng
        self._device = device
        self.storage_buff = None

    def create_storage_buffer(self, env_origin: torch.tensor) -> torch.tensor:
        """
        Generates a tensor representing hidden objects positions for each environment.

        Returns:
            torch.tensor: A tensor containing the coordinates of hidden objects.
        """
        # Generate positions
        num = torch.arange(int(math.sqrt(self._max_num_vis_objects_in_env)) + 1, device=self._device)
        x, y = torch.meshgrid(num, num, indexing="ij")
        x = x.flatten()
        y = y.flatten()
        z = torch.ones_like(x) * self._store_height
        xyz = torch.stack((x, y, z), dim=1)
        xyz = torch.repeat_interleave(xyz.unsqueeze(0), self.num_envs, dim=0)
        xyz[:, :, :2] += env_origin.unsqueeze(1)[:, :, :2]
        # Generate quats
        xyzw = torch.zeros(self.num_envs, xyz.shape[1], 4, device=self._device)
        xyzw[:, -1] = 0.0

        self.storage_buff = torch.cat((xyz, xyzw), dim=2)
        # self.storage_buff = self.storage_buff.reshape(self.num_envs, self.storage_buff.shape[1]**2, 3)

        # return self.storage_buff

    def get_positions_with_storage(
        self, objects_pos: torch.tensor, mask: torch.tensor, env_ids: torch.tensor
    ) -> torch.tensor:
        """
        Returns the position of the objects for each environment.

        Args:
            objects_pos (torch.tensor): The position of the objects.
            mask (torch.tensor): The mask to apply to the objects.
            env_ids (torch.tensor): The ids of the environments.

        Returns:
            torch.tensor: The position of the objects for each environment.
        """

        objects_pos[~mask] = self.storage_buff[env_ids, : self._max_num_vis_objects_in_env][~mask]

        return objects_pos
