# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

import math
import torch
import unittest

from isaaclab_tasks.rans.utils.object_storage import ObjectStorage

from isaaclab_tasks.rans.utils import PerEnvSeededRNG


class TestObjectStorage(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    ############################################################
    # Test Creation of ObjectStorage
    ############################################################

    def test_shape_storage_buffer(self):
        num_envs = 4
        max_num_obstacles = 20
        sotre_height = -2
        device = "cuda"
        env_origin = torch.randint(low=-100, high=100, size=(num_envs, 2), device=device)
        obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=max_num_obstacles,
            store_height=sotre_height,
            device=device,
        )
        storage_buff = obstacles_generator.create_storage_buffer(env_origin=env_origin)
        num = torch.arange(math.sqrt(max_num_obstacles) + 1, device=device).shape[0]
        self.assertEqual(storage_buff.shape, (num_envs, num, num, 3))

    def test_store_height_in_buffer(self):
        num_envs = 4
        max_num_obstacles = 20
        sotre_height = -2
        device = "cuda"
        env_origin = torch.randint(low=-100, high=100, size=(num_envs, 2), device=device)
        obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=max_num_obstacles,
            store_height=sotre_height,
            device=device,
        )
        storage_buff = obstacles_generator.create_storage_buffer(env_origin=env_origin)
        self.assertTrue(torch.all(storage_buff[:, :, :, 2] == sotre_height))

    def test_env_origin_offset(self):
        num_envs = 4
        max_num_obstacles = 20
        sotre_height = -2
        device = "cuda"
        env_origin = torch.randint(low=-100, high=100, size=(num_envs, 2), device=device)
        obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=max_num_obstacles,
            store_height=sotre_height,
            device=device,
        )
        storage_buff = obstacles_generator.create_storage_buffer(env_origin=env_origin)
        self.assertTrue(torch.all(storage_buff[:, 0, 0, :2] == env_origin))

    def test_zero_visible_objects(self):
        num_envs = 4
        max_num_obstacles = 0
        sotre_height = -2
        device = "cuda"
        env_origin = torch.randint(low=-100, high=100, size=(num_envs, 2), device=device)
        obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=max_num_obstacles,
            store_height=sotre_height,
            device=device,
        )
        storage_buff = obstacles_generator.create_storage_buffer(env_origin=env_origin)
        self.assertTrue(torch.all(storage_buff == torch.tensor([], device=device)))


if __name__ == "__main__":
    run_tests()
