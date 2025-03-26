# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app
import torch
import unittest

from omni.isaac.lab_tasks.rans.utils import PerEnvSeededRNG, TrackGenerator


class TestTrackGenerator(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def test_track_generator_simple(self):
        # Create the RNG
        seeds = torch.randint(0, 1000, (1000,), dtype=torch.int32, device="cuda")
        pesrng_1 = PerEnvSeededRNG(seeds, 1000, "cuda")
        # Create the TrackGenerator
        TG = TrackGenerator(rng=pesrng_1, scale=20.0)
        # Create a list of ids
        ids = torch.arange(0, 1000, device="cuda")
        # Generate Tracks
        for i in range(100):
            points, tangents = TG.generate_tracks_points(ids)

    def test_track_generator_complex(self):
        # Create the RNG
        seeds = torch.randint(0, 1000, (1000,), dtype=torch.int32, device="cuda")
        pesrng_1 = PerEnvSeededRNG(seeds, 1000, "cuda")
        # Create the TrackGenerator
        TG = TrackGenerator(rng=pesrng_1, scale=20.0)
        # Create a list of ids
        ids = torch.arange(0, 1000, device="cuda")
        # Generate Tracks
        for i in range(100):
            points, tangents, num_points_per_track = TG.generate_tracks_points_non_fixed_points(ids)

    def test_track_generator_simple_decreasing_ids(self):
        # Create the RNG
        seeds = torch.randint(0, 1000, (1000,), dtype=torch.int32, device="cuda")
        pesrng_1 = PerEnvSeededRNG(seeds, 1000, "cuda")
        # Create the TrackGenerator
        TG = TrackGenerator(rng=pesrng_1, scale=20.0)
        # Create a list of ids
        for i in range(1000):
            ids = torch.arange(0, 1000 - i, device="cuda")
            # Generate Tracks
            points, tangents = TG.generate_tracks_points(ids)

    def test_track_generator_complex_decreasing_ids(self):
        # Create the RNG
        seeds = torch.randint(0, 1000, (1000,), dtype=torch.int32, device="cuda")
        pesrng_1 = PerEnvSeededRNG(seeds, 1000, "cuda")
        # Create the TrackGenerator
        TG = TrackGenerator(rng=pesrng_1, scale=20.0)
        # Create a list of ids
        for i in range(1000):
            ids = torch.arange(0, 1000 - i, device="cuda")
            # Generate Tracks
            points, tangents, num_points_per_track = TG.generate_tracks_points_non_fixed_points(ids)


if __name__ == "__main__":
    run_tests()
