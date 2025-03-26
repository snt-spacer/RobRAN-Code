# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch
from scipy.special import binom

from . import PerEnvSeededRNG


def bernstein(n, k, t):
    return binom(n, k) * t**k * (1.0 - t) ** (n - k)


torch.manual_seed(0)


class TrackGenerator:
    def __init__(
        self,
        min_num_points: int = 9,
        max_num_points: int = 13,
        num_points_per_segment: int = 30,
        min_point_distance: float = 0.05,
        min_angle: float = (12.5 / 180) * np.pi,
        scale: float = 1.0,
        rad: float = 0.2,
        edgy: float = 0.0,
        device: str = "cuda",
        rng: PerEnvSeededRNG | None = None,
    ) -> None:
        """Initializes the TrackGenerator.

        Args:
            min_num_points: The minimum number of points that can be generated.
            max_num_points: The maximum number of points that can be generated.
            num_points_per_segment: The number of points to generate for each segment.
            min_point_distance: The minimum distance between the points sampled to create the track. Should be between
                0 and 1. Smaller values can create more complex tracks.
            min_angle: The minimum angle between the segments of the track. Should be between 0 and pi. Smaller values
                can create more complex tracks. Values close too small can create tracks that are not drivable. I.e. the
                the tracks may self-intersect.
            scale: The scale of the unit square. This defines the size of track in meters.
            rad: The radius of the curve.
            edgy: The edginess of the curve.
            device: The device to use for computation."""

        # Assign the parameters
        self._min_num_points = min_num_points
        self._max_num_points = max_num_points
        self._min_point_distance = min_point_distance
        self._min_angle = min_angle
        self._scale = scale
        self._rad = rad
        self._edgy = edgy
        self._device = device
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        self._rng = rng

        self._num_points_per_segment = num_points_per_segment

        # Compute the angle between the two segments map it to [0, 1]
        self._p = math.atan(self._edgy) / math.pi + 0.5
        # Compute the number of cells in the grid
        self._num_cells = int(1.0 / (self._min_point_distance * 2))
        # Precompute the bernstein polynomials
        t = np.linspace(0, 1, num=self._num_points_per_segment)
        self._bernstein_0 = torch.tensor(bernstein(3, 0, t), device=self._device)
        self._bernstein_1 = torch.tensor(bernstein(3, 1, t), device=self._device)
        self._bernstein_2 = torch.tensor(bernstein(3, 2, t), device=self._device)
        self._bernstein_3 = torch.tensor(bernstein(3, 3, t), device=self._device)

    @staticmethod
    def ccw_sort(points: torch.Tensor):
        """Computes the mean of all the points, and orders them in the trigonometric direction around it.

        Args:
            points: A 3D tensor, [num_envs, num_points, 2]."""

        # Compute the center of the points
        mean = torch.mean(points, axis=1)
        # Generate a vector from the center to the points
        dist = points - mean.unsqueeze(1)
        # Get the angle between the vector and the positive x-axis
        angles = torch.arctan2(dist[:, :, 0], dist[:, :, 1])
        # Sort the angles
        ids = torch.argsort(angles, dim=1)
        # Gather the points in the sorted order along the points dimension
        points = torch.gather(points, 1, ids.unsqueeze(-1).expand(-1, -1, points.size(2)))
        return points

    def get_random_points(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Create n random points in the unit square, which are at least *mindst* apart, then scale them.

        Args:
            num_envs: The number of environments to generate random points for.
            seeds: The seed to use for the random number generator. This is not used yet. We need to implement our own
                multinomial sampling function to make this work.

        Returns:
            A 3D tensor of shape [num_envs, num_points, 2]."""

        # This creates an artificial grid to sample from
        # Generate equal probability weights for each cell in the grid
        # weights = torch.ones((num_envs, self._num_cells * self._num_cells), device=self._device)
        # Using a multinomial distribution, sample N cells without replacement
        cell_idxs = self._rng.sample_unique_integers_torch(
            0, self._num_cells * self._num_cells, self._max_num_points, ids=ids
        )
        # ids = torch.multinomial(weights, num_samples=self._max_num_points, replacement=False)
        # Compute the x and y coordinates of the sampled cells
        x = cell_idxs % self._num_cells
        y = cell_idxs // self._num_cells
        # Add noise to the coordinates so that the problem becomes continuous
        noise = self._rng.sample_uniform_torch(-0.5, 0.5, (self._max_num_points, 2), ids=ids)
        # torch.rand((num_envs, self._max_num_points, 2), device=self._device) * self._min_point_distance
        xy = torch.stack([x, y], dim=2) * self._min_point_distance * 2 + noise  # - 0.5
        return xy * self._scale

    @staticmethod
    def cast_to_0_2pi(ang: torch.Tensor) -> torch.Tensor:
        """Make sure the angles are in the range [0, 2*pi].

        Args:
            ang: A tensor of angles in radians. Dim is [num_envs, num_points].

        Returns:
            A tensor of angles in the range [0, 2*pi]. Dim is [num_envs, num_points]."""

        return (ang >= 0) * ang + (ang < 0) * (ang + 2 * math.pi)

    def get_curve_tangents(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a set of points, compute the tangents of the curve that passes through them.

        Args:
            points: A 3D tensor of shape [num_envs, num_points, 2].

        Returns:
            A tuple containing the points and the angles of the tangents.
            The points are a 3D tensor of shape [num_envs, num_points, 2].
            The angles are a 2D tensor of shape [num_envs, num_points]."""

        # Sort the points in the trigonometric direction
        points = self.ccw_sort(points)
        # Add the first point to the end to close the loop
        points = torch.cat([points, points[:, 0, :].unsqueeze(1)], dim=1)
        # Compute the difference between the points
        dist = torch.diff(points, dim=1)
        # Compute the angle between the points
        ang = torch.arctan2(dist[:, :, 1], dist[:, :, 0])
        # Make sure the angles are in the range [0, 2*pi]
        ang = self.cast_to_0_2pi(ang)
        # Compute the angle of the tangents
        ang1 = ang
        ang2 = torch.roll(ang, 1, dims=1)
        ang = self._p * ang1 + (1 - self._p) * ang2 + (torch.abs(ang2 - ang1) > np.pi) * np.pi
        return points[:, :-1], ang

    def get_curve_tangents_non_fixed_points(
        self, points: torch.Tensor, rng_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a set of points, compute the tangents of the curve that passes through them.
        Unlike get_curve_tangents, this function will prune a random number of points from the input points tensor.
        The method to achieve this is convoluted, if we were looping around all the tensor elements it would be straight
        forward, but here we want to perform tensorized operations only. Here is a step-by-step explanation:

        To make the code more efficient, the first point cannot be pruned. This is so that it's easy to set the removed
        points to the first point in the sequence.

        1. We first generate a tensor which stores the ids of all the points. We will use this to index the points tensor.
        2. We then need to create a mask with the same shape as the ids tensor, that will tell us which points to prune.
          2.a. We generate a binary tensor of length [num_points, max_points_to_prune] with 1s where the points need to be pruned.
          2.b. We then generate a tensor of shape [num_envs, max_points_to_prune] with the ids of the points to prune.
          2.c. We multiply the two tensors to set the ids of the points to prune to the maximum number of points.
          2.d. We then convert this tensor to a mask by using one-hot encoding and summing along the points dimension.
        3. Apply the mask to the ids tensor, the values to be masked are set to the maximum number of points.
        4. Sort the ids tensor.
        5. Set the ids above the maximum number of points to 0. This is done so that the extraneous points are set to be
        the same as the first point in the sequence. This allows for loops to be created.

        Why do we use multinomial sampling over randint? Multinomial sampling allows us to sample without replacement,
        i.e. we are certain we won't sample the same point twice. This is important because we don't want to prune the
        same point twice.

        Why do we need step 2.a.? Step 2.a. allows us to prune random number of points without relying on sparse tensors
        or loops.

        Note:
        multinomial is faster than argsort(rand) when B > 100. Which is why we use it here.

        Args:
            points: A 3D tensor of shape [num_envs, num_points, 2].

        Returns:
            A tuple containing the points and the angles of the tangents.
            The points are a 3D tensor of shape [num_envs, num_points, 2].
            The angles are a 2D tensor of shape [num_envs, num_points]."""

        # Sort the points in the trigonometric direction
        points = self.ccw_sort(points)
        # For each point sequence, randomly select a number of points to prune
        # 1. Generate the ids for the points
        ids = torch.arange(points.shape[1], device=self._device).view(1, -1).expand(points.shape[0], -1)
        # 2. We want to set some of these ids to the maximum number of points so that they can be pruned
        max_points_to_prune = self._max_num_points - self._min_num_points
        # 2.a. First we pick a random number of points to prune, between 0 and max_points_to_prune
        # It's expressed as a fixed shape tensor to enable batch operations.
        # Using a multinomial distribution here would result in an uneven distribution of the number of points pruned.
        num = self._rng.sample_integer_torch(0, max_points_to_prune, (1,), ids=rng_ids)
        # num = torch.randint(0, max_points_to_prune, (points.shape[0],), device=self._device)
        x = torch.arange(max_points_to_prune, device=self._device).unsqueeze(0).expand(points.shape[0], -1)
        pruning_mask = torch.ones((points.shape[0], max_points_to_prune), device=self._device, dtype=torch.int)
        pruning_mask[x > num.unsqueeze(1)] = 0
        # 2.b. We then sample the ids to prune
        ids_to_prune = self._rng.sample_unique_integers_torch(
            0, self._max_num_points - 1, max_points_to_prune, ids=rng_ids
        ).long()
        # 2.c. We then multiply the two so that points that don't have to be pruned are set to the maximum number of points + 1
        # This way we can easily filter them out later
        final_ids_to_prune = pruning_mask * ids_to_prune + (1 - pruning_mask) * (self._max_num_points - 1)
        # 2.d. Convert them to a mask using one-hot encoding and summing along the points dimension allows to create a mask
        ids_mask = torch.sum(torch.nn.functional.one_hot(final_ids_to_prune + 1, self._max_num_points + 1), dim=1)
        # The values above the maximum number of points are removed
        ids_mask = ids_mask[:, :-1]
        # 3. Apply the mask to the ids, the values to be masked are set to the maximum number of points
        ids = ids * (1 - ids_mask) + self._max_num_points * ids_mask
        # 4. Sort the ids
        ids, _ = torch.sort(ids, dim=1)
        # 5. Set the ids above the maximum number of points to 0
        ids[ids >= self._max_num_points] = 0
        # 6. Gather the points
        points = torch.gather(points, 1, ids.unsqueeze(-1).expand(-1, -1, points.size(2)))
        # 7. Get the number of points per track
        num_points_per_track = self._max_num_points - torch.sum(ids_mask == 1, dim=1)

        # Add the first point to the end to close the loop
        points = torch.cat([points, points[:, 0, :].unsqueeze(1)], dim=1)
        # Compute the difference between the points
        dist = torch.diff(points, dim=1)
        # Compute the angle between the points
        ang = torch.arctan2(dist[:, :, 1], dist[:, :, 0])
        # Make sure the angles are in the range [0, 2*pi]
        ang = self.cast_to_0_2pi(ang)
        # Compute the angle of the tangents
        ang1 = ang
        ang2 = torch.roll(ang, 1, dims=1)
        # Manually fix the broken roll operation:
        # adding the first element at the end multiple times screws up the roll
        ang2[torch.arange(ang2.shape[0]), 0] = ang1[torch.arange(ang1.shape[0]), num_points_per_track - 1]
        ang = self._p * ang1 + (1 - self._p) * ang2 + (torch.abs(ang2 - ang1) > np.pi) * np.pi
        # Manually fix the broken roll operation:
        # adding the first element at the end multiple times screws up the roll
        ang = torch.cat([ang, ang[:, 0].unsqueeze(1)], dim=1)
        ang[torch.arange(ang.shape[0]), num_points_per_track] = ang[torch.arange(ang.shape[0]), 0]
        return points[:, :-1], ang[:, :-1], num_points_per_track

    @staticmethod
    def compute_angle(points: torch.Tensor) -> torch.Tensor:
        """
        Compute the angle between the different segments of the curve.

        Args:
            points: A 3D tensor of shape [num_envs, num_points, 2].

        Returns:
            A 2D tensor of shape [num_envs, num_points]."""

        # Create 3 points ordered in sequence
        p0 = torch.roll(points, -1, dims=1)
        p1 = points
        p2 = torch.roll(points, 1, dims=1)
        # Generate 2 vectors from them
        v1 = p1 - p0
        v2 = p1 - p2
        # Compute the angle between them
        angles = torch.arccos(
            torch.einsum("ijk,ijk->ij", v1, v2) / (torch.linalg.norm(v1, axis=2) * torch.linalg.norm(v2, axis=2))
        )
        return angles

    def compute_angle_unsorted(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute the angle between the different segments of the curve.

        Args:
            points: A 3D tensor of shape [num_envs, num_points, 2].

        Returns:
            A 2D tensor of shape [num_envs, num_points]."""

        # Sort the points in the trigonometric direction
        points = self.ccw_sort(points)
        # Compute the angle between the points
        return self.compute_angle(points)

    def get_segment(
        self, point_1: torch.Tensor, point_2: torch.Tensor, angle_1: torch.Tensor, angle_2: torch.Tensor
    ) -> torch.Tensor:
        """Given two points and their angles, compute the bezier curve that passes through them.

        Args:
            point_1: A tensor of shape [num_envs, 2].
            point_2: A tensor of shape [num_envs, 2].
            angle_1: A tensor of shape [num_envs].
            angle_2: A tensor of shape [num_envs].

        Returns:
            A tensor of shape [num_envs, num_points*num_points_per_segment, 2]."""

        # Get the first two points
        p0 = point_1
        p3 = point_2
        # Compute the distance between the points
        d = torch.sqrt(torch.sum((p3 - p0) ** 2, axis=1))
        # Compute the intermediate points
        p1 = p0 + torch.stack([torch.cos(angle_1), torch.sin(angle_1)], dim=1) * self._rad * d.unsqueeze(-1)
        p2 = p3 + torch.stack([-torch.cos(angle_2), -torch.sin(angle_2)], dim=1) * self._rad * d.unsqueeze(-1)
        # Generate the bezier curve
        return self.bezier(points=[p0, p1, p2, p3])

    @staticmethod
    def batch_outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the outer product of two tensors.

        Args:
            a: A tensor of shape [b, n, m].
            b: A tensor of shape [b, n, m].

        Returns:
            A tensor of shape [b, n, m]."""

        return torch.einsum("ij,ik->ijk", a, b)

    def bezier(self, points: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute the bezier curve that passes through the given points.

        Args:
            points: A list of 4 tensors. Each tensor has shape [num_envs, 2].

        Returns:
            A tensor of shape [num_envs, num_points_per_segment, 2]."""

        # Compute the bezier curve
        curve = torch.zeros((points[0].shape[0], self._num_points_per_segment, 2), device=self._device)
        curve += self.batch_outer_product(
            self._bernstein_0.unsqueeze(dim=0).expand(points[0].shape[0], -1),
            points[0],
        )
        curve += self.batch_outer_product(
            self._bernstein_1.unsqueeze(dim=0).expand(points[1].shape[0], -1),
            points[1],
        )
        curve += self.batch_outer_product(
            self._bernstein_2.unsqueeze(dim=0).expand(points[2].shape[0], -1),
            points[2],
        )
        curve += self.batch_outer_product(
            self._bernstein_3.unsqueeze(dim=0).expand(points[3].shape[0], -1),
            points[3],
        )
        return curve

    def get_bezier_curve(self, points: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Given an array of points, create a curve through those points.

        Args:
            points: A 3D tensor of shape [num_envs, num_points, 2].
            angles: A 2D tensor of shape [num_envs, num_points].

        Returns:
            A tensor of shape [num_envs, num_points*num_points_per_segment, 2]."""

        # For each ordered pair of points, compute the bezier curve that passes through them
        segments = []
        for i in range(points.shape[1] - 1):
            seg = self.get_segment(points[:, i, :2], points[:, i + 1, :2], angles[:, i], angles[:, i + 1])
            segments.append(seg)
        seg = self.get_segment(points[:, -1, :2], points[:, 0, :2], angles[:, -1], angles[:, 0])
        segments.append(seg)
        curve = torch.cat(segments, dim=1)
        return curve

    def get_bezier_curve_non_fixed_points(
        self, points: torch.Tensor, angles: torch.Tensor, num_points_per_track: torch.Tensor
    ) -> torch.Tensor:
        """Given an array of points, create a curve through those points.
        Unlike get_bezier_curve, this function can handle tracks with a variable number of points.

        Args:
            points: A 3D tensor of shape [num_envs, num_points, 2].
            angles: A 2D tensor of shape [num_envs, num_points].
            num_points_per_track: A 1D tensor of shape [num_envs] containing the number of points in each track.

        Returns:
            A tensor of shape [num_envs, num_points*num_points_per_segment, 2]."""

        # For each ordered pair of points, compute the bezier curve that passes through them
        segments = []
        for i in range(points.shape[1] - 1):
            seg = self.get_segment(points[:, i, :2], points[:, i + 1, :2], angles[:, i], angles[:, i + 1])
            # Perform the check only if the number of points is greater than the minimum number of points
            if i >= self._min_num_points:
                overflow = num_points_per_track < i
                seg[overflow, :, :] = segments[0][overflow, 0].unsqueeze(1)
            segments.append(seg)
        seg = self.get_segment(points[:, -1, :2], points[:, 0, :2], angles[:, -1], angles[:, 0])
        overflow = num_points_per_track < (i + 1)
        seg[overflow, :, :] = segments[0][overflow, 0].unsqueeze(1)
        segments.append(seg)
        curve = torch.cat(segments, dim=1)
        return curve

    def generate_tracks(self, num_tracks: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random tracks.

        Args:
            num_tracks: The number of tracks to generate.

        Returns:
            A tuple containing the points, the tangents, and the curve.
            The points are a 3D tensor of shape [num_tracks, num_points, 2].
            The tangents are a 2D tensor of shape [num_tracks, num_points].
            The curve is a 3D tensor of shape [num_tracks, num_points*num_points_per_segment, 2]."""

        points, tangents = self.generate_tracks_points(num_tracks)
        curve = self.get_bezier_curve(points, tangents)
        return points, tangents, curve

    def generate_tracks_non_fixed_points(
        self, num_tracks: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random tracks.
        Unlike generate_tracks, this function can will generate tracks with a variable number of corners.
        This will result in a larger diversity of tracks generated. Usually, the fewer the number of corners, the
        simpler the track.

        Args:
            num_tracks: The number of tracks to generate.

        Returns:
            A tuple containing the points, the tangents, the number of points per track, and the curve.
            The points are a 3D tensor of shape [num_tracks, num_points, 2].
            The tangents are a 2D tensor of shape [num_tracks, num_points].
            The number of points per track is a 1D tensor of shape [num_tracks].
            The curve is a 3D tensor of shape [num_tracks, num_points*num_points_per_segment, 2]."""

        points, tangents, num_points_per_track = self.generate_tracks_points_non_fixed_points(num_tracks)
        curve = self.get_bezier_curve_non_fixed_points(points, tangents, num_points_per_track)
        return points, tangents, num_points_per_track, curve

    def generate_tracks_points(
        self,
        ids: torch.Tensor | None = None,
        prev_points: torch.Tensor | None = None,
        prev_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate random tracks but only return the points.

        Args:
            num_tracks: The number of tracks to generate.
            prev_points: The previous points to append the new points to.

        Returns:
            A tuple containing the points and the tangents.
            The points are a 3D tensor of shape [num_tracks, num_points, 2].
            The tangents are a 2D tensor of shape [num_tracks, num_points]."""

        # Generate an ordered set of ids
        if prev_ids is None:
            prev_ids = torch.arange(0, len(ids), device=self._device)
        # Generate random points
        points = self.get_random_points(ids=ids)
        # Check if the angles created by the different segments are greater than the minimum angle allowed
        angles = self.compute_angle_unsorted(points)
        keep = torch.prod(angles > self._min_angle, dim=1) != 0
        # Assign the newly generated points to the correct ids
        if prev_points is not None:
            prev_points[prev_ids] = points
        else:
            prev_points = points
        # Get the local ids that need to be regenerated
        prev_ids = prev_ids[torch.logical_not(keep)]
        # Get the global ids that need to be regenerated
        ids_to_regenerate = ids[torch.logical_not(keep)]

        # Check if we are done
        if keep.sum() == len(ids):
            prev_points, tangents = self.get_curve_tangents(prev_points)
        else:
            prev_point, tangents = self.generate_tracks_points(
                ids_to_regenerate.clone(), prev_points=prev_points.clone(), prev_ids=prev_ids.clone()
            )

        return prev_points, tangents

    def generate_tracks_points_non_fixed_points(
        self,
        ids: torch.Tensor,
        prev_points: torch.Tensor | None = None,
        prev_ids: torch.Tensor | None = None,
        og_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Generate random tracks but only return the points.
        Unlike generate_tracks_points, this function will prune a random number of points from the input points tensor.

        Args:
            num_tracks: The number of tracks to generate.
            prev_points: The previous points to append the new points to.

        Returns:
            A tuple containing the points, the tangents, and the number of points per track.
            The points are a 3D tensor of shape [num_tracks, num_points, 2].
            The tangents are a 2D tensor of shape [num_tracks, num_points].
            The number of points per track is a 1D tensor of shape [num_tracks]."""
        # Generate an ordered set of ids
        if prev_ids is None:
            prev_ids = torch.arange(0, len(ids), device=self._device)
            og_ids = ids.clone()
        points = self.get_random_points(ids=ids)
        # Check if the angles created by the different segments are greater than the minimum angle allowed
        angles = self.compute_angle_unsorted(points)
        keep = torch.prod(angles > self._min_angle, dim=1) != 0
        # Assign the newly generated points to the correct ids
        if prev_points is not None:
            prev_points[prev_ids] = points
        else:
            prev_points = points
        # Get the local ids that need to be regenerated
        prev_ids = prev_ids[torch.logical_not(keep)]
        # Get the global ids that need to be regenerated
        ids_to_regenerate = ids[torch.logical_not(keep)]

        # Check if we are done
        if keep.sum() == len(ids):
            prev_points, tangents, num_points_per_track = self.get_curve_tangents_non_fixed_points(
                prev_points, rng_ids=og_ids
            )
        else:
            prev_points, tangents, num_points_per_track = self.generate_tracks_points_non_fixed_points(
                ids_to_regenerate.clone(), prev_points=prev_points.clone(), prev_ids=prev_ids.clone(), og_ids=og_ids
            )

        return prev_points, tangents, num_points_per_track
