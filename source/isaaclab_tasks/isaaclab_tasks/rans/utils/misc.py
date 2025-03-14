# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class factory:
    def __init__(self):
        self._pairs = {}

    def __call__(self, key, **kwargs):
        return self.create(key, **kwargs)

    @property
    def get_keys(self):
        return self._pairs.keys()

    @property
    def get_values(self):
        return self._pairs.values()

    def register(self, key, value):
        self._pairs[key] = value

    def create(self, key, **kwargs):
        return self._pairs[key](**kwargs)
