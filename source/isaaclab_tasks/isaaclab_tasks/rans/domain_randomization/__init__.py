# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .randomization_core import RandomizationCore, RandomizationCoreCfg


class Registerable:
    def __init_subclass__(cls):
        RandomizerFactory.register(cls.__name__, cls)


class RandomizerFactory:
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable):
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cfg: RandomizationCoreCfg, *args, **kwargs):
        cls_name = cfg.__class__.__name__[:-3]  # Remove "Cfg" from the class name

        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered.")

        return cls.registry[cls_name](cfg, *args, **kwargs)


from .actions_rescaler import ActionsRescaler, ActionsRescalerCfg
from .com import CoMRandomization, CoMRandomizationCfg
from .inertia import InertiaRandomization, InertiaRandomizationCfg
from .mass import MassRandomization, MassRandomizationCfg
from .noisy_actions import NoisyActions, NoisyActionsCfg
from .noisy_observations import NoisyObservations, NoisyObservationsCfg
from .wrench import WrenchRandomization, WrenchRandomizationCfg
