#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


class Pojo:
    def to_dict(self):
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} must be a @dataclass subclass of Pojo for to_dict()"
            )
        return asdict(self)

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()


@dataclass
class RoleProfile(Pojo):
    name: str
    age: str
    gender: str
    archetype: str
    arch_traits: str
    arch_desc: str
    job_category: str
    job_desc: str
    culture: str
    culture_countries: str
    summary: str
    biography: str


@dataclass
class EventSample(Pojo):
    index: int
    lang: str
    event: str
    category: str = ""
    label: int = None


@dataclass
class RoleSample(Pojo):
    index: int
    en: RoleProfile


@dataclass
class DebateSample(Pojo):
    index: str
    event: EventSample
    role: RoleSample

    def __post_init__(self):
        event_index = self.event.index
        role_index = self.role.index
        if not self.index:
            self.index = f"event_{event_index}_role_{role_index}"


@dataclass
class TrainSample(Pojo):
    """Alpaca format training sample."""

    system: str
    instruction: str
    output: str
    event_idx: int = None
    role_idx: int = None
    relation: str = None


@dataclass
class DatasetAblationArguments(Pojo):
    """Ablation experiment flags for dataset construction."""

    wo_cot: bool = False
    wo_ocean: bool = False
    wo_gender: bool = False
    wo_age: bool = False
    wo_culture: bool = False
    wo_job_category: bool = False
    wo_general: bool = False
