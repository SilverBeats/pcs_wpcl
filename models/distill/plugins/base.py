#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import is_dataclass, asdict, dataclass
from enum import Enum
from typing import Any

import torch.nn as nn


class Scope(Enum):
    pre_loss = "PreLoss"   # 损失计算前: 预处理数据、特征对齐等
    loss = "Loss"          # 损失计算中: 影响损失计算方式
    post_loss = "PostLoss"  # 损失计算后: 对损失做后处理、加额外损失


@dataclass
class PluginConfig:
    enabled: bool = False

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


class Plugin(ABC, nn.Module):
    SCOPE = None

    def __init__(self, config):
        super(Plugin, self).__init__()
        assert self.SCOPE is not None
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @abstractmethod
    def inject(self, *args, **kwargs):
        """作用 Plugin 逻辑"""
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(self, save_dir: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_pretrained(ckpt_dir: str, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.inject(*args, **kwargs)
