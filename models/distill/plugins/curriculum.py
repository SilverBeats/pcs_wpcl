#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CurriculumPlugin: 课程权重插件
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter

from .base import Plugin, PluginConfig, Scope
from ..methods.base import DistillMethodOutput


class CurriculumWeightScheduler:
    """课程权重调度：think权重递减，infer权重递增

    训练早期：更多关注 think（推理逻辑链）
    训练后期：更多关注 infer（最终结论）
    """

    def __init__(
        self,
        total_steps: int,
        think_start: float = 1.0,
        think_end: float = 0.3,
        infer_start: float = 0.3,
        infer_end: float = 1.0,
    ):
        self.total_steps = max(total_steps, 1)
        self.think_schedule = np.linspace(think_start, think_end, self.total_steps)
        self.infer_schedule = np.linspace(infer_start, infer_end, self.total_steps)

    def get_weights(self, step: int) -> Tuple[float, float]:
        t = min(step / self.total_steps, 1.0)
        idx = min(int(t * (self.total_steps - 1)), self.total_steps - 1)
        return float(self.think_schedule[idx]), float(self.infer_schedule[idx])


@dataclass
class CurriculumPluginConfig(PluginConfig):
    think_start: float = 1.0
    think_end: float = 0.3
    infer_start: float = 0.3
    infer_end: float = 1.0


class CurriculumPlugin(Plugin):
    """课程权重插件"""

    SCOPE = Scope.loss

    def __init__(self, config: CurriculumPluginConfig):
        super().__init__(config)
        self.scheduler = None

    def set_total_steps(self, total_steps: int):
        if not self.enabled:
            return
        self.scheduler = CurriculumWeightScheduler(
            total_steps=total_steps,
            think_start=self.config["think_start"],
            think_end=self.config["think_end"],
            infer_start=self.config["infer_start"],
            infer_end=self.config["infer_end"],
        )

    def inject(
        self,
        distill_output: "DistillMethodOutput",
        indices: torch.Tensor,
        step: int,
        is_training: bool = True,
        **kwargs
    ):
        if not self.enabled:
            return
        think_w, inf_w = 1.0, 1.0
        if is_training and self.scheduler is not None:
            think_w, inf_w = self.scheduler.get_weights(step)

        # 给蒸馏损失加权
        for distill_method, loss_output in distill_output.items():
            if loss_output is None:
                continue
            for loss_name in loss_output.keys():
                if "valid_tokens" in loss_name or loss_output[loss_name] is None:
                    continue
                weight = think_w if "think" in loss_name else inf_w
                if "loss_per_token" in loss_name:
                    if isinstance(loss_output[loss_name], list):
                        for i in range(len(loss_output[loss_name])):
                            loss_output[loss_name][i][indices] *= weight
                    else:
                        loss_output[loss_name][indices] *= weight

    def save_pretrained(self, save_dir: str):
        if not self.enabled:
            return
        save_dir = os.path.join(save_dir, type(self).__name__)
        os.makedirs(save_dir, exist_ok=True)
        FileWriter.dump(
            self.config.to_dict(), os.path.join(save_dir, "config.json"), indent=4
        )

    @staticmethod
    def from_pretrained(ckpt_dir: str, *args, **kwargs):
        ckpt_dir = os.path.join(ckpt_dir, CurriculumPlugin.__name__)
        if not os.path.exists(ckpt_dir):
            return
        plugin_config = FileReader.read(
            os.path.join(ckpt_dir, "config.json"), return_dict=True
        )
        plugin = CurriculumPlugin(CurriculumPluginConfig(**plugin_config))
        return plugin
