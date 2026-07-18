#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter

from .base import Scope, PluginConfig, Plugin
from ..methods.base import DistillMethodOutput


@dataclass
class UncertaintyPluginConfig(PluginConfig):
    adopt_temp: bool = True
    temperature: float = 2.0
    min_weight: float = 0.3


class UncertaintyPlugin(Plugin):

    SCOPE = Scope.loss

    def __init__(self, config: UncertaintyPluginConfig):
        super().__init__(config)

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 token-level 熵"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def _compute_soft_uncertainty_weight(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算平滑后的不确定性权重。
        相比于原版的 exp(-entropy)，这个版本对发散性 Token 更友好。
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # 使用温度平滑，并确保权重不低于 min_weight
        weight = torch.exp(-entropy / self.config.temperature)
        return torch.max(weight, torch.ones_like(weight) * self.config.min_weight)

    def inject(
        self,
        distill_output: "DistillMethodOutput",
        tea_logits: torch.Tensor,
        indices: torch.Tensor,
        **kwargs
    ):
        if not self.enabled:
            return
        assert tea_logits is not None
        # 计算不确定性权重
        func = (
            self._compute_soft_uncertainty_weight
            if self.config.adopt_temp
            else self._compute_entropy
        )
        weight = func(tea_logits)
        # 给蒸馏损失加权
        for distill_method, loss_output in distill_output.items():
            if loss_output is None:
                continue
            start_idx = 1 if "ce" in distill_method else 0
            for loss_name in loss_output.keys():
                if "valid_tokens" in loss_name or loss_output[loss_name] is None:
                    continue
                if "loss_per_token" in loss_name:
                    if isinstance(loss_output[loss_name], list):
                        for i in range(len(loss_output[loss_name])):
                            loss_output[loss_name][i][indices] *= weight[
                                indices, start_idx:
                            ]
                    else:
                        loss_output[loss_name][indices] *= weight[indices, start_idx:]

    @staticmethod
    def from_pretrained(ckpt_dir: str, *args, **kwargs):
        ckpt_dir = os.path.join(ckpt_dir, UncertaintyPlugin.__name__)
        if not os.path.exists(ckpt_dir):
            return
        plugin_config = FileReader.read(
            os.path.join(ckpt_dir, "config.json"), return_dict=True
        )
        plugin = UncertaintyPlugin(UncertaintyPluginConfig(**plugin_config))
        return plugin

    def save_pretrained(self, save_dir: str):
        if not self.enabled:
            return
        save_dir = os.path.join(save_dir, type(self).__name__)
        os.makedirs(save_dir, exist_ok=True)
        FileWriter.dump(
            self.config.to_dict(), os.path.join(save_dir, "config.json"), indent=4
        )
