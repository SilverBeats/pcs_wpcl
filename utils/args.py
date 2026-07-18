#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行参数定义
支持通过 YAML 配置文件加载所有参数
"""

from dataclasses import dataclass, field, is_dataclass, asdict
from typing import List, Optional, Literal

from transformers import TrainingArguments

from models.distill import (
    ContrastivePluginConfig,
    CurriculumPluginConfig,
    UncertaintyPluginConfig,
)


class Pojo:

    def to_dict(self):
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} must be a @dataclass subclass of Pojo for to_dict()"
            )
        return asdict(self)


@dataclass
class DatasetAblationArguments(Pojo):
    """
    数据集消融实验配置。
    用于控制角色属性的使用（Exp 2 消融实验）。
    """

    wo_cot: bool = False  # 消融 CoT 提示词（不用 CoT，直接 infer）
    wo_ocean: bool = False  # 消融人格特质
    wo_gender: bool = False  # 消融性别
    wo_age: bool = False  # 消融年龄
    wo_culture: bool = False  # 消融文化背景
    wo_job_category: bool = False  # 消融职业类别
    wo_general: bool = False  # 消融通用数据（只用个性化数据）

    def to_exp_name(self) -> str:
        """生成实验名称"""
        parts = []
        if self.wo_cot:
            parts.append("no_cot")
        if self.wo_ocean:
            parts.append("no_ocean")
        if self.wo_gender:
            parts.append("no_gender")
        if self.wo_age:
            parts.append("no_age")
        if self.wo_culture:
            parts.append("no_culture")
        if self.wo_job_category:
            parts.append("no_job")
        if self.wo_general:
            parts.append("no_general")
        if not parts:
            return "full"
        return "_".join(parts)


@dataclass
class LoraConfig(Pojo):
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "gate_proj",
            "k_proj",
            "o_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "q_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DistillConfigArguments(Pojo):
    # seqkd | featkd | kd
    distill_method: str = "featkd"
    lora_config: LoraConfig = field(default_factory=LoraConfig)

    adopt_hidden_loss: bool = True
    adopt_logits_loss: bool = True
    adopt_ce_loss: bool = True
    hidden_loss_config: dict = None
    logits_loss_config: dict = None
    ce_loss_config: dict = None

    # 纯净版配置: True = 合并所有 token 计算损失，不拆分 think/infer
    general_pure: bool = True
    personalization_pure: bool = True

    # plugin
    plugins: List[str] = field(default_factory=list)
    contrastive_config: ContrastivePluginConfig = field(
        default_factory=ContrastivePluginConfig
    )
    curriculum_config: CurriculumPluginConfig = field(
        default_factory=CurriculumPluginConfig
    )
    uncertainty_config: "UncertaintyPluginConfig" = field(
        default_factory=lambda: UncertaintyPluginConfig()
    )

    def __post_init__(self):
        if self.ce_loss_config is None:
            self.ce_loss_config = {"weight": 1.0}
        if self.hidden_loss_config is None:
            self.hidden_loss_config = {"loss_type": "cosine", "weight": 1.0}
        if self.logits_loss_config is None:
            self.logits_loss_config = {"temperature": 3.0, "weight": 0.3}
        """根据 distill_method 设置 adopt_* 标志"""
        method_to_flags = {
            "seqkd": (False, False, True),  # (logits, hidden, ce)
            "kd": (True, False, True),  # CE + logits
            "featkd": (True, True, True),  # CE + logits + hidden
        }
        if self.distill_method in method_to_flags:
            logits, hidden, ce = method_to_flags[self.distill_method]
            self.adopt_logits_loss = logits
            self.adopt_hidden_loss = hidden
            self.adopt_ce_loss = ce

        if isinstance(self.lora_config, dict):
            self.lora_config = LoraConfig(**self.lora_config)
        if isinstance(self.contrastive_config, dict):
            self.contrastive_config = ContrastivePluginConfig(**self.contrastive_config)
        if isinstance(self.uncertainty_config, dict):
            self.uncertainty_config = UncertaintyPluginConfig(**self.uncertainty_config)
        if isinstance(self.curriculum_config, dict):
            self.curriculum_config = CurriculumPluginConfig(**self.curriculum_config)


@dataclass
class ModelArguments(Pojo):
    """
    模型路径和层配置。
    """

    teacher_name_or_path: str = "/hy-tmp/Qwen3-8B"
    student_name_or_path: str = "/hy-tmp/Qwen3-0.6B"
    target_layer_indexes: Optional[List[int]] = None  # None 表示所有层


@dataclass
class CustomTrainingArguments(TrainingArguments, Pojo):
    patience: int = 0
    delta: float = 0
    max_length: int = 512
