#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒸馏方法基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from typing import List, Optional, Tuple

import torch

from .components.losses import LossOutput


@dataclass
class DistillMethodOutput:
    """蒸馏方法输出"""

    ce_output: Optional[LossOutput] = None
    log_output: Optional[LossOutput] = None
    hid_output: Optional[LossOutput] = None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self) -> List[str]:
        return self.__dict__.keys()

    def values(self) -> List[LossOutput]:
        return self.__dict__.values()

    def items(self) -> List[Tuple[str, LossOutput]]:
        return self.__dict__.items()


class DistillMethod(ABC):
    """蒸馏方法基类"""

    def __init__(self, distill_config: Any):
        self.distill_config = distill_config

    @abstractmethod
    def compute_loss(
        self,
        stu_logits: torch.Tensor,
        tea_logits: Optional[torch.Tensor] = None,
        student_feats: Optional[Dict[int, torch.Tensor]] = None,
        teacher_feats: Optional[Dict[int, torch.Tensor]] = None,
        label_ids: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DistillMethodOutput:
        """计算蒸馏损失，返回 loss 字典

        纯净版蒸馏：所有 token 一视同仁，不区分 think/infer
        返回的 loss 字典中：
        - ce_loss/log_loss/hid_loss/att_loss 表示纯净版蒸馏损失（合并的）
        - think_*/infer_* 保持为 None
        """
        pass

    @property
    def need_teacher_model(self) -> bool:
        """是否需要 teacher 模型"""
        return True

    def __call__(
        self,
        stu_logits: torch.Tensor,
        tea_logits: Optional[torch.Tensor] = None,
        student_feats: Optional[Dict[int, torch.Tensor]] = None,
        teacher_feats: Optional[Dict[int, torch.Tensor]] = None,
        label_ids: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.compute_loss(
            stu_logits=stu_logits,
            tea_logits=tea_logits,
            student_feats=student_feats,
            teacher_feats=teacher_feats,
            label_ids=label_ids,
            think_mask=think_mask,
            infer_mask=infer_mask,
            **kwargs
        )
