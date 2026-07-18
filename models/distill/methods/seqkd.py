#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from .base import DistillMethod, DistillMethodOutput
from .components.losses import CELoss


class SeqKDMethod(DistillMethod):
    """SeqKD: 仅使用 CE loss

    纯净版蒸馏：所有 token 平等计算 CE loss，不拆分 think/infer
    """

    def __init__(self, model, distill_config):
        super().__init__(distill_config)
        self.ce_loss_fn = CELoss(**self.distill_config.ce_loss_config)

    def compute_loss(
        self,
        stu_logits: torch.Tensor,
        label_ids: torch.Tensor,
        think_mask: torch.Tensor,
        infer_mask: torch.Tensor,
        **kwargs
    ) -> DistillMethodOutput:
        output = self.ce_loss_fn(
            logits=stu_logits,
            label_ids=label_ids,
            attention_mask=kwargs.get("attention_mask", None),
            think_mask=think_mask,
            infer_mask=infer_mask,
        )
        return DistillMethodOutput(ce_output=output)
