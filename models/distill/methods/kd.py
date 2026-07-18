#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from .base import DistillMethod, DistillMethodOutput
from .components.losses import CELoss, LogitsLoss


class KDMethod(DistillMethod):
    """KD: CE + Logits Loss

    纯净版蒸馏：所有 token 平等计算 CE + logits loss，不拆分 think/infer
    """

    def __init__(self, model, distill_config):
        super().__init__(distill_config)
        self.ce_loss_fn = CELoss(**self.distill_config.ce_loss_config)
        self.logits_loss_fn = LogitsLoss(**self.distill_config.logits_loss_config)

    def compute_loss(
        self,
        stu_logits: torch.Tensor,
        tea_logits: torch.Tensor,
        label_ids: torch.Tensor,
        think_mask: torch.Tensor,
        infer_mask: torch.Tensor,
        **kwargs
    ) -> DistillMethodOutput:
        log_output = self.logits_loss_fn(
            stu_logits=stu_logits,
            tea_logits=tea_logits,
            attention_mask=kwargs.get("attention_mask", None),
            think_mask=think_mask,
            infer_mask=infer_mask,
            label_ids=label_ids,
        )
        ce_output = self.ce_loss_fn(
            logits=stu_logits,
            label_ids=label_ids,
            attention_mask=kwargs.get("attention_mask", None),
            think_mask=think_mask,
            infer_mask=infer_mask,
        )
        return DistillMethodOutput(ce_output, log_output)
