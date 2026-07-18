#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import torch

from .base import DistillMethod, DistillMethodOutput
from .components.losses import CELoss, LogitsLoss, HiddenLoss


class FeatKDMethod(DistillMethod):
    """FeatKD: CE + Logits + Hidden Loss

    纯净版蒸馏：所有 token 平等计算各类损失，不拆分 think/infer

    可选：Attention Matrix Distillation（注意力矩阵蒸馏）
    """

    def __init__(self, model, distill_config):
        super().__init__(distill_config)
        self.ce_loss_fn = CELoss(**self.distill_config.ce_loss_config)
        self.logits_loss_fn = LogitsLoss(**self.distill_config.logits_loss_config)

        # Hidden loss
        self.hidden_loss_fn = HiddenLoss(
            layer_mapping=model.layer_mapping,
            adapter_layers=model.adapter_layers,
            **self.distill_config.hidden_loss_config,
        )

    def compute_loss(
        self,
        stu_logits: torch.Tensor,
        tea_logits: torch.Tensor,
        stu_feats: Dict[int, torch.Tensor],
        tea_feats: Dict[int, torch.Tensor],
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
        hid_output = self.hidden_loss_fn(
            stu_feats=stu_feats,
            tea_feats=tea_feats,
            attention_mask=kwargs.get("attention_mask", None),
            think_mask=think_mask,
            infer_mask=infer_mask,
            label_ids=label_ids,
        )

        return DistillMethodOutput(ce_output, log_output, hid_output)
