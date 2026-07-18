#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F

from .modules import MultiLayerAdaptationLayer


@dataclass
class LossOutput:
    # *_loss: scalar
    # *_per_token: [bsz, seq_len]
    # *_valid_tokens: [bsz,]
    loss: Optional[torch.Tensor] = None
    loss_per_token: Optional[torch.Tensor | List[torch.Tensor]] = None
    valid_tokens: Optional[torch.Tensor] = None

    think_loss: Optional[torch.Tensor] = None
    think_loss_per_token: Optional[torch.Tensor | List[torch.Tensor]] = None
    think_valid_tokens: Optional[torch.Tensor] = None

    infer_loss: Optional[torch.Tensor] = None
    infer_loss_per_token: Optional[torch.Tensor | List[torch.Tensor]] = None
    infer_valid_tokens: Optional[torch.Tensor] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()


def _format_loss_output(
    loss_per_token: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> LossOutput:
    # loss_per_token.shape = [bsz, seq_len]
    # attention_mask.shape = [bsz, seq_len], 0 表示要盖住
    seq_len = loss_per_token.shape[1]
    # [bsz, ]
    valid_tokens = torch.ones_like(loss_per_token) * seq_len
    if attention_mask is not None:
        # [bsz, seq_len]
        loss_per_token = loss_per_token * attention_mask
        # [bsz, ]
        valid_tokens = attention_mask.sum(-1).clamp(min=1)

    loss = (loss_per_token.sum(-1) / valid_tokens).mean()
    output = LossOutput(
        loss=loss,
        valid_tokens=valid_tokens,
        loss_per_token=loss_per_token,
    )
    return output


def _build_output(
    loss_per_token: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    think_mask: Optional[torch.Tensor] = None,
    infer_mask: Optional[torch.Tensor] = None,
    weight: float = 1.0,
) -> LossOutput:
    output = _format_loss_output(loss_per_token, attention_mask)

    if think_mask is not None:
        think_output = _format_loss_output(loss_per_token, think_mask)
        output.think_loss = think_output.loss
        output.think_loss_per_token = think_output.loss_per_token
        output.think_valid_tokens = think_output.valid_tokens

    if infer_mask is not None:
        infer_output = _format_loss_output(loss_per_token, infer_mask)
        output.infer_loss = infer_output.loss
        output.infer_loss_per_token = infer_output.loss_per_token
        output.infer_valid_tokens = infer_output.valid_tokens

    for key in output.keys():
        if output[key] is not None or "valid_tokens" in key:
            continue
        if isinstance(output[key], list):
            for i in range(len(output[key])):
                output[key][i] *= weight
        else:
            output[key] *= weight

    return output


def _build_attention_mask(
    attention_mask=None, think_mask=None, infer_mask=None, label_ids=None
) -> torch.Tensor | None:
    if attention_mask is None:
        if label_ids is not None:
            attention_mask = (label_ids != -100).float()
        elif think_mask is not None and infer_mask is not None:
            attention_mask = think_mask + infer_mask
    return attention_mask


class CELoss:
    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(
        self,
        logits: torch.Tensor,
        label_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """计算交叉熵损失

        Args:
            logits (torch.Tensor): 学生 logits
            label_ids (torch.Tensor):  拟合的 label ids
            attention_mask (torch.Tensor):
            think_mask (Optional[torch.Tensor], optional): 标记 label_ids 中 think 的部分
            infer_mask (Optional[torch.Tensor], optional): 标记 label_ids 中 infer 的部分
        """
        logits = logits[:, :-1].contiguous()
        label_ids = label_ids[:, 1:].contiguous()
        if think_mask is not None:
            think_mask = think_mask[:, 1:].contiguous()
        if infer_mask is not None:
            infer_mask = infer_mask[:, 1:].contiguous()
        attention_mask = _build_attention_mask(
            attention_mask, think_mask, infer_mask, label_ids
        )

        # [bsz, seq_len - 1]
        loss_per_token = F.cross_entropy(
            input=logits.reshape(-1, logits.shape[-1]),
            target=label_ids.reshape(-1),
            reduction="none",
        ).reshape(label_ids.shape)

        output = _build_output(
            loss_per_token, attention_mask, think_mask, infer_mask, self.weight
        )

        return output


class LogitsLoss:
    def __init__(self, weight: float = 0.3, temperature: float = 3.0, **kwargs):
        self.temperature = temperature
        self.weight = weight

    def __call__(
        self,
        stu_logits: torch.Tensor,
        tea_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LossOutput:
        """_summary_

        Args:
            stu_logits (torch.Tensor): student 的词表概率分布
            tea_logits (torch.Tensor): teacher 的词表概率分布
            think_mask (Optional[torch.Tensor]): 标记 label_ids 中 think 的部分
            infer_mask (Optional[torch.Tensor]): 标记 label_ids 中 infer 的部分
            reduction (str): mean| sum | none. 默认 mean
        """
        attention_mask = _build_attention_mask(
            attention_mask, think_mask, infer_mask, kwargs.get("label_ids", None)
        )
        device = stu_logits.device
        student_logits_scaled = stu_logits / self.temperature
        teacher_logits_scaled = (tea_logits / self.temperature).to(device)
        log_stu = F.log_softmax(student_logits_scaled, dim=-1)
        prob_tea = F.softmax(teacher_logits_scaled, dim=-1)
        # [bs, seq_len]
        loss_per_token = F.kl_div(log_stu, prob_tea, reduction="none").sum(dim=-1)
        output = _build_output(
            loss_per_token, attention_mask, think_mask, infer_mask, self.weight
        )
        return output


class HiddenLoss:

    def __init__(
        self,
        layer_mapping: Dict[int, int],
        adapter_layers: "MultiLayerAdaptationLayer",
        loss_type: str = "cosine",
        weight: float = 1.0,
        **kwargs,
    ):
        self.layer_mapping = layer_mapping
        self.adapter_layers = adapter_layers
        self.loss_type = loss_type
        self.weight = weight

    def _inter_cosine_loss(
        self,
        stu_hs: torch.Tensor,
        tea_hs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        # [bsz, seq_len]
        loss_per_token = F.cosine_similarity(stu_hs, tea_hs, -1)
        output = _build_output(
            loss_per_token, attention_mask, think_mask, infer_mask, self.weight
        )

        for key in output.keys():
            if "loss" in key and output[key] is not None:
                output[key] = 1 - output[key]
        return output

    def _inter_mse_loss(
        self,
        stu_hs: torch.Tensor,
        tea_hs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
    ):
        # [bsz, seq_len]
        loss_per_token = F.mse_loss(stu_hs, tea_hs, reduction="none").mean(dim=-1)
        output = _build_output(
            loss_per_token, attention_mask, think_mask, infer_mask, self.weight
        )
        return output

    def __call__(
        self,
        stu_feats: Dict[int, torch.Tensor],
        tea_feats: Dict[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        think_mask: Optional[torch.Tensor] = None,
        infer_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attention_mask = _build_attention_mask(
            attention_mask, think_mask, infer_mask, kwargs.get("label_ids", None)
        )
        adapted_feats = (
            self.adapter_layers(stu_feats)
            if self.adapter_layers is not None
            else stu_feats
        )

        all_loss_output = []
        func_map = {
            "mse": self._inter_mse_loss,
            "cosine": self._inter_cosine_loss,
        }

        for stu_idx in sorted(list(adapted_feats.keys()), reverse=False):
            stu_hs = adapted_feats[stu_idx]
            device = stu_hs.device
            tea_idx = self.layer_mapping[stu_idx]
            tea_hs = tea_feats[tea_idx].to(device)

            current_layer_loss = func_map[self.loss_type](
                stu_hs, tea_hs, attention_mask, think_mask, infer_mask
            )
            all_loss_output.append(current_layer_loss)

        output = LossOutput()
        for k in output.keys():
            if "loss_per_token" in k:
                output[k] = []

        for loss_output in all_loss_output:
            for k, v in loss_output.items():
                if "loss_per_token" in k:
                    output[k].append(v)
                elif "loss" in k:
                    if output[k] is None:
                        output[k] = v
                    else:
                        output[k] += v
                else:
                    output[k] = v

        for k in output.keys():
            if "loss" in k and "per_token" not in k:
                output[k] = output[k] / len(adapted_feats)

        return output
