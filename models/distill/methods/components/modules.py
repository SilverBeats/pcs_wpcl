#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict

import torch
import torch.nn as nn


def create_layer_mapping(
    num_student_layers: int, num_teacher_layers: int
) -> Dict[int, int]:
    return {
        i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
        for i in range(num_student_layers)
    }


class MultiLayerAdaptationLayer(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        target_student_layers: List[int],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(student_dim, teacher_dim),
                    nn.LayerNorm(teacher_dim),
                )
                for _ in range(len(target_student_layers))
            ]
        )
        self.dtype = dtype
        self.distill_stu_layer_idx_to_proj_idx = {
            stu_idx: proj_idx for proj_idx, stu_idx in enumerate(target_student_layers)
        }
        self._init_weights()
        self.projections = self.projections.to(dtype=self.dtype)

    def _init_weights(self):
        for block in self.projections:
            for module in block:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def align_dim(self, hidden_state: torch.Tensor, proj_idx: int) -> torch.Tensor:
        ori_type = hidden_state.dtype
        hidden_state = hidden_state.to(self.dtype)
        hidden_state = self.projections[proj_idx](hidden_state)
        hidden_state = hidden_state.to(ori_type)
        return hidden_state

    def forward(
        self, student_hidden_states: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        adapted_hidden_states = {}
        for stu_idx, hidden_state in student_hidden_states.items():
            proj_idx = self.distill_stu_layer_idx_to_proj_idx[stu_idx]
            adapted_hidden_states[stu_idx] = self.align_dim(hidden_state, proj_idx)
        return adapted_hidden_states
