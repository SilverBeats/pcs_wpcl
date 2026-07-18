#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
from lwj_tools.utils.model import freeze_model
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from utils.args import DistillConfigArguments
from utils.tools import log
from .distill import (
    ContrastivePlugin,
    CurriculumPlugin,
    UncertaintyPlugin,
    Plugin,
    Scope,
    SeqKDMethod,
    KDMethod,
    FeatKDMethod,
    DistillMethodOutput,
    create_layer_mapping,
    MultiLayerAdaptationLayer,
)

NAME_TO_PLUGIN = {
    "contrastive": ContrastivePlugin,
    "curriculum": CurriculumPlugin,
    "uncertainty": UncertaintyPlugin,
}


class StudentModel(nn.Module):
    def __init__(
        self,
        name_or_path: str,
        teacher_name_or_path: str,
        target_layer_indexes: list,
        distill_config: DistillConfigArguments,
        rank: int = 0,
        **kwargs,
    ):
        """学生模型

        Args:
            name_or_path (str):  学生模型路径/名称
            teacher_name_or_path (str): 教师模型路径/名称
            target_layer_indexes (list): 学生模型蒸馏的目标层
            distill_config (DistillConfigArguments): 蒸馏配置
            rank (int, optional): 针对多卡，日志输出.
        """
        super().__init__()
        self.rank = rank
        self.distill_config = distill_config
        self._init_student_model(name_or_path)
        self._init_teacher_model_if_needed(teacher_name_or_path, target_layer_indexes)
        self._init_distill_method()
        self._init_plugins()

    @property
    def _need_teacher(self) -> bool:
        return (
            self.distill_config.adopt_hidden_loss
            or self.distill_config.adopt_logits_loss
            or self.distill_config.contrastive_config.enabled
            or self.distill_config.uncertainty_config.enabled
        )

    def _init_student_model(self, name_or_path: str):
        """初始化 student 模型（LoRA）"""
        base_model = AutoModelForCausalLM.from_pretrained(
            name_or_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model = get_peft_model(
            base_model,
            LoraConfig(**self.distill_config.lora_config.to_dict()),
        )
        self.config = self.model.config
        if self.rank == 0:
            self.model.print_trainable_parameters()

        self._teacher_feats = {}
        self._student_feats = {}
        self._hooks = []

    def _init_teacher_model_if_needed(
        self, teacher_name_or_path: str, target_layer_indexes: List[int]
    ):
        """按需加载 teacher 模型"""
        self.teacher_model = None
        self.layer_mapping = None
        self.adapter_layers = None
        self.target_layer_indexes = None

        if not self._need_teacher:
            return

        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name_or_path, dtype=torch.bfloat16, trust_remote_code=True
        ).eval()
        freeze_model(self.teacher_model)
        if (
            self.distill_config.adopt_hidden_loss
            or self.distill_config.contrastive_config.enabled
        ):
            self._setup_hidden_distillation(target_layer_indexes)

    def _setup_hidden_distillation(self, target_layer_indexes: List[int]):
        num_stu = self.base_model.config.num_hidden_layers
        num_tea = self.teacher_model.config.num_hidden_layers

        self.target_layer_indexes = (
            [list(range(num_stu))[i] for i in target_layer_indexes]
            if target_layer_indexes
            else list(range(num_stu))
        )

        self.layer_mapping = create_layer_mapping(num_stu, num_tea)

        stu_hs = self.base_model.config.hidden_size
        tea_hs = self.teacher_model.config.hidden_size
        if stu_hs != tea_hs:
            self.adapter_layers = MultiLayerAdaptationLayer(
                stu_hs, tea_hs, self.target_layer_indexes, dtype=torch.bfloat16
            )

        self._register_hooks()

    def _register_hooks(self):
        tea_layers = self.teacher_model.model.layers
        stu_layers = self.base_model.model.layers
        for stu_idx in self.target_layer_indexes:
            tea_idx = self.layer_mapping[stu_idx]
            self._hooks.append(
                tea_layers[tea_idx].register_forward_hook(
                    lambda m, i, o, idx=tea_idx: self._teacher_feats.update(
                        {idx: o.detach()}
                    )
                )
            )
            self._hooks.append(
                stu_layers[stu_idx].register_forward_hook(
                    lambda m, i, o, idx=stu_idx: self._student_feats.update({idx: o})
                )
            )

    def _init_distill_method(self):
        method_map = {"seqkd": SeqKDMethod, "kd": KDMethod, "featkd": FeatKDMethod}
        self.distill_method = method_map.get(
            self.distill_config.distill_method, FeatKDMethod
        )(self, self.distill_config)

    def _init_plugins(self):
        for plugin_name in self.distill_config.plugins:
            plugin_config = getattr(self.distill_config, f"{plugin_name}_config", None)
            if plugin_config is not None and plugin_config.enabled:
                if (
                    hasattr(plugin_config, "student_hidden_size")
                    and plugin_config["student_hidden_size"] is None
                ):
                    plugin_config.student_hidden_size = (
                        self.base_model.config.hidden_size
                    )
                if (
                    hasattr(plugin_config, "teacher_hidden_size")
                    and plugin_config["teacher_hidden_size"] is None
                ):
                    plugin_config.teacher_hidden_size = (
                        self.teacher_model.config.hidden_size
                    )
                plugin = NAME_TO_PLUGIN[plugin_name](plugin_config)
                setattr(self, f"{plugin_name}_plugin", plugin)

    @property
    def base_model(self):
        return self.model.base_model.model

    def set_total_steps(self, total_steps: int):
        if hasattr(self, "curriculum_plugin") and self.curriculum_plugin.enabled:
            self.curriculum_plugin.set_total_steps(total_steps)

    @torch.no_grad()
    def _get_teacher_outputs(self, input_ids, attention_mask=None):
        return (
            self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            if self._need_teacher
            else None
        )

    def _get_last_layer_hidden_states(self) -> Tuple[Any, Any]:
        stu_layer_num = self.base_model.config.num_hidden_layers
        stu_layer_idx = stu_layer_num - 1
        stu_hidden = self._student_feats.get(stu_layer_idx, None)

        if self.teacher_model is None:
            tea_hidden = None
        else:
            if self.layer_mapping is None:
                tea_layer_num = self.teacher_model.config.num_hidden_layers
                tea_layer_idx = tea_layer_num - 1
            else:
                tea_layer_idx = self.layer_mapping[stu_layer_idx]
            tea_hidden = self._teacher_feats.get(tea_layer_idx, None)
        return stu_hidden, tea_hidden

    def _get_plugins_by_scope(self, scope: Scope):
        return [
            getattr(self, f"{name}_plugin")
            for name in self.distill_config.plugins
            if hasattr(self, f"{name}_plugin")
            and getattr(self, f"{name}_plugin").SCOPE == scope
        ]

    def _run_plugins(self, scope: Scope, *args, **kwargs):
        results = []
        for plugin in self._get_plugins_by_scope(scope):
            ret = plugin.inject(*args, **kwargs)
            if ret is not None:
                results.append(ret)
        return results

    def train(self, mode: bool = True):
        self.training = mode
        self.model.train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
            freeze_model(self.teacher_model)
        if self.adapter_layers is not None:
            self.adapter_layers.train(mode)
        for plugin_name in self.distill_config.plugins:
            if hasattr(self, f"{plugin_name}_plugin"):
                plugin = getattr(self, f"{plugin_name}_plugin")
                plugin.train(mode)
        return self

    def save_pretrained(self, ckpt_dir: str, *args, **kwargs):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)

        if self.distill_config.adopt_hidden_loss:
            with open(f"{ckpt_dir}/student_teacher_layer_mapping.json", "w") as f:
                json.dump(self.layer_mapping or {}, f)
            if self.adapter_layers is not None:
                torch.save(
                    self.adapter_layers.state_dict(), f"{ckpt_dir}/distill_adapters.pt"
                )

        for plugin_name in self.distill_config.plugins:
            if hasattr(self, f"{plugin_name}_plugin"):
                plugin: Plugin = getattr(self, f"{plugin_name}_plugin")
                if plugin is not None and plugin.enabled:
                    plugin.save_pretrained(ckpt_dir)

    def _process_loss_by_task(
        self,
        task_id: int,
        task_type_ids: torch.Tensor,
        distill_output: DistillMethodOutput,
        tea_logits: Optional[torch.Tensor] = None,
        step: int = 0,
    ):
        indices = task_type_ids == task_id
        if (
            (self.distill_config.general_pure and task_id == 0)
            or (self.distill_config.personalization_pure and task_id == 1)
            or (indices.sum().int().item() == 0)
        ):
            return

        for plugin in self._get_plugins_by_scope(Scope.loss):
            plugin.inject(
                distill_output=distill_output,
                tea_logits=tea_logits,
                step=step,
                indices=indices,
                is_training=self.training,
            )

    def _sum_loss_from_distill_output(
        self,
        distill_output: DistillMethodOutput,
        dtype,
        device,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, dtype=dtype, device=device)
        for distill_method, loss_output in distill_output.items():
            if loss_output is None:
                continue
            if (
                self.distill_config.general_pure
                and self.distill_config.personalization_pure
            ):
                total_loss += loss_output.loss
            else:
                curriculum_plugin = (
                    self.curriculum_plugin
                    if hasattr(self, "curriculum_plugin")
                    else None
                )
                if curriculum_plugin is not None and curriculum_plugin.enabled:
                    # 使用 thnk_loss_per_token, infer_loss_per_token
                    if isinstance(loss_output.think_loss_per_token, list):
                        # [bsz, ]
                        think_loss = sum(loss_output.think_loss_per_token).sum(-1)
                        think_loss = think_loss / loss_output.think_valid_tokens
                        think_loss = think_loss.mean()  # scalar
                    else:
                        think_loss = (
                            loss_output.think_loss_per_token.sum(-1)
                            / loss_output.think_valid_tokens
                        )
                        think_loss = think_loss.mean()

                    if isinstance(loss_output.infer_loss_per_token, list):
                        # [bsz, ]
                        infer_loss = sum(loss_output.infer_loss_per_token).sum(-1)
                        infer_loss = infer_loss / loss_output.infer_valid_tokens
                        infer_loss = infer_loss.mean()  # scalar
                    else:
                        infer_loss = (
                            loss_output.infer_loss_per_token.sum(-1)
                            / loss_output.infer_valid_tokens
                        )
                        infer_loss = infer_loss.mean()
                    method_loss = think_loss + infer_loss
                else:
                    if isinstance(loss_output.loss_per_token, list):
                        # [layers, bsz] -> [bsz, ]
                        method_loss = torch.stack(
                            [
                                loss_per_token.sum(-1) / loss_output.valid_tokens
                                for loss_per_token in loss_output.loss_per_token
                            ],
                            dim=0,
                        ).mean(0)
                    else:
                        method_loss = (
                            loss_output.loss_per_token.sum(-1)
                            / loss_output.valid_tokens
                        )
                    method_loss = method_loss.mean()

                total_loss = total_loss + method_loss
        return total_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_ids: torch.Tensor,
        think_mask: torch.Tensor,
        infer_mask: torch.Tensor,
        task_type_ids: torch.Tensor,
        samples: Optional[List[dict]] = None,
        step: int = 0,
        **kwargs,
    ):
        try:
            self._teacher_feats.clear()
            self._student_feats.clear()

            stu_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
            )
            tea_out = self._get_teacher_outputs(input_ids, attention_mask)

            # 原地修改distill_output
            distill_output = self.distill_method(
                stu_logits=stu_out.logits,
                tea_logits=None if not self._need_teacher else tea_out.logits,
                stu_feats=self._student_feats,
                tea_feats=self._teacher_feats,
                label_ids=label_ids,
                think_mask=think_mask,
                infer_mask=infer_mask,
            )
            for task_id in range(2):
                self._process_loss_by_task(
                    task_id=task_id,
                    tea_logits=tea_out.logits if tea_out is not None else None,
                    task_type_ids=task_type_ids,
                    distill_output=distill_output,
                    step=step,
                )

            dtype, device = stu_out.logits.dtype, input_ids.device
            total_loss = self._sum_loss_from_distill_output(
                distill_output, dtype, device
            )

            # 拿到学生和教师最后一层的 hidden states, 用于 对比学习
            stu_hidden, tea_hidden = self._get_last_layer_hidden_states()
            post_loss_results = self._run_plugins(
                Scope.post_loss,
                samples=samples,
                stu_hidden=stu_hidden,
                tea_hidden=tea_hidden,
                attention_mask=attention_mask,
            )
            for extra_loss in post_loss_results:
                if extra_loss is not None:
                    total_loss = total_loss + extra_loss

            return CausalLMOutputWithPast(total_loss)
        except Exception as e:
            log(str(e), self.rank)
            raise e
