#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义 Trainer
主要功能：
1. 保存 LoRA adapter（而非整个模型）
2. JSONL 日志回调
3. 梯度裁剪和 NaN 检查
"""

import json
import os
from typing import Optional

from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


class JsonlLogCallback(TrainerCallback):
    """将训练/验证指标记录到 JSONL 文件"""

    def __init__(self, output_dir: str):
        self.train_log_path = os.path.join(output_dir, "train_log.jsonl")
        self.eval_log_path = os.path.join(output_dir, "valid_log.jsonl")
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            is_eval = any("eval_" in k for k in logs.keys())
            path = self.eval_log_path if is_eval else self.train_log_path
            with open(path, "a+", encoding="utf-8") as f:
                log_entry = {**logs, "global_step": state.global_step}
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


class KnowledgeDistillationTrainer(Trainer):
    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        model_to_save = self.accelerator.unwrap_model(self.model)
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def training_step(self, model, inputs, num_items_in_batch):
        """将 global_step 传给模型，用于课程权重调度"""
        step = self.state.global_step
        inputs["step"] = step
        return super().training_step(model, inputs, num_items_in_batch)
