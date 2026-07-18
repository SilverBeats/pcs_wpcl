#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
from functools import partial
from typing import Optional
from typing import Tuple

from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter
from lwj_tools.utils.common import str2bool
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    set_seed,
    HfArgumentParser,
)

from models.model import StudentModel
from utils.args import (
    CustomTrainingArguments,
    DatasetAblationArguments,
    DistillConfigArguments,
    ModelArguments,
)
from utils.data_prepare import PCSDataset, load_alpaca_dataset
from utils.tools import log
from utils.trainer import JsonlLogCallback, KnowledgeDistillationTrainer


def resolve_data_file_path(
    split: str,
    base_dir: str,
    ablation_name: str,
    mode: str = "auto",
    file_path: Optional[str] = None,
):
    if mode == "direct":
        if file_path is None:
            raise ValueError("If you set mode = 'direct', you must provide a file_path")
        return file_path

    data_path = os.path.join(base_dir, ablation_name, f"{split}.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    return data_path


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--personalization_pure", type=str2bool, default=False)
    parser.add_argument("--distill_method", type=str)
    parser.add_argument("--plugins", type=lambda v: v.split(), default=[])
    return parser.parse_args()


def load_config(
    config_path,
) -> Tuple[
    ModelArguments,
    CustomTrainingArguments,
    DistillConfigArguments,
    DatasetAblationArguments,
    dict,
]:
    config = FileReader.read(config_path)

    custom_args = get_args()
    config["distill_method"] = custom_args.distill_method
    config["personalization_pure"] = custom_args.personalization_pure
    config["plugins"] = custom_args.plugins

    model_name = config["distill_method"]
    for plugin in config["plugins"]:
        model_name += f"_{plugin}"
    config["output_dir"] = os.path.join(config["output_dir"], model_name)

    set_seed(config.get("seed", 42))
    data_args = config.pop("data")
    ablation_args = DatasetAblationArguments(**config.pop("ablation"))

    # 处理文件路径
    ablation_name = ablation_args.to_exp_name()
    base_dir = data_args["base_dir"]
    for split in ["train", "valid"]:
        data_args[f"{split}_file_path"] = resolve_data_file_path(
            split=split,
            base_dir=base_dir,
            ablation_name=ablation_name,
            mode=data_args[split].get("mode", "auto"),
            file_path=data_args[split].get("file_path", None),
        )
    # ./data_prepare/output/dataset_ethics/alpaca/qwen3-8b/llmit_en_all/threshold_1.0/sample_role_5/full/train.jsonl
    config["output_dir"] = os.path.join(
        config["output_dir"], *data_args["train_file_path"].split("/")[-6:-1]
    )
    os.makedirs(config["output_dir"], exist_ok=True)
    parser = HfArgumentParser(
        (ModelArguments, CustomTrainingArguments, DistillConfigArguments)
    )
    model_args, training_args, distill_args = parser.parse_dict(config)
    if training_args.local_rank == 0:
        FileWriter.dump(config, os.path.join(config["output_dir"], "config.yaml"))
        if training_args.deepspeed:
            ds_config_path = training_args.deepspeed
            ds_config = FileReader.read(ds_config_path)
            output_file_path = os.path.join(
                config["output_dir"], os.path.basename(ds_config_path)
            )
            FileWriter.dump(ds_config, output_file_path)
    return model_args, training_args, distill_args, ablation_args, data_args


def main():
    model_args, training_args, distill_args, ablation_args, data_args = load_config(
        "./configs/base_train.yaml"
    )
    rank = training_args.local_rank

    log_path = str(os.path.join(training_args.output_dir, "log.txt"))
    log("=" * 60, rank, log_path)
    log(f"Model Args: {model_args}", rank, log_path)
    log(f"Training Args: {training_args}", rank, log_path)
    log(f"Distillation Args: {distill_args}", rank, log_path)
    log(f"Ablation Args: {ablation_args}", rank, log_path)
    log(f"Data Args: {data_args}", rank, log_path)

    log("加载 Tokenizer...", rank, log_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.student_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("构建训练集...", rank, log_path)
    train_dataset, train_skip_count = load_alpaca_dataset(
        data_args["train_file_path"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        rank=rank,
    )

    log("构建验证集...", rank, log_path)
    valid_dataset, valid_skip_count = load_alpaca_dataset(
        data_args["valid_file_path"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        rank=rank,
    )

    log(f"训练集样本数: {len(train_dataset)}, 跳过: {train_skip_count}", rank, log_path)
    log(f"训练集样例: {train_dataset[-1]}", rank, log_path)
    log(f"验证集样本数: {len(valid_dataset)}, 跳过: {valid_skip_count}", rank, log_path)
    log(f"验证集样例: {valid_dataset[0]}", rank, log_path)

    collate_fn = partial(
        PCSDataset.collate_fn,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
    )

    log("加载学生模型...", rank, log_path)
    # lora_config = {
    #     "r": distill_args.lora_r,
    #     "lora_alpha": distill_args.lora_alpha,
    #     "lora_dropout": distill_args.lora_dropout,
    #     "target_modules": distill_args.lora_target_modules,
    #     "task_type": distill_args.lora_task_type,
    #     "bias": distill_args.lora_bias,
    # }

    model = StudentModel(
        name_or_path=model_args.student_name_or_path,
        teacher_name_or_path=model_args.teacher_name_or_path,
        distill_config=distill_args,
        target_layer_indexes=model_args.target_layer_indexes,
        rank=rank,
    )

    # 设置课程权重调度的总步数
    if distill_args.curriculum_config.get("enabled", False):
        total_steps = int(
            len(train_dataset)
            / (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
            * training_args.num_train_epochs
        )
        model.set_total_steps(total_steps)

    callbacks = []
    if training_args.patience > 0:
        callbacks.append(
            EarlyStoppingCallback(training_args.patience, training_args.delta)
        )
    callbacks.append(JsonlLogCallback(training_args.output_dir))

    trainer = KnowledgeDistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    if training_args.do_train:
        log("开始训练...", rank, log_path)
        trainer.train()
        trainer.save_model()
        log("训练完成!", rank, log_path)


if __name__ == "__main__":
    main()
