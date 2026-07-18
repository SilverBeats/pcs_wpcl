#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立推理脚本（vLLM 后端）
使用 vllm Python 包直接推理，支持多卡。

支持的参数：
- temperature: 采样温度
- top_p: top-p 采样
- max_new_tokens: 最大生成长度
- base_model_path: 基座模型路径
- lora_path: 训练好的 LoRA 路径（可选）
- tensor_parallel_size: 多卡张量并行数
- output_path: 输出文件路径
- batch_size: 批处理大小
- test_input_file: 输入数据文件路径（alpaca 格式）
"""

import argparse
import os
from collections import defaultdict
from typing import List

import jieba
from lwj_tools.evaluators.nlg import (
    NLGEvaluator,
    NLGMetric,
    BertScoreConfig,
    BartScoreConfig,
)
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.data_prepare import TrainSample, load_alpaca_dataset, post_process

GENERATE_FILE_NAME = "my_generations.jsonl"
EVALUATE_FILE_NAME = "my_evaluations_with_think.json"
TASK_TYPE_FILE_NAME = "my_{}_evaluations_with_think.json"

STR_TO_ENUM = {
    "distinct": NLGMetric.DISTINCT,
    "bleu": NLGMetric.BLEU,
    "bert_score": NLGMetric.BERT_SCORE,
    "bart_score": NLGMetric.BART_SCORE,
    "meteor": NLGMetric.METEOR,
    "rouge": NLGMetric.ROUGE,
}


def get_args():
    parser = argparse.ArgumentParser(description="PCS 推理脚本 (vLLM)")
    parser.add_argument(
        "--base_model_path", type=str, required=True, help="基座模型路径"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="训练好的 LoRA 路径（可选）"
    )
    parser.add_argument(
        "--test_input_file",
        type=str,
        default=None,
        help="输入数据文件路径（alpaca 格式的 jsonl）",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="采样温度 (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="top-p 采样 (default: 0.9)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="最大生成长度 (default: 256)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="批处理大小 (default: 8)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行数，即使用 GPU 数量 (default: 1)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU 显存占用比例 (default: 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    parser.add_argument("--do_evaluate", action="store_true")
    parser.add_argument("--do_generate", action="store_true")
    parser.add_argument("--do_split_evaluate", action="store_true")
    parser.add_argument(
        "--evaluate_config_file", type=str, default="./configs/eval_config.yaml"
    )
    parser.add_argument("--gen_file", type=str, default=None)
    args = parser.parse_args()

    if args.gen_file is None or (args.do_evaluate and args.do_generate):
        args.gen_file = os.path.join(args.output_dir, GENERATE_FILE_NAME)
    print(args)
    return args


def build_prompt_from_sample(sample: TrainSample, tokenizer) -> str:
    """将 TrainSample 转换为 prompt 字符串"""
    conversations = [
        {"role": "system", "content": sample.system},
        {"role": "user", "content": sample.instruction},
    ]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_generate(
    llm: LLM,
    tokenizer,
    samples: List[TrainSample],
    output_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 8,
    lora_request=None,
    seed: int = 42,
) -> str:
    """
    批量调用 vLLM 生成，输出 label/predict 格式。
    """
    prompts = [
        build_prompt_from_sample(s, tokenizer)
        for s in tqdm(samples, dynamic_ncols=True, desc="Processing Data Samples")
    ]
    print("Input:")
    print(prompts[0])
    print("Labels:")
    print(samples[0].output)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
        seed=seed,
    )

    results: List[dict] = []

    for i in tqdm(range(0, len(prompts), batch_size), dynamic_ncols=True):
        batch_prompts = prompts[i : i + batch_size]
        batch_samples = samples[i : i + batch_size]

        outputs = llm.generate(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

        for j, output in enumerate(outputs):
            response_text = output.outputs[0].text
            results.append(
                {
                    "input": batch_prompts[j],
                    "label": batch_samples[j].output,
                    "predict": response_text,
                }
            )

    output_path = os.path.join(output_dir, GENERATE_FILE_NAME)
    FileWriter.dump(results, output_path)
    print(f"  Saved {len(results)} results to {output_path}")
    return output_path


def do_generate(args):
    output_path = os.path.join(args.output_dir, GENERATE_FILE_NAME)
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} 已存在")
        return
    print("=" * 60)
    print("PCS 推理配置 (vLLM):")
    print(f"  基座模型: {args.base_model_path}")
    print(f"  LoRA 路径: {args.lora_path}")
    print(f"  输入文件: {args.test_input_file}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-P: {args.top_p}")
    print(f"  Max-New-Tokens: {args.max_new_tokens}")
    print(f"  Batch-Size: {args.batch_size}")
    print(f"  Tensor-Parallel-Size: {args.tensor_parallel_size}")
    print(f"  GPU-Memory-Utilization: {args.gpu_memory_utilization}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    print("加载数据集...")
    samples = load_alpaca_dataset(args.test_input_file, None, None)[0]
    print(f"  样本数: {len(samples)}")

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest(
            lora_name="full",
            lora_int_id=1,
            lora_path=args.lora_path,
        )
        print(f"  LoRA 加载: {args.lora_path}")

    print(f"初始化 vLLM（{args.tensor_parallel_size} GPU）...")
    llm = LLM(
        model=args.base_model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=args.lora_path is not None,
        max_lora_rank=64,
    )

    tokenizer = llm.get_tokenizer()

    print("开始生成...")
    run_generate(
        llm=llm,
        tokenizer=tokenizer,
        samples=samples,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        lora_request=lora_request,
        seed=args.seed,
    )
    print("生成完成!")


def _get_evaluator(args):
    eval_config_file = args.evaluate_config_file
    eval_config = FileReader.read(eval_config_file)
    nlg_evaluator = NLGEvaluator(
        metric_list=[STR_TO_ENUM[item] for item in eval_config["metrics"]],
        verbose=eval_config["verbose"],
        bart_score_config=BartScoreConfig(**eval_config.get("bart_score_config", {})),
        bert_score_config=BertScoreConfig(**eval_config.get("bert_score_config", {})),
        tokenizer=jieba.cut,
    )
    specific_config = {
        STR_TO_ENUM[k]: v for k, v in eval_config["specific_config"].items()
    }
    return nlg_evaluator, specific_config


def do_evaluate(args):
    output_file_path = os.path.join(args.output_dir, EVALUATE_FILE_NAME)
    if os.path.exists(output_file_path):
        print(f"[SKIP] {output_file_path} 已存在")
        return

    print("开始评估")
    nlg_evaluator, specific_config = _get_evaluator(args)
    hyps, refs = [], []
    for sample in FileReader.read(args.gen_file, return_iter=True):
        hyp = post_process(sample.get("predict")).strip()
        ref = post_process(sample.get("label")).strip()
        if len(hyp) == 0 or len(ref) == 0:
            continue
        hyps.append(hyp)
        refs.append(ref)

    metrics = nlg_evaluator(hyps, refs, specific_config=specific_config)
    print(metrics)
    FileWriter.dump(metrics, output_file_path, indent=4)
    print(f"评估完成, 保存至： {output_file_path}")


def do_split_evaluate(args):
    nlg_evaluator, specific_config = _get_evaluator(args)
    group_by_task_type = defaultdict(lambda: defaultdict(list))
    for sample in FileReader.read(args.gen_file, return_iter=True):
        hyp = post_process(sample.get("predict")).strip()
        ref = post_process(sample.get("label")).strip()
        if len(hyp) == 0 or len(ref) == 0:
            continue
        _dict = (
            group_by_task_type["general"]
            if "General Task" in sample["input"]
            else group_by_task_type["personalization"]
        )
        _dict["hyps"].append(hyp)
        _dict["refs"].append(ref)

    for task_type, v_dict in group_by_task_type.items():
        output_file_path = os.path.join(
            args.output_dir, TASK_TYPE_FILE_NAME.format(task_type)
        )
        if os.path.exists(output_file_path):
            continue

        task_type_metrics = nlg_evaluator(
            v_dict["hyps"], v_dict["refs"], specific_config=specific_config
        )

        FileWriter.dump(task_type_metrics, output_file_path, indent=4)
        print(f"[INFO] 评估结果保存至: {output_file_path}")


def main():
    args = get_args()

    if args.do_generate:
        do_generate(args)

    if args.do_evaluate:
        do_evaluate(args)

    if args.do_split_evaluate:
        do_split_evaluate(args)


if __name__ == "__main__":
    main()
