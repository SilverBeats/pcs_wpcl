#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Threshold Filtering
Filter high-quality events based on similarity threshold.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils.tools import get_controversy_output_dir


def _get_split_event_dir(controversy_dir: str) -> str:
    # controversy_dir: ./output/dataset_ethics/calc_event_controversy/qwen3-8b/limit_en_all
    # temp_dir: ./output/dataset_ethics/split_events/qwen3-8b/limit_en_all
    temp_dir = controversy_dir.replace("calc_event_controversy", "split_events")
    parts = temp_dir.split("/")
    # split_dir: ./output/dataset_ethics/split_events/limit_en_all
    split_dir = "/".join(parts[:-3] + parts[-1:])
    return split_dir


def load_config(config_path: str) -> dict:
    config = FileReader.read("./configs/step3_filter.yaml")
    load_dotenv(".env")

    controversy_config = FileReader.read(config["controversy_config_path"])
    _config = controversy_config["calc_controversy"]
    _config["api_bases"] = os.environ["CONTROVERSY_API_BASES"].split(",")
    _config["models"] = os.environ["CONTROVERSY_API_MODELS"].split(",")
    _config["api_keys"] = os.environ["CONTROVERSY_API_KEYS"].split(",")

    # init dirs
    controversy_dir = get_controversy_output_dir(controversy_config)
    config["controversy_dir"] = controversy_dir
    config["split_event_dir"] = _get_split_event_dir(controversy_dir)
    config["similarity_score_dir"] = controversy_dir.replace(
        "calc_event_controversy", "calc_similarity"
    )
    config["output_dir"] = os.path.join(
        controversy_dir.replace("calc_event_controversy", "filter_low_quality_event"),
        f"threshold_{config['threshold_std']:.1f}",
    )
    os.makedirs(config["output_dir"], exist_ok=True)

    config["split"] = controversy_config["splits"]

    return config


def calculate_grade_score(
    sim_matrix: np.ndarray, llm_json_result: dict, alpha=0.5, beta=0.5
):
    # 1. 计算文本分歧度 (Divergence)
    mask = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)
    avg_sim = sim_matrix[mask].mean()
    dist_e = 1 - avg_sim  # 范围 [0, 1]

    # 2. 计算规范偏离指数 (NDI)
    weights = {
        "Self-Centered": 0.35,
        "Role Model": 0.30,
        "Reserved": 0.20,
        "Average": 0.15,
    }

    ndi_e = 0.0
    perspectives = llm_json_result.get("perspectives", [])
    for p in perspectives:
        p_type = p.get("type", None)
        if p.get("stance_relative_to_norm", None) == "Deviate":
            ndi_e += weights.get(p_type, 0)  # 范围 [0, 1]

    # 3. 最终加权汇总
    final_score = alpha * dist_e + beta * ndi_e  # 最终范围严格锁定在 [0, 1]
    return final_score


def filter_events(
    event_file_path: str,
    similarity_score_path: str,
    controversy_file_path: str,
    threshold_std: float,
    output_dir: str,
    split_name: Optional[str] = None,
) -> str:
    """Filter events by threshold (mean + std * threshold_std)."""
    if split_name is None:
        split_name = Path(similarity_score_path).stem

    output_file_path = os.path.join(output_dir, f"{split_name}.jsonl")
    log_file_path = os.path.join(output_dir, f"{split_name}.txt")
    fig_file_path = os.path.join(output_dir, f"{split_name}.pdf")

    # If already filtered, skip
    if os.path.exists(output_file_path):
        print(f"[SKIP] Already filtered: {output_file_path}")
        return output_file_path

    index_to_controversy = {
        sample["index"]: sample
        for sample in FileReader.read(controversy_file_path, return_iter=True)
    }

    # Load similarity scores
    scores, samples = [], []
    for sample in FileReader.read(similarity_score_path, return_iter=True):
        sim_matrix = sample.get("sim_matrix", [])
        if not sim_matrix:
            continue
        # score = float(np.asarray(sim_matrix).min())
        score = calculate_grade_score(
            np.asarray(sim_matrix), index_to_controversy[sample["index"]]
        )
        scores.append(score)
        samples.append((score, sample["index"]))

    if not scores:
        print(f"[WARN] No similarity scores found in {similarity_score_path}")
        FileWriter.dump([], output_file_path)
        return output_file_path

    # Calculate threshold
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    median_val = np.median(scores)
    threshold = mean_val + threshold_std * std_val

    # Write log
    with open(log_file_path, "w", encoding="utf-8") as log_fp:
        print(f"过滤前数据量: {len(scores)}", file=log_fp)
        print(f"均值 (Mean): {mean_val:.4f}", file=log_fp)
        print(f"标准差 (Std): {std_val:.4f}", file=log_fp)
        print(f"中位数 (Median): {median_val:.4f}", file=log_fp)
        print(f"阈值 (Threshold): {threshold:.4f}", file=log_fp)
        print(f"threshold_std: {threshold_std}", file=log_fp)
        print(
            f"过滤后保留数量: {sum(1 for s in scores if s >= threshold)}",
            file=log_fp,
        )

    # Draw distribution
    plt.figure(figsize=(12, 6))
    plt.hist(
        scores,
        bins=50,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Distribution",
    )
    x = np.linspace(float(min(scores)), float(max(scores)), 100)
    y = stats.norm.pdf(x, mean_val, std_val)
    plt.plot(
        x,
        y,
        "r-",
        linewidth=2,
        label=f"Normal Fit (μ={mean_val:.2f}, σ={std_val:.2f})",
    )
    plt.axvline(
        float(threshold),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.2f})",
    )
    plt.axvline(float(mean_val), color="gray", linestyle=":", linewidth=1, label="Mean")
    plt.title("Distribution of Minimum Pairwise Similarity Scores")
    plt.xlabel("Similarity Score (Lower = More Controversial)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_file_path, dpi=300)
    plt.close()

    # Load original event samples
    ori_event_samples = {}
    for event in FileReader.read(event_file_path, return_iter=True):
        if isinstance(event, dict) and "index" in event:
            ori_event_samples[event["index"]] = event

    # Filter
    kept_samples = []
    for score, sample_index in samples:
        if score >= threshold and sample_index in ori_event_samples:
            kept_samples.append(ori_event_samples[sample_index])

    # Save
    FileWriter.dump(kept_samples, output_file_path)
    print(f"[INFO] Threshold: {threshold:.4f}")
    print(f"[INFO] Kept {len(kept_samples)} / {len(scores)} events")

    return output_file_path


def main():
    config = load_config("./configs/step3_filter.yaml")

    for split in config["splits"]:
        similarity_score_path = os.path.join(
            config["similarity_score_dir"], f"{split}.jsonl"
        )
        if not os.path.exists(similarity_score_path):
            print(f"[WARN] {similarity_score_path} does not exist")
            continue

        event_file_path = os.path.join(config["split_event_dir"], f"{split}.jsonl")
        if not os.path.exists(event_file_path):
            print(f"[WARN] {event_file_path} does not exist")
            continue

        controversy_file_path = os.path.join(
            config["controversy_dir"], f"{split}.jsonl"
        )
        if not os.path.exists(controversy_file_path):
            print(f"[WARN] {controversy_file_path} does not exist")
            continue

        filter_events(
            event_file_path=event_file_path,
            similarity_score_path=similarity_score_path,
            controversy_file_path=controversy_file_path,
            threshold_std=config["threshold_std"],
            output_dir=config["output_dir"],
            split_name=split,
        )


if __name__ == "__main__":
    main()
