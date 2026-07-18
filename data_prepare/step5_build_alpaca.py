#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from collections import defaultdict
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import List, Dict, Tuple

from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter
from lwj_tools.utils.common import random_choice

sys.path.insert(0, str(Path(__file__).parent))

from utils.tools import (
    set_seed,
    build_index_to_event,
    build_index_to_role,
    convert_personalization_sample_to_pojo,
    build_general_sample_to_pojo,
)
from utils.pojo import DatasetAblationArguments, RoleSample, EventSample, TrainSample


def load_config(config_path: str) -> dict:
    config = FileReader.read(config_path)
    set_seed(config.get("seed", 42))
    config["ablation_config"] = DatasetAblationArguments(
        **config.get("ablation_config", {})
    )

    if config.get("output_dir", None) is None:
        config["output_dir"] = config["debate_source"].replace(
            "debate_generate", "alpaca"
        )

    base_output_dir = config["output_dir"]
    if config["adopt_diff_general_ratio"]:
        general_ratio = config.get("general_ratio", 0)
        config["output_dir"] = os.path.join(
            base_output_dir, "diff_general_ratio", f"general_ratio_{general_ratio}"
        )
        config["splits"] = ["train", "valid"]
    else:
        if all(not v for v in config["ablation_config"].values()):
            sub_dir = "full"
        else:
            config["splits"] = ["train", "valid"]
            sub_dir = None
            for k, v in config["ablation_config"].items():
                if v:
                    sub_dir = k
                    break
            if not sub_dir:
                raise ValueError(
                    f"Invalid ablation_config: {config['ablation_config']}"
                )
        config["output_dir"] = os.path.join(base_output_dir, sub_dir)

    os.makedirs(config["output_dir"], exist_ok=True)
    return config


def log(msg: str, log_file: TextIOWrapper = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def _get_personalization_samples(
    personalization_samples: List[dict],
    split: str,
    ablation_config: DatasetAblationArguments,
    log_file: TextIOWrapper,
    index_to_event: Dict[int, EventSample],
    index_to_role: Dict[int, RoleSample],
) -> List[TrainSample]:
    samples = []
    for sample in personalization_samples:
        try:
            index = sample["index"]
            parts = index.replace("event_", "").split("_role_")
            event_index = int(parts[0])
            role_index = int(parts[1])

            event_sample = index_to_event[event_index]
            role_sample = index_to_role[role_index]

            train_sample = convert_personalization_sample_to_pojo(
                infer=sample["infer"],
                cot=sample["cot"],
                relation=sample["relation"],
                role_profile=role_sample.en,
                event_sample=event_sample,
                ablation_config=ablation_config,
            )
            train_sample.role_idx = role_sample.index
            samples.append(train_sample)
        except Exception as e:
            log(f"[{split}] ERROR parsing personalization sample: {e}", log_file)
    return samples


def build_alpaca_dataset(
    config: dict, split: str, log_file: TextIOWrapper
) -> Tuple[list, list]:
    debate_source = config["debate_source"]
    event_source = config["event_source"]
    role_source = config["role_source"]
    ablation_config = config["ablation_config"]
    general_source = config["general_source"]

    personalization_path = os.path.join(debate_source, f"{split}.jsonl")
    general_path = os.path.join(general_source, f"{split}.jsonl")

    # Build index mappings
    event_file = os.path.join(event_source, f"{split}.jsonl")
    role_file = os.path.join(role_source, f"{split}.jsonl")

    index_to_event = build_index_to_event(event_file)
    index_to_role = build_index_to_role(role_file)

    log(
        f"[{split}] Events: {len(index_to_event)}, Roles: {len(index_to_role)}",
        log_file,
    )

    ori_personalization_samples = FileReader.read(personalization_path)
    ori_general_samples = FileReader.read(general_path)

    if config["adopt_diff_general_ratio"]:
        general_count = int(config["total_samples"][split] * config["general_ratio"])
        personalization_count = config["total_samples"][split] - general_count
        ori_general_samples = random_choice(ori_general_samples, general_count)

        group_by_event = defaultdict(lambda: defaultdict(list))
        for sample in ori_personalization_samples:
            index = sample["index"]
            parts = index.replace("event_", "").split("_role_")
            event_index = int(parts[0])
            role_index = int(parts[1])
            group_by_event[event_index][role_index].append(sample)

        ori_personalization_samples = []
        group_by_event = list(group_by_event.items())
        need = personalization_count
        while need != 0:
            index, values = group_by_event.pop(0)
            for v in values.values():
                if len(v) <= need:
                    ori_personalization_samples.extend(v)
                    need -= len(v)
                else:
                    ori_personalization_samples.extend(v[:need])
                    need = 0
                    break
    else:
        if ablation_config.wo_general:
            ori_general_samples = []

    personalization_samples = _get_personalization_samples(
        ori_personalization_samples,
        split,
        ablation_config,
        log_file,
        index_to_event,
        index_to_role,
    )

    general_samples = []
    for sample in ori_general_samples:
        try:
            train_sample = build_general_sample_to_pojo(sample)
            if train_sample:
                general_samples.append(train_sample)
        except Exception as e:
            log(f"[{split}] ERROR parsing general sample: {e}", log_file)
            continue

    return personalization_samples, general_samples


def main():
    config = load_config("./configs/step5_build_alpaca.yaml")
    output_dir = config["output_dir"]

    log_file_path = os.path.join(output_dir, "build_alpaca_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log(
            f"Start Building Alpaca Format Dataset | output_dir: {output_dir}", log_file
        )
        log(f"seed: {config['seed']}", log_file)
        log(f"ablation_config: {config['ablation_config']}", log_file)

        total_stats = {"personalization": 0, "general": 0}

        for split in config["splits"]:
            log(f"{'='*50}", log_file)
            log(f"SPLIT: {split.upper()}", log_file)

            try:
                personalization_samples, general_samples = build_alpaca_dataset(
                    config, split, log_file
                )

                # Save split dataset
                split_samples = []
                split_samples.extend(general_samples)
                split_samples.extend(personalization_samples)

                split_output = os.path.join(output_dir, f"{split}.jsonl")
                FileWriter.dump(
                    split_samples, split_output, default=TrainSample.to_dict
                )

                log(
                    f"[{split}] Saved {len(split_samples)} samples to {split_output}",
                    log_file,
                )
                total_stats["personalization"] += len(personalization_samples)
                total_stats["general"] += len(general_samples)
            except Exception as e:
                log(f"[{split}] ERROR: {e}", log_file)
                import traceback

                traceback.print_exc()

        log("=" * 50, log_file)
        log(
            f"Build Complete | personalization={total_stats['personalization']}, "
            f"general={total_stats['general']}",
            log_file,
        )

    print(f"[INFO] Build complete")
    print(f"[INFO] Log saved: {log_file_path}")


if __name__ == "__main__":
    main()
