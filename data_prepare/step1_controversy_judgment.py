#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Event Controversy Judgment
"""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter

sys.path.insert(0, str(Path(__file__).parent))
from utils.tools import (
    set_seed,
    build_lang_limit_tag,
    split_datasets,
    get_controversy_output_dir,
)
from utils.pojo import EventSample
from utils.calls import RolePostEventViewLLMCall


def load_config(config_path: str) -> dict:
    config = FileReader.read(config_path)
    set_seed(config["seed"])

    load_dotenv("./.env")
    _config = config["calc_controversy"]
    _config["api_bases"] = os.environ["CONTROVERSY_API_BASES"].split(",")
    _config["models"] = os.environ["CONTROVERSY_API_MODELS"].split(",")
    _config["api_keys"] = os.environ["CONTROVERSY_API_KEYS"].split(",")

    dataset_name = config["dataset"]
    dataset_cfg = config["datasets"][dataset_name]
    limit_tag = build_lang_limit_tag(dataset_cfg["sample_limit_by_lang"])

    config["split_output_dir"] = os.path.join(
        config["output_dir"],
        f"dataset_{dataset_name}",
        "split_events",
        f"limit_{limit_tag}",
    )
    os.makedirs(config["split_output_dir"], exist_ok=True)

    config["controversy_output_dir"] = get_controversy_output_dir(config)
    os.makedirs(config["controversy_output_dir"], exist_ok=True)

    return config


def _load_samples_from_file(file_path: str) -> List[EventSample]:
    samples = []
    for sample in FileReader.read(file_path, return_iter=True):
        samples.append(EventSample(**sample))
    return samples


def _apply_lang_limits(
    samples: List[EventSample],
    limit_by_lang: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> List[EventSample]:
    if limit_by_lang is None:
        return samples

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.lang].append(sample)

    kept = []
    for lang, lang_samples in grouped.items():
        raw_limit = limit_by_lang.get(lang, None)
        if raw_limit is None:
            kept.extend(lang_samples)
            continue
        limit = int(raw_limit)
        if limit <= 0:
            continue
        rng = random.Random(seed + sum(ord(ch) for ch in lang))
        if limit >= len(lang_samples):
            selected = list(lang_samples)
        else:
            selected = rng.sample(lang_samples, limit)
        kept.extend(selected)
    return kept


def _build_split_samples(config: dict) -> dict:
    dataset_name = config["dataset"]
    dataset_cfg = config["datasets"][dataset_name]

    source_type = dataset_cfg["source_type"]
    all_samples = []
    if source_type == "single_file":
        file_paths = [dataset_cfg["file_path"]]
    elif source_type == "multi_files":
        file_paths = dataset_cfg["file_paths"]
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path}, skipping")
            continue
        current = _load_samples_from_file(file_path)
        all_samples.extend(current)

    # e.g. {'en': 20, 'zh': 100}
    limit_by_lang = dataset_cfg.get("sample_limit_by_lang", None)
    limited_samples = _apply_lang_limits(all_samples, limit_by_lang, config["seed"])

    # Group by language & category
    group_by_lang = defaultdict(list)
    for sample in limited_samples:
        category = sample.get("category", "default")
        lang = sample.get("lang", "en")
        group_by_lang[f"{lang}_{category}"].append(sample)

    # Split each language group
    split_config = config["split_events"]
    split_samples = {"train": [], "valid": [], "test": []}
    for lang_samples in group_by_lang.values():
        train_s, valid_s, test_s = split_datasets(
            samples=lang_samples,
            train_ratio=split_config["train_ratio"],
            valid_ratio=split_config["valid_ratio"],
            test_ratio=split_config["test_ratio"],
            seed=config["seed"],
        )
        split_samples["train"].extend(train_s)
        split_samples["valid"].extend(valid_s)
        split_samples["test"].extend(test_s)

    return split_samples


def _calc_event_controversy(config: dict, file_path: str) -> str:
    calc_cfg = config["calc_controversy"]
    split = Path(file_path).stem

    controversy_output_dir = get_controversy_output_dir(config)
    os.makedirs(controversy_output_dir, exist_ok=True)

    output_file_path = os.path.join(config["controversy_output_dir"], f"{split}.jsonl")

    RolePostEventViewLLMCall(
        api_bases=calc_cfg["api_bases"],
        model_names=calc_cfg["models"],
        api_keys=calc_cfg["api_keys"],
        max_worker=calc_cfg["max_workers"],
        generation_config=calc_cfg["generation_config"],
    )(
        output_path=output_file_path,
        samples=FileReader.read(file_path),
        id_field="index",
        desc=f"calc_event_controversy {split}",
    )

    return output_file_path


def main():
    config = load_config("./configs/step1_controversy.yaml")
    print(f"[INFO] Dataset: {config['dataset']}")

    split_output_dir = config["split_output_dir"]
    if not all(
        os.path.exists(os.path.join(split_output_dir, f"{split}.jsonl"))
        for split in config["splits"]
    ):
        split_samples = _build_split_samples(config)
        for split, samples in split_samples.items():
            split_file_path = os.path.join(split_output_dir, f"{split}.jsonl")
            if not os.path.exists(split_file_path):
                FileWriter.dump(samples, split_file_path)
            print(f"[INFO] Split {split}: {len(samples)} samples -> {split_file_path}")

    for split in config["splits"]:
        split_file_path = os.path.join(split_output_dir, f"{split}.jsonl")
        if not os.path.exists(split_file_path):
            print(f"[WARN] Split file not found: {split_file_path}, skipping")
            continue

        controversy_file_path = _calc_event_controversy(config, split_file_path)
        print(f"[DONE] split={split} -> {controversy_file_path}")

    print(f"\n[INFO] Output root: {config['output_dir']}/dataset_{config['dataset']}")


if __name__ == "__main__":
    main()
