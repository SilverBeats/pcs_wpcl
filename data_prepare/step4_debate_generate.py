#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Teacher Debate Generation
Generate teacher model responses with debate (9 ATOMIC dimensions).
"""

import os
import random
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter

sys.path.insert(0, str(Path(__file__).parent))
from utils.tools import (
    set_seed,
    get_controversy_output_dir,
    convert_role_to_pojo,
    convert_event_to_pojo,
)
from utils.calls import DebateGenerateSampleLLMCall
from utils.pojo import DebateSample, RoleSample, EventSample


def _resolver_filter_event_dir(
    controversy_config_path: str, filter_config_path: str
) -> str:
    controversy_config = FileReader.read(controversy_config_path)
    _config = controversy_config["calc_controversy"]
    _config["models"] = os.environ["CONTROVERSY_API_MODELS"].split(",")
    controversy_dir = get_controversy_output_dir(controversy_config)

    filter_config = FileReader.read(filter_config_path)
    threshold_std = filter_config["threshold_std"]

    event_dir = os.path.join(
        controversy_dir.replace("calc_event_controversy", "filter_low_quality_event"),
        f"threshold_{threshold_std:.1f}",
    )
    return event_dir


def load_config(config_path: str) -> dict:
    config = FileReader.read(config_path)
    set_seed(config["seed"])
    load_dotenv(".env")

    # resolve dirs
    config["event_dir"] = _resolver_filter_event_dir(**config["event_source"])
    sample_role_cnt = config["debate_generate"]["sample_role_cnt"]
    config["output_dir"] = os.path.join(
        config["event_dir"].replace("filter_low_quality_event", "debate_generate"),
        f"sample_role_{sample_role_cnt}",
    )
    os.makedirs(config["output_dir"], exist_ok=True)

    return config


def build_debate_samples(
    roles: List[RoleSample],
    events: List[EventSample],
    sample_role_cnt: int,
    seed: int = 42,
) -> List[DebateSample]:
    """Build debate samples by pairing roles with events."""
    samples = []
    rng = random.Random(seed)

    for event in events:
        # Sample roles
        if sample_role_cnt >= len(roles):
            sampled_roles = roles
        else:
            sampled_roles = rng.sample(roles, sample_role_cnt)

        for role in sampled_roles:
            samples.append(
                DebateSample(
                    index=f"event_{event.index}_role_{role.index}",
                    event=event,
                    role=role,
                )
            )
    return samples


def post_process_raw_debate(raw_file_path: str) -> str:
    """
    Convert raw debate output to training format.

    Data cleaning rules:
    - Skip empty/error/unparseable results
    - Skip DISCARD status
    - For ADJUST status: use final; for KEEP status: use initial
    - Keep: index, relation (dimension), cot (rationale), infer (final/initial)
    """
    output_file_path = raw_file_path.replace("raw_", "")

    if os.path.exists(output_file_path):
        print(f"[SKIP] Already processed: {output_file_path}")
        return output_file_path

    global_total = global_empty = global_discard = global_keep = global_adjust = 0
    arr = []
    for sample in FileReader.read(raw_file_path, return_iter=True):
        global_total += 1

        result = sample.get("result", {})
        if not result:
            global_empty += 1
            continue

        debate_results = result.get("debate_results", {})
        if not debate_results:
            global_empty += 1
            continue

        _current_total = len(debate_results)
        _current_empty = _current_discard = _current_keep = _current_adjust = 0
        for relation, relation_results in debate_results.items():
            if not isinstance(relation_results, dict):
                _current_empty += 1
                continue

            status = str(relation_results.get("status", "")).upper()

            if not status or status == "DISCARD":
                _current_discard += 1
                continue

            cot = relation_results.get("rationale", "").strip()
            if not cot:
                _current_empty += 1
                continue

            infer = relation_results.get(
                "final" if status == "ADJUST" else "initial", ""
            ).strip()

            if not infer:
                _current_empty += 1
                continue

            if status == "KEEP":
                _current_keep += 1
            else:
                _current_adjust += 1

            out_record = {
                "index": sample["index"],
                "relation": relation,
                "cot": cot,
                "infer": infer,
            }
            arr.append(out_record)

        global_discard += _current_discard / _current_total
        global_keep += _current_keep / _current_total
        global_adjust += _current_adjust / _current_total
        global_empty += _current_empty / _current_total

    FileWriter.dump(arr, output_file_path)

    print(f"[INFO] 应当获得的数据总量: {global_total * 9}")
    print(f"[INFO] 实际获得的数据总量: {len(arr)}")
    print(f"[INFO] EMPTY占比: {global_empty / global_total}")
    print(f"[INFO] DISCARD 占比: {global_discard / global_total}")
    print(f"[INFO] KEEP占比: {global_keep / global_total}")
    print(f"[INFO] ADJUST 占比: {global_adjust / global_total}")

    return output_file_path


def main():
    config = load_config("./configs/step4_debate.yaml")
    print(f"[INFO] Sample Role Count: {config['sample_role_cnt']}")
    print(f"[INFO] Event Dir: {config['event_dir']}")
    print(f"[INFO] Role Dir: {config['role_dir']}")
    print(f"[INFO] Output Dir: {config['output_dir']}")

    role_dir = config["role_dir"]
    event_dir = config["event_dir"]
    output_dir = config["output_dir"]

    for split in config["splits"]:
        temp_file = os.path.join(output_dir, f"{split}_temp.jsonl")
        if os.path.exists(temp_file):
            continue
        role_file = os.path.join(role_dir, f"{split}.jsonl")
        event_file = os.path.join(event_dir, f"{split}.jsonl")

        if not os.path.exists(role_file):
            print(f"[WARN] Role file not found: {role_file}, skipping {split}")
            continue
        if not os.path.exists(event_file):
            print(f"[WARN] Event file not found: {event_file}, skipping {split}")
            continue

        # Load roles
        roles = [
            convert_role_to_pojo(role_sample)
            for role_sample in FileReader.read(role_file, return_iter=True)
        ]
        print(f"[INFO] Split {split}: {len(roles)} roles")

        events = [
            convert_event_to_pojo(event_sample)
            for event_sample in FileReader.read(event_file, return_iter=True)
        ]
        print(f"[INFO] Split {split}: {len(events)} events")
        # Build debate samples
        debate_samples = build_debate_samples(
            roles, events, config["sample_role_cnt"], config["seed"]
        )
        print(f"[INFO] Split {split}: {len(debate_samples)} debate samples")

        FileWriter.dump(debate_samples, temp_file, default=DebateSample.to_dict)

    for split in config["splits"]:
        temp_file = os.path.join(output_dir, f"{split}_temp.jsonl")
        raw_output_path = os.path.join(output_dir, f"raw_{split}.jsonl")
        DebateGenerateSampleLLMCall(
            api_bases=config["debate_generate"]["api_bases"],
            model_names=config["debate_generate"]["models"],
            api_keys=config["debate_generate"].get("api_keys"),
            max_worker=config["debate_generate"]["max_workers"],
            generation_config=config["debate_generate"]["generation_config"],
        )(
            file_path=temp_file,
            output_path=raw_output_path,
        )

        # Post-process
        final_output = post_process_raw_debate(raw_output_path)
        print(f"[DONE] {split}: {final_output}")
        print("=" * 20)

    print(f"[INFO] All debate generation complete")
    print(f"[INFO] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
