#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter
from lwj_tools.utils.common import random_choice

sys.path.insert(0, str(Path(__file__).parent))
from utils.tools import set_seed, split_datasets
from utils.constant import (
    ARCHETYPES,
    JOB_CATEGORIES,
    CULTURAL_CLUSTERS,
    GENDERS,
    AGES,
    EN_TO_ZH_MAP,
)
from utils.calls import GenerateRolesLLMCall


def load_config(config: dict, config_path: str) -> dict:
    config = FileReader.read(config_path)
    set_seed(config.get("seed", 42))

    load_dotenv("./.env")
    config["create_roles"]["api_bases"] = os.environ["CREATE_ROLE_API_BASES"].split(",")
    config["create_roles"]["models"] = os.environ["CREATE_ROLE_API_MODELS"].split(",")
    config["create_roles"]["api_keys"] = os.environ["CREATE_ROLE_API_KEYS"].split(",")

    config["output_dir"] = os.path.join(
        config["output_dir"], str(config["per_arch_type_cnt"])
    )
    os.makedirs(config["output_dir"], exist_ok=True)
    return config


def generate_role_configs(config: dict, output_path: str):
    index = 0
    personas = []
    for arch_name, arch_traits in ARCHETYPES.items():
        for _ in range(config["per_arch_type_cnt"]):
            random_job = random_choice(list(JOB_CATEGORIES.keys()))[0]
            random_culture = random_choice(list(CULTURAL_CLUSTERS.keys()))[0]
            config = {
                "index": index,
                "archetype": arch_name,
                "arch_traits": arch_traits["ocean"],
                "arch_desc": arch_traits["desc"],
                "age": random_choice(AGES)[0],
                "gender": random_choice(GENDERS)[0],
                "job_category": random_job,
                "job_desc": JOB_CATEGORIES[random_job],
                "culture": random_culture,
                "culture_countries": CULTURAL_CLUSTERS[random_culture],
            }
            personas.append(config)
            index += 1

    FileWriter.dump(personas, output_path)


def generate_role_profile(config: dict, role_config_path: str, output_path: str):
    GenerateRolesLLMCall(
        api_bases=config["create_roles"]["api_bases"],
        model_names=config["create_roles"]["models"],
        api_keys=config["create_roles"]["api_keys"],
        max_worker=config["create_roles"]["max_workers"],
        generation_config=config["create_roles"]["generation_config"],
    )(role_config_path, output_path, desc="Generate role profile")


def split_roles(config, role_profiles_path: str):
    # prepare zh version
    role_profiles = []
    for sample in FileReader.read(role_profiles_path, return_iter=True):
        keys = [
            "archetype",
            "arch_traits",
            "arch_desc",
            "age",
            "gender",
            "job_category",
            "job_desc",
            "culture",
            "culture_countries",
        ]
        try:
            role_profiles.append(
                {
                    "index": sample["index"],
                    "en": {**sample["en"], **{k: sample[k] for k in keys}},
                    "zh": {
                        **sample["zh"],
                        **{k: EN_TO_ZH_MAP[sample[k]] for k in keys},
                    },
                }
            )
        except Exception:
            print(sample["index"])

    train, valid, test = split_datasets(
        samples=role_profiles,
        train_ratio=config["split_role_profiles"]["train_ratio"],
        valid_ratio=config["split_role_profiles"]["valid_ratio"],
        test_ratio=config["split_role_profiles"]["test_ratio"],
        seed=config["seed"],
    )
    FileWriter.dump(train, os.path.join(config["output_dir"], "train.jsonl"))
    FileWriter.dump(valid, os.path.join(config["output_dir"], "valid.jsonl"))
    FileWriter.dump(test, os.path.join(config["output_dir"], "test.jsonl"))


def main():
    config = load_config("./configs/step0_create_roles.yaml")

    # generate role config file
    role_config_path = os.path.join(config["output_dir"], "role_configs.jsonl")
    if not os.path.exists(role_config_path):
        generate_role_configs(config, role_config_path)

    # generate summary and biography
    role_profiles_path = os.path.join(config["output_dir"], "role_profiles.jsonl")
    if not os.path.exists(role_profiles_path):
        generate_role_profile(config, role_config_path, role_profiles_path)
    split_roles(config, role_profiles_path)


if __name__ == "__main__":
    main()
