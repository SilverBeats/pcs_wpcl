#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import re
from typing import List, Any, Optional, Dict, Tuple

import numpy as np
from lwj_tools.io.reader import FileReader

from .constant import RELATION_TO_QUESTION, SYSTEM_PROMPT, INSTRUCTION_PROMPT
from .pojo import (
    EventSample,
    RoleSample,
    RoleProfile,
    DebateSample,
    DatasetAblationArguments,
    TrainSample,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def _normalize_ratio(a, b, c):
    _s = a + b + c
    return a / _s, b / _s, c / _s


def split_datasets(
    samples: List[Any],
    train_ratio: float = 0,
    valid_ratio: float = 0,
    test_ratio: float = 0,
    seed: int = 42,
):
    if not (train_ratio or valid_ratio or test_ratio):
        raise ValueError("At least one ratio must be non-zero")

    if len(samples) == 0:
        return [], [], []

    train_ratio, valid_ratio, test_ratio = _normalize_ratio(
        train_ratio, valid_ratio, test_ratio
    )

    rng = random.Random(seed)
    np.random.seed(seed)

    # Shuffle with indices
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_indices = set(indices[:n_train])
    valid_indices = set(indices[n_train : n_train + n_valid])
    test_indices = set(indices[n_train + n_valid :])

    train_samples = [samples[i] for i in train_indices]
    valid_samples = [samples[i] for i in valid_indices]
    test_samples = [samples[i] for i in test_indices]

    return train_samples, valid_samples, test_samples


def safe_path_tag(text: str) -> str:
    """Convert string to stable path segment."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text)).strip("-")


def build_lang_limit_tag(limit_by_lang: Optional[Dict[str, int]] = None) -> str:
    if not limit_by_lang:
        return "all"
    parts = []
    for lang in sorted(limit_by_lang.keys()):
        parts.append(f"{lang}_{limit_by_lang.get(lang, 'all')}")
    return "_".join(parts) if parts else "all"


def get_controversy_output_dir(controversy_config: dict) -> str:
    models = controversy_config["calc_controversy"]["models"]
    model_tag = safe_path_tag(models[0])

    dataset = controversy_config["dataset"]
    limit_tag = build_lang_limit_tag(
        controversy_config["datasets"][dataset].get("sample_limit_by_lang", None)
    )
    controversy_output_dir = os.path.join(
        controversy_config["output_dir"],
        f"dataset_{dataset}",
        "calc_event_controversy",
        model_tag,
        f"limit_{limit_tag}",
    )
    return controversy_output_dir


def convert_event_to_pojo(event_sample: dict) -> EventSample:
    return EventSample(
        index=event_sample["index"],
        lang=event_sample["lang"],
        event=event_sample["event"],
        category=event_sample.get("category", ""),
        label=event_sample.get("label", None),
    )


def convert_role_to_pojo(role_sample: dict) -> RoleSample:
    return RoleSample(
        index=role_sample["index"],
        # zh=RoleProfile(**role_sample.get("zh", {})),
        en=RoleProfile(**role_sample.get("en", {})),
    )


def convert_debate_sample_to_pojo(debate_sample: dict) -> DebateSample:
    return DebateSample(
        index=debate_sample["index"],
        event=convert_event_to_pojo(debate_sample["event"]),
        role=convert_role_to_pojo(debate_sample["role"]),
    )


def build_index_to_event(event_file_path: str) -> Dict[int, EventSample]:
    """Load events and build index -> event mapping."""
    out = {}
    for sample in FileReader.read(event_file_path, return_iter=True):
        event_sample = convert_event_to_pojo(sample)
        out[event_sample.index] = event_sample
    return out


def build_index_to_role(role_file_path: str) -> Dict[int, RoleSample]:
    """Load index -> role mapping."""
    out = {}
    for sample in FileReader.read(role_file_path, return_iter=True):
        role_sample = convert_role_to_pojo(sample)
        out[role_sample.index] = role_sample
    return out


# =============================================================================
# Step 6: Alpaca Dataset Building
# =============================================================================


def _format_instruction(task_type: str, task_input: str) -> str:
    """Format instruction prompt for alpaca format."""
    return INSTRUCTION_PROMPT.format(
        TaskType=task_type.strip(), TaskInput=task_input.strip()
    ).strip()


def _build_persona(
    role_profile: RoleProfile, ablation_config: DatasetAblationArguments
) -> str:
    """Build persona string from role profile, respecting ablation flags."""
    arch_traits = role_profile.arch_traits or ""
    age = role_profile.age or ""
    gender = role_profile.gender or ""
    job_category = role_profile.job_category or ""
    cultural_cluster = role_profile.culture or ""
    arr = ["\n"]

    if not ablation_config.wo_ocean:
        arr.append(f"- Personality(OCEAN): {arch_traits}")

    if not ablation_config.wo_age:
        arr.append(f"- Age: {age}")

    if not ablation_config.wo_gender:
        arr.append(f"- Gender: {gender}")

    if not ablation_config.wo_job_category:
        arr.append(f"- Job Category: {job_category}")

    if not ablation_config.wo_culture:
        arr.append(f"- Cultural Background: {cultural_cluster}")

    persona = "\n".join(arr).strip()
    return persona


def _extract_general_triplet(sample: dict) -> Optional[Tuple[str, str, str]]:
    """Extract (instruction, input, output) from general sample."""
    instruction = sample.get("instruction")
    input_text = sample.get("input")
    output = sample.get("output")

    if instruction is None or output is None:
        return None
    return str(instruction), str(input_text or ""), str(output)


def build_general_sample_to_pojo(sample: dict) -> Optional[TrainSample]:
    """Convert a general (non-personalized) sample to TrainSample."""
    extracted = _extract_general_triplet(sample)
    if extracted is None:
        return None
    instruction, input_text, output = extracted
    task_input = f"{instruction}\n{input_text}".strip()

    persona_info = _build_persona(
        role_profile=RoleProfile(
            name="N/A",
            age="N/A",
            gender="N/A",
            archetype="N/A",
            arch_traits="N/A",
            arch_desc="N/A",
            job_category="N/A",
            job_desc="N/A",
            culture="N/A",
            culture_countries="N/A",
            summary="N/A",
            biography="N/A",
        ),
        ablation_config=DatasetAblationArguments(),
    )
    return TrainSample(
        system=SYSTEM_PROMPT.format(Persona=persona_info),
        instruction=_format_instruction(
            task_type="General Task",
            task_input=task_input,
        ),
        output=f"<think>\nI will answer the user's question directly and concisely.\n</think>\n{output}",
    )


def convert_personalization_sample_to_pojo(
    infer: str,
    cot: str,
    relation: str,
    role_profile: RoleProfile,
    event_sample: EventSample,
    ablation_config: DatasetAblationArguments,
) -> TrainSample:
    """Convert a personalization sample (event + role + debate result) to TrainSample."""
    persona = _build_persona(role_profile, ablation_config)

    task_input = (
        f"Event: {event_sample.event}\nQuestion: {RELATION_TO_QUESTION[relation]}"
    )
    input_text = _format_instruction(
        task_type="Personalization Task",
        task_input=task_input,
    )
    if ablation_config.wo_cot:
        output = f"<think>\n\n</think>\n{infer}"
    else:
        output = f"<think>\n{cot}\n</think>\n{infer}"

    return TrainSample(
        system=SYSTEM_PROMPT.format(Persona=persona),
        instruction=input_text,
        output=output,
        event_idx=event_sample.index,
        relation=relation,
    )
