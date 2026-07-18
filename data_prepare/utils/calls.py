#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM call wrappers for data preparation pipeline.
"""

import abc
import json
from functools import partial
from io import TextIOWrapper
from typing import Any, List, Optional, ClassVar, Type

import requests
from lwj_tools.io.reader import FileReader
from lwj_tools.llms.chain import LLMChain
from lwj_tools.llms.client import LLMClientGroup, APIConfig
from lwj_tools.llms.prompt import PromptTemplate
from lwj_tools.utils.common import get_unprocessed_samples
from lwj_tools.utils.concurrent import MultiThreadingRunner

from .pojo import EventSample, DebateSample
from .prompts import (
    GenerateRoleProfilePromptTemplate,
    FilterEventPromptTemplate,
    DebatePromptTemplate,
)
from .tools import convert_debate_sample_to_pojo

DEFAULT_GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 1.0,
    "seed": 42,
    "extra_body": {"think": False},
    "response_format": {"type": "json_object"},
}


def _write_jsonl_record(
    idx: int, result: Any, fp: Optional[TextIOWrapper] = None
) -> Any:
    if fp:
        fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        fp.flush()
    return result


# =============================================================================
# LLM Call Classes
# =============================================================================


class PromptConfiguredLLMCall(abc.ABC):
    """Base class for LLM batch calls."""

    _prompt_template_cls: ClassVar[Type[PromptTemplate]]

    def __init__(
        self,
        api_bases: List[str],
        model_names: Optional[List[str]] = None,
        api_keys: Optional[List[str]] = None,
        generation_config: Optional[dict] = None,
        max_worker: int = -1,
    ):
        if api_keys is None:
            api_keys = ["xx"] * len(api_bases)
        if model_names is None:
            model_names = ["xx"] * len(api_bases)

        if generation_config is None:
            generation_config = DEFAULT_GENERATION_CONFIG

        self.chain = LLMChain(
            client_group=LLMClientGroup(
                api_configs=[
                    APIConfig(
                        api_base=api_base,
                        model=model_name,
                        api_key=api_key,
                    )
                    for api_base, model_name, api_key in zip(
                        api_bases, model_names, api_keys
                    )
                ]
            ),
            prompt_template=self._prompt_template_cls(),
            **generation_config,
        )
        self.runner = MultiThreadingRunner(max_worker)

    def _worker_func(self, idx: int, sample: Any):
        raise NotImplementedError

    def __call__(
        self,
        file_path: Optional[str] = None,
        output_path: Optional[str] = None,
        id_field: str = "index",
        desc: str = "Running",
        samples: Optional[List[Any]] = None,
        existed_samples: Optional[List[Any]] = None,
        **kwargs,
    ):
        unprocessed_samples = get_unprocessed_samples(
            data_file_path=file_path,
            samples=samples,
            output_file_path=output_path,
            existed_samples=existed_samples,
            id_field=id_field,
            return_iter=False,
        )
        if len(list(unprocessed_samples)) == 0:
            print("No sample need to process.")
            return

        fp = None
        if output_path is not None:
            fp = open(output_path, "a+", encoding="utf-8")

        self.runner(
            samples=unprocessed_samples,
            worker_func=self._worker_func,
            finished_func=partial(_write_jsonl_record, fp=fp),
            pbar_desc=desc,
        )
        if fp:
            fp.close()


class GenerateRolesLLMCall(PromptConfiguredLLMCall):
    _prompt_template_cls = GenerateRoleProfilePromptTemplate

    def _worker_func(self, idx: int, sample: dict):
        try:
            resp = self.chain(sample)
            if resp.error:
                raise resp.error
            result = resp.result
            return {**sample, **result}
        except Exception as e:
            print(e)


class RolePostEventViewLLMCall(PromptConfiguredLLMCall):
    """Step 1: Event controversy judgment."""

    prompt_template_cls = FilterEventPromptTemplate

    def _worker_func(self, idx: int, sample: EventSample):
        try:
            resp = self.chain(sample)
            if resp.error:
                raise resp.error
            result = resp.result
            assert isinstance(result, dict), f"result is not a dict. Result={result}"
            assert (
                "perspectives" in result
            ), f"perspectives not found in response. Result={result}"
            assert (
                len(result["perspectives"]) == 4
            ), f"length of perspectives is not 4. Perspectives={result['perspectives']}"
            assert all(
                isinstance(perspective["viewpoint_summary"], str)
                for perspective in result["perspectives"]
            ), f"viewpoint_summary is not a string. Perspectives={result['perspectives']}"
            return {**sample.to_dict(), **result}
        except Exception as e:
            print(f"[ERROR] idx={idx}: {e}")


class DebateGenerateSampleLLMCall(PromptConfiguredLLMCall):
    system_prompt = """
You are a Professional Psychological Auditor. Your mission is to generate **deeply biased, persona-specific** ATOMIC reasoning data. 

### Audit Logic (Chairman)
For each dimension, simulate a raw reaction then audit it:
- **KEEP**: The raw reaction is perfectly biased and persona-aligned.
- **ADJUST**: The raw reaction is too "AI-like" or generic. Refine it to be more "humanly flawed" and persona-specific.
- **DISCARD**: Dimension is irrelevant to the event.

### Core Reasoning Logic (The 3-Step Rationale)
Every 'rationale' field MUST follow this internal chain:
1. **Trait Mapping**: How do the OCEAN traits (e.g., High N, Low A) trigger an immediate emotional undercurrent or instinct in this event?
2. **Contextual Filter**: How does the persona's Culture and Occupation modify or suppress that instinct? (e.g., A Self-Centered person in a 'Social' job might hide their greed behind professional politeness).
3. **Action Conversion**: How does this internal conflict result in the **specific** ATOMIC inference?

### Anti-Mediocrity Rules
- **BAN VAGUE VIRTUES**: Forbidden to use abstract terms like "to help others", "to be a role model", "to work hard", or "to be kind" unless you provide specific, concrete details. 
- **SHARP BIAS**: If the persona is Self-Centered, show their greed/vanity. If they are Reserved, show their fear of change. 
- **SPECIFICITY**: Inferences must be concrete. Instead of "to study", use "to memorize the exam syllabus until dawn".

### Style & Structure
- **Format**: First-person ("I"), infinitive phrase ("to..."), max 15 words for final inferences.
- **Tone**: Strictly aligned with the biography. No AI neutrality.

### ATOMIC Dimensions
xAttr (Self-description), xWant (Next action), xNeed (Pre-requisite), xIntent (Motive), xEffect (Self-impact), xReact (Self-feeling), oEffect (Impact on others), oWant (Others' reaction), oReact (Others' feeling).

Output strictly JSON. No filler.
"""

    _prompt_template_cls = DebatePromptTemplate

    def _worker_func(self, idx: int, sample: DebateSample):
        try:
            resp = self.chain(sample, system_prompt=self.system_prompt.strip())
            if resp.error:
                raise resp.error
            result = resp.result
            return {"index": sample.index, "result": result}
        except Exception as e:
            print(e)

    def __call__(
        self,
        file_path: Optional[str] = None,
        output_path: Optional[str] = None,
        id_field: str = "index",
        desc: str = "Generating Debate Samples",
        samples: Optional[List[DebateSample]] = None,
        existed_samples: Optional[List[dict]] = None,
        **kwargs,
    ):
        if not (samples or existed_samples):
            raise TypeError(
                "DebateGenerateSample requires file_path, data_file_path, or samples"
            )

        if samples is None and file_path:
            samples = [
                convert_debate_sample_to_pojo(sample)
                for sample in FileReader.read(file_path, return_iter=True)
            ]

        super().__call__(
            file_path=None,
            output_path=output_path,
            id_field=id_field,
            desc=desc,
            samples=samples,
            existed_samples=existed_samples,
            **kwargs,
        )


# =============================================================================
# Simple HTTP-based calls (without lwj_tools dependency)
# =============================================================================
class InferSimilarityCall:
    """Step2: Calculate similarity between inference results."""

    def __init__(self, base_url: str, max_workers: int = -1):
        self.base_url = base_url
        self.runner = MultiThreadingRunner(max_workers)

    def _worker_func(self, idx: int, sample: dict):
        try:
            views = [item["viewpoint_summary"] for item in sample["perspectives"]]
            lang = sample["lang"]
            resp = requests.post(
                url=self.base_url,
                json={
                    "sentences": views,
                    "lang": lang,
                },
            )
            sim_scores = resp.json()
            return {"index": sample["index"], "sim_matrix": sim_scores}
        except Exception as e:
            print(e)

    def __call__(
        self,
        file_path: str,
        output_path: str,
        id_field: str = "index",
        desc: str = "Infer Similarity",
    ):
        samples = get_unprocessed_samples(file_path, output_path, id_field=id_field)
        if len(samples) == 0:
            return
        with open(output_path, "a+", encoding="utf-8") as fp:
            self.runner(
                samples=samples,
                worker_func=self._worker_func,
                finished_func=partial(_write_jsonl_record, fp=fp),
                pbar_desc=desc,
            )
