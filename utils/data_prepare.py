#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial
from typing import List, Tuple

import torch
from lwj_tools.io.reader import FileReader
from lwj_tools.utils.concurrent import MultiThreadingRunner
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Qwen2Tokenizer

from data_prepare.utils.pojo import TrainSample


def is_empty(s: str) -> bool:
    if not s:
        return True

    i = s.find("<think>")
    j = s.find("</think>")
    if i == -1 or j == -1:
        return True

    s = s[j + len("</think>") :].strip()
    return len(s) == 0


def post_process(s: str) -> str:
    if is_empty(s):
        return ""
    return s.strip()
    tokens = ["</think>", "[Infer]"]
    for token in tokens:
        i = s.find(token)
        if i != -1:
            s = s[i + len(token) :].strip()
    return s.strip()


def _worker_func(idx, sample: dict, max_length: int = 512, tokenizer=None):
    for k in ["output"]:
        if is_empty(sample[k]):
            return None
    if tokenizer is None:
        return TrainSample(**sample)
    think_start_token_id = tokenizer.encode("<think>")[0]
    think_end_token_id = tokenizer.encode("</think>")[0]
    try:
        conversations = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_assistant_tokens_mask=False,
            return_dict=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"].flatten()
        assert input_ids.shape[0] > 0
        assert think_start_token_id in input_ids
        assert think_end_token_id in input_ids
        assert input_ids[-1].item() != think_end_token_id
        return TrainSample(**sample)
    except Exception as e:
        print(e)
        return None


def _finished_func(idx, result: TrainSample | None, container: List):
    if result is None:
        return
    container.append(result)


def load_alpaca_dataset(
    file_path: str, tokenizer, max_length: int = 512, rank: int = -1
) -> Tuple[List[TrainSample], int]:
    samples: List[TrainSample] = []
    raw_samples = FileReader.read(file_path)
    runner = MultiThreadingRunner(num_workers=-1, use_pbar=rank in [-1, 0])
    runner(
        samples=raw_samples,
        pbar_desc="Processing Data",
        finished_func=partial(_finished_func, container=samples),
        worker_func=partial(_worker_func, tokenizer=tokenizer, max_length=max_length),
        n_samples=len(raw_samples),
    )
    skip_count = len(raw_samples) - len(samples)
    return samples, skip_count


def _prepare_think_infer_mask(
    think_start_token_id: int, think_end_token_id: int, input_ids: torch.LongTensor
):
    think_start_index = (input_ids == think_start_token_id).nonzero().flatten()[0]
    think_end_index = (input_ids == think_end_token_id).nonzero().flatten()[0]

    think_mask = torch.zeros_like(input_ids)
    think_mask[think_start_index : think_end_index + 1] = 1.0

    infer_mask = torch.zeros_like(input_ids)
    infer_mask[think_end_index + 1 :] = 1.0

    return think_mask, infer_mask


def prepare_inputs(
    batch_sample: List[TrainSample], tokenizer: Qwen2Tokenizer, max_length: int = 512
) -> Tuple[torch.Tensor, ...]:
    batch_input_ids, batch_label_ids = [], []
    batch_attention_mask, batch_kd_mask = [], []
    batch_think_mask, batch_infer_mask = [], []

    think_start_token_id = tokenizer.encode("<think>")[0]
    think_end_token_id = tokenizer.encode("</think>")[0]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    task_type_ids = []
    for sample in batch_sample:
        if "Personalization" in sample.instruction:
            task_type_ids.append(1)
        else:
            task_type_ids.append(0)

        conversations = [
            {"role": "system", "content": sample.system},
            {"role": "user", "content": sample.instruction},
            {"role": "assistant", "content": sample.output},
        ]
        output = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_assistant_tokens_mask=False,
            return_dict=True,
            truncation=True,
            max_length=max_length,
        )
        try:
            input_ids = output["input_ids"][0]
            batch_input_ids.append(input_ids)
            attention_mask = output["attention_mask"][0]
            batch_attention_mask.append(attention_mask)

            label_ids = input_ids.clone()
            think_start_index = (
                (input_ids == think_start_token_id).nonzero().flatten()[0]
            )
            label_ids[:think_start_index] = -100
            batch_label_ids.append(label_ids)

            # KD mask：output 部分为 1
            kd_mask = (label_ids != -100).float()
            batch_kd_mask.append(kd_mask)

            think_mask, infer_mask = _prepare_think_infer_mask(
                think_start_token_id, think_end_token_id, input_ids
            )
            batch_think_mask.append(think_mask)
            batch_infer_mask.append(infer_mask)
        except Exception as e:
            print(sample)
            print(input_ids)
            print((input_ids == think_start_token_id).nonzero())
            raise e

    # Padding
    input_ids = pad_sequence(
        batch_input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = pad_sequence(
        batch_attention_mask, batch_first=True, padding_value=0.0
    )
    label_ids = pad_sequence(batch_label_ids, batch_first=True, padding_value=-100)
    kd_mask = pad_sequence(batch_kd_mask, batch_first=True, padding_value=0.0)
    think_mask = pad_sequence(batch_think_mask, batch_first=True, padding_value=0.0)
    infer_mask = pad_sequence(batch_infer_mask, batch_first=True, padding_value=0.0)
    task_type_ids = torch.tensor(task_type_ids)
    return (
        input_ids,
        attention_mask,
        label_ids,
        kd_mask,
        think_mask,
        infer_mask,
        task_type_ids,
    )


class PCSDataset(Dataset):
    """Personalization + Cognitive Service Dataset"""

    def __init__(self, samples: List[TrainSample]):
        self.samples = samples
        self.size = len(samples)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TrainSample:
        return self.samples[index]

    @staticmethod
    def collate_fn(
        batch: List[TrainSample], tokenizer: Qwen2Tokenizer, max_length: int
    ) -> dict:
        (
            input_ids,
            attention_mask,
            label_ids,
            kd_mask,
            think_mask,
            infer_mask,
            task_type_ids,
        ) = prepare_inputs(batch, tokenizer, max_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "kd_mask": kd_mask,
            "infer_mask": infer_mask,
            "think_mask": think_mask,
            "samples": batch,
            "task_type_ids": task_type_ids,
        }
