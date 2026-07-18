#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContrastivePlugin: 对比学习插件
"""

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lwj_tools.io.reader import FileReader
from lwj_tools.io.writer import FileWriter

from .base import Plugin, Scope, PluginConfig


class ProjectionMLP(nn.Module):
    """将 hidden state 映射到低维对比学习空间"""

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, embedding_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.net(x), dim=-1)
        return x


class ContrastiveMemoryBank:
    """存储 teacher hidden states 的记忆库，按 (event_idx, role_idx, relation) 索引
    通用数据不存入。
    """

    def __init__(self, memory_bank_size: int = 4096):
        self.memory_bank_size = memory_bank_size
        # key: (event_idx, role_idx, relation), value: deque of teacher embeddings
        self.bank: Dict[Tuple[int, int, str], deque] = {}
        self._access_order: List[Tuple[int, int, str]] = []
        # Track role_idx for each key so we can compute hamming distance
        self._bank_role_keys: Dict[Tuple[int, int, str], int] = {}

    def _make_key(
        self, event_idx: int, role_idx: int, relation: str
    ) -> Tuple[int, int, str]:
        return event_idx, role_idx, relation

    def push(
        self,
        event_idx: int,
        role_idx: int,
        relation: str,
        teacher_embedding: torch.Tensor,
    ):
        """存入一组 teacher embedding"""
        key = self._make_key(event_idx, role_idx, relation)
        if key not in self.bank:
            self.bank[key] = deque(maxlen=1)
            self._access_order.append(key)

        self.bank[key].append(teacher_embedding)
        self._bank_role_keys[key] = role_idx

        if len(self._access_order) > self.memory_bank_size:
            old_key = self._access_order.pop(0)
            del self.bank[old_key]
            del self._bank_role_keys[old_key]

    def get_positive(
        self, event_idx: int, role_idx: int, relation: str
    ) -> Optional[torch.Tensor]:
        """获取同一 (event, role, relation) 的 teacher embedding"""
        key = self._make_key(event_idx, role_idx, relation)
        if key not in self.bank or len(self.bank[key]) == 0:
            return None
        return self.bank[key][-1]

    def get_negatives(
        self, event_idx: int, role_idx: int, relation: str, max_negatives: int = 15
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """获取负样本（返回 embeddings 和对应的 role_ids）

        Returns:
            (neg_embeddings, neg_role_ids): 所有负样本的 embeddings 和 role_ids
        """
        negatives = []

        for key, deque_ in self.bank.items():
            e_idx, r_idx, rel = key
            if len(deque_) == 0:
                continue

            # Skip positive
            if e_idx == event_idx and r_idx == role_idx and rel == relation:
                continue

            if not (e_idx == event_idx and rel == relation):
                continue

            embedding = deque_[-1]
            neg_role_idx = self._bank_role_keys.get(key, r_idx)
            negatives.append((embedding, neg_role_idx))

        # 限制数量
        if len(negatives) > max_negatives:
            indices = torch.randperm(len(negatives))[:max_negatives]
            negatives = [negatives[i] for i in indices]

        neg_embs = [n[0] for n in negatives]
        neg_role_ids = [n[1] for n in negatives]

        return neg_embs, neg_role_ids

    def get_pairs(
        self, event_idx: int, role_idx: int, relation: str, max_negatives: int = 15
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor], List[int]]:
        """同时获取正样本和所有负样本"""
        positive = self.get_positive(event_idx, role_idx, relation)
        neg_embs, neg_role_ids = self.get_negatives(
            event_idx, role_idx, relation, max_negatives
        )
        return positive, neg_embs, neg_role_ids

    def clear(self):
        self.bank.clear()
        self._access_order.clear()
        self._bank_role_keys.clear()

    def state_dict(self):
        return {
            "bank": {k: list(v) for k, v in self.bank.items()},
            "_access_order": self._access_order,
            "_bank_role_keys": self._bank_role_keys,
        }

    def load_state_dict(self, state_dict: dict):
        self._access_order = state_dict["_access_order"]
        self._bank_role_keys = state_dict["_bank_role_keys"]
        self.bank = {k: deque(v) for k, v in state_dict.items()}


class PersonaContrastiveLoss(nn.Module):
    """个性化对比损失: student 学习 teacher 在同事件同维度不同角色下的 hidden state 差异

    Anchor: S(E_i, R_j, D_k) -- Student 对事件 i、角色 j、维度 k 的 hidden state
    Positive: T(E_i, R_j, D_k) -- Teacher 对事件 i、角色 j、维度 k 的 hidden state
    Negative: T(E_i, R_m, D_k), m != j -- 同事件同维度不同角色

    权重计算: 基于 Hamming 距离（role profile 五元组的差异程度）
    Profile 越相似 -> Hamming 距离越小 -> 权重越高
    """

    def __init__(
        self,
        temperature: float = 0.07,
        max_negatives: int = 15,
        role_profile_map: Optional[Dict[int, Tuple]] = None,
        hamming_weight_scale: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.max_negatives = max_negatives
        self.role_profile_map = role_profile_map or {}
        self.hamming_weight_scale = hamming_weight_scale

    def _get_negative_score_weights(
        self,
        student_emb: torch.Tensor,
        neg_samples: List[torch.Tensor],
        neg_role_ids: List[int],
        anchor_role_id: int,
        role_profile_map: Dict[int, Tuple],
        base_weight: float = 1.0,
        hamming_weight_scale: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            student_emb: [embedding_dim, ]
            neg_samples: [(emb_dim, ), ...]
            neg_role_ids: role_idx for each negative sample
            anchor_role_id: role_idx of the anchor (positive sample)
            role_profile_map: role_idx -> (archetype, culture, age, gender, job_category)
            hamming_weight_scale: controls how much hamming distance affects weight (0~1)
        """
        dtype = student_emb.dtype
        device = student_emb.device

        anchor_profile = role_profile_map.get(anchor_role_id, None)
        if anchor_profile is None:
            weights = torch.tensor(
                [base_weight] * len(neg_role_ids), dtype=dtype, device=device
            )
            neg_embs = torch.stack([n.to(device) for n in neg_samples])
            neg_sim = torch.matmul(student_emb.unsqueeze(0), neg_embs.T).squeeze(0)
            return neg_sim, weights

        # Compute hamming distance for each negative
        hamming_weights = []
        for nid in neg_role_ids:
            neg_profile = role_profile_map.get(nid, None)
            if neg_profile is None:
                hamming_weights.append(hamming_weight_scale)
            else:
                hamming = sum(a != b for a, b in zip(anchor_profile, neg_profile))
                # Normalize by number of attributes (5)
                hamming_norm = hamming / len(anchor_profile)
                weight = base_weight * (1.0 - hamming_norm * hamming_weight_scale)
                hamming_weights.append(weight)

        # [num_neg, embedding_dim]
        neg_embs = torch.stack([n.to(device) for n in neg_samples])
        # [num_neg]
        neg_sim = torch.matmul(student_emb.unsqueeze(0), neg_embs.T).squeeze(0)
        weights = torch.tensor(hamming_weights, dtype=dtype, device=device)

        return neg_sim, weights

    def forward(
        self,
        student_hidden: torch.Tensor,
        event_ids: List[int],
        role_ids: List[int],
        relations: List[str],
        memory_bank: ContrastiveMemoryBank,
        teacher_proj: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            student_hidden: [n, hidden_dim] - proj 后的 dim
            event_ids: [n,]
            role_ids: [n,]
            relations: [n,] - 推理维度 (ATOMIC relation)
            memory_bank: ContrastiveMemoryBank instance
            teacher_proj:
        Returns:
            scalar contrastive loss
        """
        batch_size = student_hidden.shape[0]
        device = student_hidden.device
        dtype = student_hidden.dtype

        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        for i in range(batch_size):
            e_idx = event_ids[i]
            r_idx = role_ids[i]
            rel = relations[i]

            positive, neg_embs, neg_role_ids = memory_bank.get_pairs(
                e_idx, r_idx, rel, self.max_negatives
            )

            if positive is None or len(neg_embs) == 0:
                continue

            positive = teacher_proj(positive.to(device).unsqueeze(0)).squeeze(0)
            neg_embs = [
                teacher_proj(neg_emb.to(device).unsqueeze(0)).squeeze(0)
                for neg_emb in neg_embs
            ]

            # [1]
            pos_sim = torch.dot(student_hidden[i], positive).unsqueeze(0)

            neg_sim, neg_weights = self._get_negative_score_weights(
                student_hidden[i],
                neg_embs,
                neg_role_ids,
                r_idx,
                self.role_profile_map,
                base_weight=1.0,
                hamming_weight_scale=self.hamming_weight_scale,
            )

            # [1 + num_neg, ]
            logits = torch.cat([pos_sim, neg_sim]) / self.temperature
            weights = torch.cat(
                [torch.tensor([1.0], device=device, dtype=dtype), neg_weights]
            )

            exp_logits = logits.exp() * weights
            loss = -torch.log(exp_logits[0] / exp_logits.sum())

            total_loss = total_loss + loss
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / valid_count


def _get_last_valid_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    last_positions = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_positions]


def _load_role_profiles(file_path: str) -> Dict[int, Tuple]:
    """加载 role profile，建立 role_idx -> 五元组映射"""
    profile_map = {}
    if os.path.exists(file_path):
        for d in FileReader.read(file_path, return_iter=True):
            idx = d["index"]
            profile = (
                d["archetype"],  # 1
                d["culture"],  # 2
                d["age"],  # 3
                d["gender"],  # 4
                d["job_category"],  # 5
            )
            profile_map[idx] = profile
    return profile_map


@dataclass
class ContrastivePluginConfig(PluginConfig):
    embedding_dim: int = 128
    temperature: float = 0.1
    memory_bank_size: int = 4096
    loss_weight: float = 1.0
    max_negatives: int = 10
    role_file_path: Optional[str] = None
    hamming_weight_scale: float = 1.0
    role_profile_path: str = None
    student_hidden_size: int = None
    teacher_hidden_size: int = None


class ContrastivePlugin(Plugin):
    """对比学习插件"""

    SCOPE = Scope.post_loss

    def __init__(self, config: ContrastivePluginConfig):
        super().__init__(config)

        student_hs = config.student_hidden_size
        teacher_hs = config.teacher_hidden_size
        self.student_proj = ProjectionMLP(student_hs, config["embedding_dim"])
        self.teacher_proj = ProjectionMLP(teacher_hs, config["embedding_dim"])
        self.memory_bank = ContrastiveMemoryBank(config["memory_bank_size"])

        # Load role profile map from role_configs.jsonl
        role_profile_map = _load_role_profiles(config["role_profile_path"])

        self.contrastive_loss_fn = PersonaContrastiveLoss(
            temperature=config["temperature"],
            max_negatives=config["max_negatives"],
            role_profile_map=role_profile_map,
            hamming_weight_scale=config["hamming_weight_scale"],
        )

    def train(self, mode: bool = True):
        self.config.enabled = mode
        return self

    def inject(
        self,
        samples: List[dict],
        stu_hidden: torch.Tensor,
        tea_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算对比学习损失

        Args:
            samples (list):  dict, 取决于 collate_fn
            stu_hidden (torch.Tensor): shape = [bsz, seq_len, stu_hs]
            tea_hidden (torch.Tensor): [bsz, seq_len, tea_hs]
            attention_mask (torch.Tensor): 用于获取最后一个 token
        """

        if not self.enabled or self.config.loss_weight <= 0:
            return torch.tensor(0.0, device=stu_hidden.device)

        # [bsz, stu_hs]
        stu_last = _get_last_valid_token(stu_hidden, attention_mask)
        tea_last = _get_last_valid_token(tea_hidden, attention_mask).detach().cpu()
        stu_proj = self.student_proj(stu_last)  # [bsz, dim]

        event_ids, role_ids, relations = [], [], []
        for i, sample in enumerate(samples):
            e_idx = sample["event_idx"]
            r_idx = sample["role_idx"]
            rel = sample["relation"]
            self.memory_bank.push(e_idx, r_idx, rel, tea_last[i])
            event_ids.append(e_idx)
            role_ids.append(r_idx)
            relations.append(rel)

        return self.contrastive_loss_fn(
            student_hidden=stu_proj,
            event_ids=event_ids,
            role_ids=role_ids,
            relations=relations,
            memory_bank=self.memory_bank,
            teacher_proj=self.teacher_proj,
        )

    def save_pretrained(self, save_dir: str):
        if not self.enabled:
            return

        cls_name = type(self).__name__
        save_dir = os.path.join(save_dir, cls_name)
        FileWriter.dump(
            self.config.to_dict(),
            os.path.join(save_dir, "config.json"),
            indent=4,
        )
        torch.save(
            self.student_proj.state_dict(), os.path.join(save_dir, "student_proj.pt")
        )
        torch.save(
            self.teacher_proj.state_dict(), os.path.join(save_dir, "teacher_proj.pt")
        )
        torch.save(
            self.memory_bank.state_dict(), os.path.join(save_dir, "memory_bank.pt")
        )

    @staticmethod
    def from_pretrained(ckpt_dir: str, device="cpu", *args, **kwargs):
        ckpt_dir = os.path.join(ckpt_dir, ContrastivePlugin.__name__)
        config_path = os.path.join(ckpt_dir, "config.json")

        plugin_config: dict = FileReader.read(config_path)

        plugin = ContrastivePlugin(ContrastivePluginConfig(**plugin_config["config"]))

        plugin.student_proj.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "student_proj.pt"), map_location=device)
        )
        plugin.teacher_proj.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "teacher_proj.pt"), map_location=device)
        )
        plugin.memory_bank.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "memory_bank.pt"), map_location=device)
        )
        return plugin
