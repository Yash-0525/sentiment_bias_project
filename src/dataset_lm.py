"""
src/dataset_lm.py  —  load Phase 2 packed sequences for LM training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class PackedSequenceDataset(Dataset):
    """
    Reads sequences_*.jsonl written by Phase 2:
      {"input_ids": [int, ...], "has_sensitive": bool, ...}
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        sensitive_only: bool = False,
        max_sequences: int | None = None,
    ):
        self.path = Path(jsonl_path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"missing {self.path}. Run Phase 2 first "
                f"(python -m src.data_preprocessing)."
            )
        self.rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if sensitive_only and not row.get("has_sensitive", False):
                    continue
                self.rows.append(row)
                if max_sequences is not None and len(self.rows) >= max_sequences:
                    break
        if not self.rows:
            raise RuntimeError(f"no sequences loaded from {self.path} "
                               f"(sensitive_only={sensitive_only})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | bool]:
        row = self.rows[idx]
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),  # HF shifts internally when labels=input_ids
            "has_sensitive": bool(row.get("has_sensitive", False)),
        }


def collate_lm(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    # all sequences are fixed length from Phase 2 packing → full attention
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def make_loader(
    jsonl_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    sensitive_only: bool = False,
    max_sequences: int | None = None,
) -> DataLoader:
    ds = PackedSequenceDataset(
        jsonl_path,
        sensitive_only=sensitive_only,
        max_sequences=max_sequences,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_lm,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # drop last only when training
    )
