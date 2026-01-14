# gen_dataset/dataset_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DatasetRow:
    """
    One row in our (eventually parquet) dataset.
    """
    img_path: str   # for easier debugging
    img_bytes: bytes # for training

    family: str  # "perception" | "strategy"
    q_id: str  # "Q1", "Q2", ...
    focus: str  # "count_black_stones", ...z

    answer: str  # canonical answer used for training
    valid_answers: List[str] # all acceptable answers (incl. canonical)

    question: Optional[str] = None  # the natural language question (filled later after helper function executed)
    split: Optional[str] = None # "train" | "eval" | "test" (filled later when creating parquet file)