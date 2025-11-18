# gen_dataset/dataset_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DatasetRow:
    """
    One row in our (eventually parquet) dataset.
    """
    family: str  # "perception" | "strategy"
    q_id: str  # "Q1", "Q2", ...
    focus: str  # "count_black_stones", ...

    img_path: str   # for easier debugging
    img_bytes: bytes # for training

    answer: str  # canonical answer used for training
    valid_answers: Optional[List[str]] = None  # all acceptable answers (incl. canonical)

    question: Optional[str] = None  # the natural language question (filled later after helper function executed)
    split: Optional[str] = None # "train" | "eval" | "test" (filled later when creating parquet file)