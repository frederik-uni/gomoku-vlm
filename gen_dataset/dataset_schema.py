# gen_dataset/dataset_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetRow:
    """
    One row in our (eventually parquet) dataset.

    - family:       "perception" or "strategy"
    - q_id:         "Q1", "Q2", "Q3", ...
    - focus:        e.g. "horizontal_line_detection", "best_move"
    - answer:       the expected model output
    - img_path:     where the associated board image is stored
    - img_bytes:    the byte representation of the image
    - question:     the natural language question (filled later after helper function executed)
    - split:        "train" | "eval" | "test" (filled later when creating parquet file)
    """
    family: str
    q_id: str
    focus: str
    answer: str
    img_path: str
    img_bytes: bytes
    question: Optional[str] = None
    split: Optional[str] = None