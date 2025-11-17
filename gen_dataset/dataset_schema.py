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
    - question:     the natural language question
    - answer:       the expected model output
    - image_path:   where the associated board image is stored
    - split:        "train" | "eval" | "test" (filled later when creatin parquet file)
    """
    family: str
    q_id: str
    focus: str
    question: str
    answer: str
    image_path: str
    split: Optional[str] = None