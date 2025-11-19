from typing import List

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.perception.focus.count_stones import gen_question_q1_sample


def generate_perception_questions_for_episode(sim_id: int, simulated_game: np.ndarray) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # Q1
    row = gen_question_q1_sample(sim_id, simulated_game)
    rows.append(row)
    # Q2
    # ...

    return rows
