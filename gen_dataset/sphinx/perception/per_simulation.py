from typing import List

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.perception.focus import count_black_stones


def generate_perception_questions_for_episode(sim_id: int, simulated_game: np.ndarray) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # focus: count_black_stones
    # Q1
    row = count_black_stones.gen_question_q1_sample(sim_id, simulated_game)
    rows.append(row)
    # Q2
    row = count_black_stones.gen_question_q2_sample(sim_id, simulated_game)
    rows.append(row)
    # Q3
    row = count_black_stones.gen_question_q3_sample(sim_id, simulated_game)
    rows.append(row)
    # Q4
    row = count_black_stones.gen_question_q4_sample(sim_id, simulated_game)
    rows.append(row)

    return rows
