from typing import List

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.perception.focus import (
    count_black_stones,
    count_white_stones,
    count_empty_intersections
)


def generate_perception_questions_for_episode(sim_id: int, simulated_game: np.ndarray) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # focus: count_black_stones
    rows.append(count_black_stones.gen_question_q1_sample(sim_id, simulated_game))
    rows.append(count_black_stones.gen_question_q2_sample(sim_id, simulated_game))
    rows.append(count_black_stones.gen_question_q3_sample(sim_id, simulated_game))
    rows.append(count_black_stones.gen_question_q4_sample(sim_id, simulated_game))

    # focus: count_white_stones
    rows.append(count_white_stones.gen_question_q5_sample(sim_id, simulated_game))
    rows.append(count_white_stones.gen_question_q6_sample(sim_id, simulated_game))
    rows.append(count_white_stones.gen_question_q7_sample(sim_id, simulated_game))
    rows.append(count_white_stones.gen_question_q8_sample(sim_id, simulated_game))

    # focus: count_empty_intersections
    rows.append(count_empty_intersections.gen_question_q9_sample(sim_id, simulated_game))
    rows.append(count_empty_intersections.gen_question_q10_sample(sim_id, simulated_game))
    rows.append(count_empty_intersections.gen_question_q11_sample(sim_id, simulated_game))
    rows.append(count_empty_intersections.gen_question_q12_sample(sim_id, simulated_game))

    return rows
