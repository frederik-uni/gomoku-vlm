from typing import List

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.perception.focus import (
    count_black_stones,
    count_white_stones,
    count_empty_intersections,
    color_at_position,
    print_board_matrix,
    can_you_win,
    can_you_lose,
    where_are_the_black_stones,
    where_are_the_white_stones,
)


def generate_perception_questions_for_episode(
    sim_id: int, simulated_game: np.ndarray
) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # === Eugen ===
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

    # focus: color_at_position
    rows.append(color_at_position.gen_question_q13_sample(sim_id, simulated_game))
    rows.append(color_at_position.gen_question_q14_sample(sim_id, simulated_game))
    rows.append(color_at_position.gen_question_q15_sample(sim_id, simulated_game))
    rows.append(color_at_position.gen_question_q16_sample(sim_id, simulated_game))

    # focus: print_board_matrix
    rows.append(print_board_matrix.gen_question_q17_sample(sim_id, simulated_game))
    rows.append(print_board_matrix.gen_question_q18_sample(sim_id, simulated_game))
    rows.append(print_board_matrix.gen_question_q19_sample(sim_id, simulated_game))
    rows.append(print_board_matrix.gen_question_q20_sample(sim_id, simulated_game))

    # === Frederik ===
    # focus: can_you_win
    rows.append(can_you_win.gen_question_q100_sample(sim_id, simulated_game))
    rows.append(can_you_win.gen_question_q101_sample(sim_id, simulated_game))
    rows.append(can_you_win.gen_question_q102_sample(sim_id, simulated_game))
    rows.append(can_you_win.gen_question_q103_sample(sim_id, simulated_game))

    # focus: can_you_loose
    rows.append(can_you_loose.gen_question_q104_sample(sim_id, simulated_game))
    rows.append(can_you_loose.gen_question_q105_sample(sim_id, simulated_game))
    rows.append(can_you_loose.gen_question_q106_sample(sim_id, simulated_game))
    rows.append(can_you_loose.gen_question_q107_sample(sim_id, simulated_game))

    # focus: where_are_the_white_stones
    rows.append(where_are_the_white_stones.gen_question_q108_sample(sim_id, simulated_game))
    rows.append(where_are_the_white_stones.gen_question_q109_sample(sim_id, simulated_game))
    rows.append(where_are_the_white_stones.gen_question_q110_sample(sim_id, simulated_game))
    rows.append(where_are_the_white_stones.gen_question_q111_sample(sim_id, simulated_game))

    # focus: where_are_the_black_stones
    rows.append(where_are_the_black_stones.gen_question_q112_sample(sim_id, simulated_game))
    rows.append(where_are_the_black_stones.gen_question_q113_sample(sim_id, simulated_game))
    rows.append(where_are_the_black_stones.gen_question_q114_sample(sim_id, simulated_game))
    rows.append(where_are_the_black_stones.gen_question_q115_sample(sim_id, simulated_game))

    return rows
