from typing import List, Dict

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.perception.focus import (
    count_black_stones,
    count_white_stones,
    count_empty_intersections,
    color_at_position,
    print_board_matrix,
    determine_who_won,
    can_you_win,
    can_you_lose
)
from ..core import is_question_configured, should_generate_question


def generate_perception_questions_for_episode(
    sim_id: int,
    simulated_game: np.ndarray,
    generated_questions_count: Dict[str, int],
) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # === Eugen ===
    # focus: count_black_stones
    if should_generate_question("Q100", generated_questions_count):
        rows.append(count_black_stones.gen_question_q100_sample(sim_id, simulated_game))
        generated_questions_count["Q100"] = generated_questions_count.get("Q100", 0) + 1

    if should_generate_question("Q101", generated_questions_count):
        rows.append(count_black_stones.gen_question_q101_sample(sim_id, simulated_game))
        generated_questions_count["Q101"] = generated_questions_count.get("Q101", 0) + 1

    if should_generate_question("Q102", generated_questions_count):
        rows.append(count_black_stones.gen_question_q102_sample(sim_id, simulated_game))
        generated_questions_count["Q102"] = generated_questions_count.get("Q102", 0) + 1

    if should_generate_question("Q103", generated_questions_count):
        rows.append(count_black_stones.gen_question_q103_sample(sim_id, simulated_game))
        generated_questions_count["Q103"] = generated_questions_count.get("Q103", 0) + 1

    # # focus: count_white_stones
    # if is_question_configured("Q200"):
    #     rows.append(count_white_stones.gen_question_q200_sample(sim_id, simulated_game))
    # if is_question_configured("Q201"):
    #     rows.append(count_white_stones.gen_question_q201_sample(sim_id, simulated_game))
    # if is_question_configured("Q202"):
    #     rows.append(count_white_stones.gen_question_q202_sample(sim_id, simulated_game))
    # if is_question_configured("Q203"):
    #     rows.append(count_white_stones.gen_question_q203_sample(sim_id, simulated_game))
    #
    # # # focus: count_empty_intersections
    # if is_question_configured("Q300"):
    #     rows.append(count_empty_intersections.gen_question_q300_sample(sim_id, simulated_game))
    # if is_question_configured("Q301"):
    #     rows.append(count_empty_intersections.gen_question_q301_sample(sim_id, simulated_game))
    # if is_question_configured("Q302"):
    #     rows.append(count_empty_intersections.gen_question_q302_sample(sim_id, simulated_game))
    # if is_question_configured("Q303"):
    #     rows.append(count_empty_intersections.gen_question_q303_sample(sim_id, simulated_game))
    #
    # # focus: color_at_position
    # if is_question_configured("Q400"):
    #     rows.append(color_at_position.gen_question_q400_sample(sim_id, simulated_game))
    # if is_question_configured("Q401"):
    #     rows.append(color_at_position.gen_question_q401_sample(sim_id, simulated_game))
    # if is_question_configured("Q402"):
    #     rows.append(color_at_position.gen_question_q402_sample(sim_id, simulated_game))
    # if is_question_configured("Q403"):
    #     rows.append(color_at_position.gen_question_q403_sample(sim_id, simulated_game))
    #
    # # focus: print_board_matrix
    # if is_question_configured("Q500"):
    #     rows.append(print_board_matrix.gen_question_q500_sample(sim_id, simulated_game))
    # if is_question_configured("Q501"):
    #     rows.append(print_board_matrix.gen_question_q501_sample(sim_id, simulated_game))
    # if is_question_configured("Q502"):
    #     rows.append(print_board_matrix.gen_question_q502_sample(sim_id, simulated_game))
    # if is_question_configured("Q503"):
    #     rows.append(print_board_matrix.gen_question_q503_sample(sim_id, simulated_game))
    #
    # # focus: determine_who_won
    # if is_question_configured("Q600"):
    #     rows.append(determine_who_won.gen_question_q600_sample(sim_id, simulated_game))
    # if is_question_configured("Q601"):
    #     rows.append(determine_who_won.gen_question_q601_sample(sim_id, simulated_game))
    # if is_question_configured("Q602"):
    #     rows.append(determine_who_won.gen_question_q602_sample(sim_id, simulated_game))
    # if is_question_configured("Q603"):
    #     rows.append(determine_who_won.gen_question_q603_sample(sim_id, simulated_game))

    # # === Frederik ===
    # # focus: can_you_win
    # if is_question_configured("Q700"):
    #     rows.append(can_you_win.gen_question_q700_sample(sim_id, simulated_game))
    # if is_question_configured("Q701"):
    #     rows.append(can_you_win.gen_question_q701_sample(sim_id, simulated_game))
    # if is_question_configured("Q702"):
    #     rows.append(can_you_win.gen_question_q702_sample(sim_id, simulated_game))
    # if is_question_configured("Q703"):
    #     rows.append(can_you_win.gen_question_q703_sample(sim_id, simulated_game))
    #
    # # focus: can_you_loose
    # if is_question_configured("Q800"):
    #     rows.append(can_you_lose.gen_question_q800_sample(sim_id, simulated_game))
    # if is_question_configured("Q801"):
    #     rows.append(can_you_lose.gen_question_q801_sample(sim_id, simulated_game))
    # if is_question_configured("Q802"):
    #     rows.append(can_you_lose.gen_question_q802_sample(sim_id, simulated_game))
    # if is_question_configured("Q803"):
    #     rows.append(can_you_lose.gen_question_q803_sample(sim_id, simulated_game))

    return rows
