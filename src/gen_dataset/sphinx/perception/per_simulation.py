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
    can_you_lose,
    three_in_a_row,
    four_in_a_row,
)
from ..core import is_question_configured, should_generate_question


def generate_perception_questions_for_episode(
    sim_id: int,
    simulated_game: np.ndarray,
    generated_questions_count: Dict[str, int],
    non_rand_img: bool,
) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # ============================================================
    # Q1–Q4: focus: color_at_position
    # ============================================================
    if should_generate_question("Q1", generated_questions_count):
        rows.append(color_at_position.gen_question_q1_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1"] = generated_questions_count.get("Q1", 0) + 1

    if should_generate_question("Q2", generated_questions_count):
        rows.append(color_at_position.gen_question_q2_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q2"] = generated_questions_count.get("Q2", 0) + 1

    if should_generate_question("Q3", generated_questions_count):
        rows.append(color_at_position.gen_question_q3_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q3"] = generated_questions_count.get("Q3", 0) + 1

    if should_generate_question("Q4", generated_questions_count):
        rows.append(color_at_position.gen_question_q4_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q4"] = generated_questions_count.get("Q4", 0) + 1


    # ============================================================
    # Q101–Q104: focus: count_black_stones
    # ============================================================
    if should_generate_question("Q101", generated_questions_count):
        rows.append(count_black_stones.gen_question_q101_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q101"] = generated_questions_count.get("Q101", 0) + 1

    if should_generate_question("Q102", generated_questions_count):
        rows.append(count_black_stones.gen_question_q102_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q102"] = generated_questions_count.get("Q102", 0) + 1

    if should_generate_question("Q103", generated_questions_count):
        rows.append(count_black_stones.gen_question_q103_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q103"] = generated_questions_count.get("Q103", 0) + 1

    if should_generate_question("Q104", generated_questions_count):
        rows.append(count_black_stones.gen_question_q104_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q104"] = generated_questions_count.get("Q104", 0) + 1


    # ============================================================
    # Q201–Q204: focus: count_white_stones
    # ============================================================
    if should_generate_question("Q201", generated_questions_count):
        rows.append(count_white_stones.gen_question_q201_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q201"] = generated_questions_count.get("Q201", 0) + 1

    if should_generate_question("Q202", generated_questions_count):
        rows.append(count_white_stones.gen_question_q202_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q202"] = generated_questions_count.get("Q202", 0) + 1

    if should_generate_question("Q203", generated_questions_count):
        rows.append(count_white_stones.gen_question_q203_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q203"] = generated_questions_count.get("Q203", 0) + 1

    if should_generate_question("Q204", generated_questions_count):
        rows.append(count_white_stones.gen_question_q204_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q204"] = generated_questions_count.get("Q204", 0) + 1


    # ============================================================
    # Q301–Q304: focus: count_empty_intersections
    # ============================================================
    if should_generate_question("Q301", generated_questions_count):
        rows.append(count_empty_intersections.gen_question_q301_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q301"] = generated_questions_count.get("Q301", 0) + 1

    if should_generate_question("Q302", generated_questions_count):
        rows.append(count_empty_intersections.gen_question_q302_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q302"] = generated_questions_count.get("Q302", 0) + 1

    if should_generate_question("Q303", generated_questions_count):
        rows.append(count_empty_intersections.gen_question_q303_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q303"] = generated_questions_count.get("Q303", 0) + 1

    if should_generate_question("Q304", generated_questions_count):
        rows.append(count_empty_intersections.gen_question_q304_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q304"] = generated_questions_count.get("Q304", 0) + 1


    # ============================================================
    # Q401–Q404: focus: three_in_a_row
    # ============================================================
    if should_generate_question("Q401", generated_questions_count):
        rows.append(three_in_a_row.gen_question_q401_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q401"] = generated_questions_count.get("Q401", 0) + 1

    if should_generate_question("Q402", generated_questions_count):
        rows.append(three_in_a_row.gen_question_q402_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q402"] = generated_questions_count.get("Q402", 0) + 1

    if should_generate_question("Q403", generated_questions_count):
        rows.append(three_in_a_row.gen_question_q403_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q403"] = generated_questions_count.get("Q403", 0) + 1

    if should_generate_question("Q404", generated_questions_count):
        rows.append(three_in_a_row.gen_question_q404_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q404"] = generated_questions_count.get("Q404", 0) + 1


    # ============================================================
    # Q501–Q504: focus: four_in_a_row
    # ============================================================
    if should_generate_question("Q501", generated_questions_count):
        rows.append(four_in_a_row.gen_question_q501_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q501"] = generated_questions_count.get("Q501", 0) + 1

    if should_generate_question("Q502", generated_questions_count):
        rows.append(four_in_a_row.gen_question_q502_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q502"] = generated_questions_count.get("Q502", 0) + 1

    if should_generate_question("Q503", generated_questions_count):
        rows.append(four_in_a_row.gen_question_q503_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q503"] = generated_questions_count.get("Q503", 0) + 1

    if should_generate_question("Q504", generated_questions_count):
        rows.append(four_in_a_row.gen_question_q504_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q504"] = generated_questions_count.get("Q504", 0) + 1


    # ============================================================
    # Q601–Q604: focus: determine_who_won
    # ============================================================
    if should_generate_question("Q601", generated_questions_count):
        rows.append(determine_who_won.gen_question_q601_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q601"] = generated_questions_count.get("Q601", 0) + 1

    if should_generate_question("Q602", generated_questions_count):
        rows.append(determine_who_won.gen_question_q602_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q602"] = generated_questions_count.get("Q602", 0) + 1

    if should_generate_question("Q603", generated_questions_count):
        rows.append(determine_who_won.gen_question_q603_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q603"] = generated_questions_count.get("Q603", 0) + 1

    if should_generate_question("Q604", generated_questions_count):
        rows.append(determine_who_won.gen_question_q604_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q604"] = generated_questions_count.get("Q604", 0) + 1


    # ============================================================
    # Q701–Q704: focus: can_you_win
    # ============================================================
    if should_generate_question("Q701", generated_questions_count):
        rows.append(can_you_win.gen_question_q701_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q701"] = generated_questions_count.get("Q701", 0) + 1

    if should_generate_question("Q702", generated_questions_count):
        rows.append(can_you_win.gen_question_q702_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q702"] = generated_questions_count.get("Q702", 0) + 1

    if should_generate_question("Q703", generated_questions_count):
        rows.append(can_you_win.gen_question_q703_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q703"] = generated_questions_count.get("Q703", 0) + 1

    if should_generate_question("Q704", generated_questions_count):
        rows.append(can_you_win.gen_question_q704_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q704"] = generated_questions_count.get("Q704", 0) + 1


    # ============================================================
    # Q801–Q804: focus: can_you_lose
    # ============================================================
    if should_generate_question("Q801", generated_questions_count):
        rows.append(can_you_lose.gen_question_q801_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q801"] = generated_questions_count.get("Q801", 0) + 1

    if should_generate_question("Q802", generated_questions_count):
        rows.append(can_you_lose.gen_question_q802_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q802"] = generated_questions_count.get("Q802", 0) + 1

    if should_generate_question("Q803", generated_questions_count):
        rows.append(can_you_lose.gen_question_q803_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q803"] = generated_questions_count.get("Q803", 0) + 1

    if should_generate_question("Q804", generated_questions_count):
        rows.append(can_you_lose.gen_question_q804_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q804"] = generated_questions_count.get("Q804", 0) + 1


    # ============================================================
    # Q901–Q904: focus: print_board_matrix
    # ============================================================
    if should_generate_question("Q901", generated_questions_count):
        rows.append(print_board_matrix.gen_question_q901_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q901"] = generated_questions_count.get("Q901", 0) + 1

    if should_generate_question("Q902", generated_questions_count):
        rows.append(print_board_matrix.gen_question_q902_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q902"] = generated_questions_count.get("Q902", 0) + 1

    if should_generate_question("Q903", generated_questions_count):
        rows.append(print_board_matrix.gen_question_q903_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q903"] = generated_questions_count.get("Q903", 0) + 1

    if should_generate_question("Q904", generated_questions_count):
        rows.append(print_board_matrix.gen_question_q904_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q904"] = generated_questions_count.get("Q904", 0) + 1

    return rows
