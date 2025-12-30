from typing import List, Dict

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.strategy.focus import (
    win_next_turn,
    best_next_move,
    list_valid_moves, reason_next_move
)
from ..core import should_generate_question


def generate_strategy_questions_for_episode(
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
    # Q1001–Q1004: focus: list_valid_moves
    # ============================================================
    if should_generate_question("Q1001", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q1001_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1001"] = generated_questions_count.get("Q1001", 0) + 1

    if should_generate_question("Q1002", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q1002_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1002"] = generated_questions_count.get("Q1002", 0) + 1

    if should_generate_question("Q1003", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q1003_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1003"] = generated_questions_count.get("Q1003", 0) + 1

    if should_generate_question("Q1004", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q1004_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1004"] = generated_questions_count.get("Q1004", 0) + 1


    # ============================================================
    # Q1101–Q1104: focus: win_next_turn
    # ============================================================
    if should_generate_question("Q1101", generated_questions_count):
        rows.append(win_next_turn.gen_question_q1101_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1101"] = generated_questions_count.get("Q1101", 0) + 1

    if should_generate_question("Q1102", generated_questions_count):
        rows.append(win_next_turn.gen_question_q1102_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1102"] = generated_questions_count.get("Q1102", 0) + 1

    if should_generate_question("Q1103", generated_questions_count):
        rows.append(win_next_turn.gen_question_q1103_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1103"] = generated_questions_count.get("Q1103", 0) + 1

    if should_generate_question("Q1104", generated_questions_count):
        rows.append(win_next_turn.gen_question_q1104_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1104"] = generated_questions_count.get("Q1104", 0) + 1


    # ============================================================
    # Q1201–Q1204: focus: best_next_move
    # ============================================================
    if should_generate_question("Q1201", generated_questions_count):
        rows.append(best_next_move.gen_question_q1201_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1201"] = generated_questions_count.get("Q1201", 0) + 1

    if should_generate_question("Q1202", generated_questions_count):
        rows.append(best_next_move.gen_question_q1202_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1202"] = generated_questions_count.get("Q1202", 0) + 1

    if should_generate_question("Q1203", generated_questions_count):
        rows.append(best_next_move.gen_question_q1203_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1203"] = generated_questions_count.get("Q1203", 0) + 1

    if should_generate_question("Q1204", generated_questions_count):
        rows.append(best_next_move.gen_question_q1204_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1204"] = generated_questions_count.get("Q1204", 0) + 1


    # ============================================================
    # Q1301–Q1304: focus: reason_next_move
    # ============================================================
    if should_generate_question("Q1301", generated_questions_count):
        rows.append(reason_next_move.gen_question_q1301_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1301"] = generated_questions_count.get("Q1301", 0) + 1

    if should_generate_question("Q1302", generated_questions_count):
        rows.append(reason_next_move.gen_question_q1302_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1302"] = generated_questions_count.get("Q1302", 0) + 1

    if should_generate_question("Q1303", generated_questions_count):
        rows.append(reason_next_move.gen_question_q1303_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1303"] = generated_questions_count.get("Q1303", 0) + 1

    if should_generate_question("Q1304", generated_questions_count):
        rows.append(reason_next_move.gen_question_q1304_sample(sim_id, simulated_game, non_rand_img))
        generated_questions_count["Q1304"] = generated_questions_count.get("Q1304", 0) + 1

    return rows
