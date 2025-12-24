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
) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # === Eugen ===
    # focus: win_next_turn
    if should_generate_question("Q10000", generated_questions_count):
        rows.append(win_next_turn.gen_question_q10000_sample(sim_id, simulated_game))
        generated_questions_count["Q10000"] = generated_questions_count.get("Q10000", 0) + 1

    if should_generate_question("Q10001", generated_questions_count):
        rows.append(win_next_turn.gen_question_q10001_sample(sim_id, simulated_game))
        generated_questions_count["Q10001"] = generated_questions_count.get("Q10001", 0) + 1

    if should_generate_question("Q10002", generated_questions_count):
        rows.append(win_next_turn.gen_question_q10002_sample(sim_id, simulated_game))
        generated_questions_count["Q10002"] = generated_questions_count.get("Q10002", 0) + 1

    if should_generate_question("Q10003", generated_questions_count):
        rows.append(win_next_turn.gen_question_q10003_sample(sim_id, simulated_game))
        generated_questions_count["Q10003"] = generated_questions_count.get("Q10003", 0) + 1

    # focus: best_next_move
    if should_generate_question("Q10100", generated_questions_count):
        rows.append(best_next_move.gen_question_q10100_sample(sim_id, simulated_game))
        generated_questions_count["Q10100"] = generated_questions_count.get("Q10100", 0) + 1

    if should_generate_question("Q10101", generated_questions_count):
        rows.append(best_next_move.gen_question_q10101_sample(sim_id, simulated_game))
        generated_questions_count["Q10101"] = generated_questions_count.get("Q10101", 0) + 1

    if should_generate_question("Q10102", generated_questions_count):
        rows.append(best_next_move.gen_question_q10102_sample(sim_id, simulated_game))
        generated_questions_count["Q10102"] = generated_questions_count.get("Q10102", 0) + 1

    if should_generate_question("Q10103", generated_questions_count):
        rows.append(best_next_move.gen_question_q10103_sample(sim_id, simulated_game))
        generated_questions_count["Q10103"] = generated_questions_count.get("Q10103", 0) + 1

    # focus: list_valid_moves
    if should_generate_question("Q10200", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q10200_sample(sim_id, simulated_game))
        generated_questions_count["Q10200"] = generated_questions_count.get("Q10200", 0) + 1

    if should_generate_question("Q10201", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q10201_sample(sim_id, simulated_game))
        generated_questions_count["Q10201"] = generated_questions_count.get("Q10201", 0) + 1

    if should_generate_question("Q10202", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q10202_sample(sim_id, simulated_game))
        generated_questions_count["Q10202"] = generated_questions_count.get("Q10202", 0) + 1

    if should_generate_question("Q10203", generated_questions_count):
        rows.append(list_valid_moves.gen_question_q10203_sample(sim_id, simulated_game))
        generated_questions_count["Q10203"] = generated_questions_count.get("Q10203", 0) + 1

    # focus: reason_next_move
    if should_generate_question("Q10300", generated_questions_count):
        rows.append(reason_next_move.gen_question_q10300_sample(sim_id, simulated_game))
        generated_questions_count["Q10300"] = generated_questions_count.get("Q10300", 0) + 1

    if should_generate_question("Q10301", generated_questions_count):
        rows.append(reason_next_move.gen_question_q10301_sample(sim_id, simulated_game))
        generated_questions_count["Q10301"] = generated_questions_count.get("Q10301", 0) + 1

    if should_generate_question("Q10302", generated_questions_count):
        rows.append(reason_next_move.gen_question_q10302_sample(sim_id, simulated_game))
        generated_questions_count["Q10302"] = generated_questions_count.get("Q10302", 0) + 1

    if should_generate_question("Q10303", generated_questions_count):
        rows.append(reason_next_move.gen_question_q10303_sample(sim_id, simulated_game))
        generated_questions_count["Q10303"] = generated_questions_count.get("Q10303", 0) + 1

    return rows
