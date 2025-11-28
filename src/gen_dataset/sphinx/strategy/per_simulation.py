from typing import List

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.strategy.focus import (
    win_next_turn,
    best_next_move,
    list_valid_moves
)
from ..core import is_question_configured


def generate_strategy_questions_for_episode(
    sim_id: int, simulated_game: np.ndarray
) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # === Eugen ===
    # focus: win_next_turn
    if is_question_configured("Q10000"):
        rows.append(win_next_turn.gen_question_q10000_sample(sim_id, simulated_game))
    if is_question_configured("Q10001"):
        rows.append(win_next_turn.gen_question_q10001_sample(sim_id, simulated_game))
    if is_question_configured("Q10002"):
        rows.append(win_next_turn.gen_question_q10002_sample(sim_id, simulated_game))
    if is_question_configured("Q10003"):
        rows.append(win_next_turn.gen_question_q10003_sample(sim_id, simulated_game))

    # focus: best_next_move
    if is_question_configured("Q10100"):
        rows.append(best_next_move.gen_question_q10100_sample(sim_id, simulated_game))
    if is_question_configured("Q10101"):
        rows.append(best_next_move.gen_question_q10101_sample(sim_id, simulated_game))
    if is_question_configured("Q10102"):
        rows.append(best_next_move.gen_question_q10102_sample(sim_id, simulated_game))
    if is_question_configured("Q10103"):
        rows.append(best_next_move.gen_question_q10103_sample(sim_id, simulated_game))

    # focus: list_valid_moves
    if is_question_configured("Q102000"):
        rows.append(list_valid_moves.gen_question_q102000_sample(sim_id, simulated_game))
    if is_question_configured("Q102001"):
        rows.append(list_valid_moves.gen_question_q102001_sample(sim_id, simulated_game))
    if is_question_configured("Q102002"):
        rows.append(list_valid_moves.gen_question_q102002_sample(sim_id, simulated_game))
    if is_question_configured("Q102003"):
        rows.append(list_valid_moves.gen_question_q102003_sample(sim_id, simulated_game))

    return rows
