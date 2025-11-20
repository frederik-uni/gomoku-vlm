from typing import List

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.strategy.focus import (
    win_next_turn
)


def generate_strategy_questions_for_episode(
    sim_id: int, simulated_game: np.ndarray
) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # === Eugen ===
    # focus: win_next_turn
    rows.append(win_next_turn.gen_question_q500_sample(sim_id, simulated_game))
    rows.append(win_next_turn.gen_question_q501_sample(sim_id, simulated_game))
    rows.append(win_next_turn.gen_question_q502_sample(sim_id, simulated_game))
    rows.append(win_next_turn.gen_question_q503_sample(sim_id, simulated_game))

    return rows
