import random

import numpy as np

import game_logic
from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily, persist_turn_image, get_random_turn_index, get_question_text
)


def _focus_three_in_a_row(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    *,
    min_turn = 0,
    max_turn = 999,
    player = -1
) -> tuple[int, str, int, DatasetRow]:
    """
    Helper function for any question that has the focus: "three_in_a_row".

    Definition:
        A "three-in-a-row" is any contiguous window of length 3
        (horizontal, vertical, or diagonal) where all 3 cells are occupied
        by the given player's stones.

    Params:
        q_id: The q_id of the question.
        sim_id: The sim_id of the simulation.
        game: Sequence of board states (game[turn_idx] is a 15×15 board).
        min_turn: Minimum turn index to sample from.
        max_turn: Maximum turn index to sample from.
        player: 1 (black) or 2 (white). If -1, choose randomly.

    Returns:
        tuple[int, str, int, DatasetRow]: A tuple containing:
            - player: 1 if black, 2 if white.
            - color: "black" or "white".
            - num_three_in_a_row: Count of length-3 segments for player.
            - row: Pre-constructed DatasetRow (question still None).
    """
    FOCUS = "three_in_a_row"
    FAMILY = QuestionFamily.PERCEPTION

    if player == -1:
        player = 1 if random.random() < 0.5 else 2
    color = "black" if player == 1 else "white"

    idx = get_random_turn_index(game, min_turn, max_turn)
    board = game[idx]
    num_three_in_a_row = game_logic.count(board, 3, player, almost=False)
    answer = str(num_three_in_a_row)

    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, idx, sim_id)

    return player, color, num_three_in_a_row, DatasetRow(
        img_path=str(img_path),
        img_bytes=img_bytes,
        family=FAMILY,
        q_id=q_id,
        focus=FOCUS,
        answer=answer,
        valid_answers=[answer],
        question=None,
        split=None,
    )

def gen_question_q900_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    q_id = "Q900"
    player, color, _count, row = _focus_three_in_a_row(
        q_id, sim_id, simulated_game, player=1
    )
    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)
    return row
