import random

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    build_basic_dataset_row,
    select_random_turn_and_store_image,
    get_question_meta,
)
import random

def _focus_color_at_position(q_id: str, sim_id: int, simulated_game: np.ndarray) -> tuple[DatasetRow, int, int]:
    """
    Helper function for any question that has the
    focus: "color_at_position".

    Returns:
        tuple[DatasetRow, int, int]:
            - DatasetRow with answer + valid_answers filled (question still None)
            - row_idx: 0-based row index on the board (to get vertical, "y")
            - col_idx: 0-based column index on the board (to get horizontal, "x")
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_random_turn_and_store_image(
        family=family,
        q_id=q_id,
        sim_id=sim_id,
        simulated_game=simulated_game,
    )

    # pick a random board coordinate (row, col) in 0-based indexing
    num_rows, num_cols = board.shape
    row_idx = random.randint(0, num_rows - 1)
    col_idx = random.randint(0, num_cols - 1)

    # look up the value at that position
    # first index -> row (vertical, "y")
    # second index -> column (horizontal, "x")
    value = int(board[row_idx, col_idx])

    match value:
        case 0:
            answer = "empty"
            valid_answers = ["empty", "no stone", "none", "nothing"]
        case 1:
            answer = "black"
            valid_answers = ["black", "black stone", "player 1"]
        case 2:
            answer = "white"
            valid_answers = ["white", "white stone", "player 2"]
        case _:
            raise ValueError(f"Unexpected board value {value} at ({row_idx}, {col_idx})")

    row = build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer,
        valid_answers=valid_answers,
    )

    return row, row_idx, col_idx


def gen_question_q13_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q13 sample:
    focus: "color_at_position"
    """
    q_id = "Q13"
    dataset_row, row_idx, col_idx = _focus_color_at_position(q_id, sim_id, simulated_game)

    question_text = (
        "You are looking at a Gomoku game position. "
        "Black stones belong to player 1 and white stones belong to player 2. "
        f"Consider the board intersection at row {row_idx}, column {col_idx} (0-based indexing). "
        "Which piece is placed on that intersection? "
        "Answer with exactly one word: 'black', 'white', or 'empty'."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q14_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q14 sample:
    focus: "color_at_position"
    """
    q_id = "Q14"
    dataset_row, row_idx, col_idx = _focus_color_at_position(q_id, sim_id, simulated_game)

    question_text = (
        "You are looking at a Gomoku game position. "
        "Black stones belong to player 1 and white stones belong to player 2. "
        f"Consider the board intersection at row {row_idx}, column {col_idx} (0-based indexing). "
        "Which piece is placed on that intersection? "
        "Answer with exactly one word: 'black', 'white', or 'empty'."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q15_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q15 sample:
    focus: "color_at_position"
    """
    q_id = "Q15"
    dataset_row, row_idx, col_idx = _focus_color_at_position(q_id, sim_id, simulated_game)

    question_text = (
        "You are inspecting a partially played Gomoku game. "
        "Some intersections are occupied by black or white stones, others are empty. "
        f"Consider the intersection located at row {row_idx}, column {col_idx} (0-based indexing). "
        "Which piece occupies this intersection: 'black', 'white', or 'empty'? "
        "Reply with exactly one of these three words."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q16_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q16 sample:
    focus: "color_at_position"
    """
    q_id = "Q16"
    dataset_row, row_idx, col_idx = _focus_color_at_position(q_id, sim_id, simulated_game)

    question_text = (
        "You are inspecting a partially played Gomoku game. "
        "Some intersections are occupied by black or white stones, others are empty. "
        f"Consider the intersection located at row {row_idx}, column {col_idx} (0-based indexing). "
        "Which piece occupies this intersection: 'black', 'white', or 'empty'? "
        "Reply with exactly one of these three words."
    )

    dataset_row.question = question_text
    return dataset_row