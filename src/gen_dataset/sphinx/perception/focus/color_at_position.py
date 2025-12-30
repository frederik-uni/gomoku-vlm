import random

import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    persist_turn_game_state,
    get_question_text
)

def _focus_color_at_position(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    non_rand_img: bool,
    min_turns: int = 0,
    max_turns: int = 999
) -> tuple[DatasetRow, int, int]:
    """
    Helper function for any question that has the
    focus: "color_at_position".

    Returns:
        tuple[DatasetRow, int, int]:
            - DatasetRow with answer + valid_answers filled (question still None)
            - row_idx: 0-based row index on the board (to get vertical, "y")
            - col_idx: 0-based column index on the board (to get horizontal, "x")
    """
    FOCUS = "color_at_position"
    FAMILY = QuestionFamily.PERCEPTION

    # Sample random turn
    turn_index = get_random_turn_index(game, min_turns, max_turns)
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id, non_rand_img=non_rand_img)
    # Persist game state for easier debugging
    persist_turn_game_state(board, turn_index, sim_id)

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

    row = DatasetRow(
        img_path=str(img_path),
        img_bytes=img_bytes,

        family=FAMILY,
        q_id=q_id,
        focus=FOCUS,

        answer=answer,
        valid_answers=valid_answers,

        # Will be assigned later in the creation process
        question=None,
        split=None
    )

    return row, row_idx, col_idx


def gen_question_q1_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1 sample:
    focus: "color_at_position"
    """
    q_id = "Q1"
    min_turns = 0
    max_turns = 999

    dataset_row, row_idx, col_idx = _focus_color_at_position(
        q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns
    )

    template = get_question_text(q_id)
    dataset_row.question = template.format(row=row_idx, col=col_idx)

    return dataset_row


def gen_question_q2_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q2 sample:
    focus: "color_at_position"
    """
    q_id = "Q2"
    min_turns = 25
    max_turns = 999

    dataset_row, row_idx, col_idx = _focus_color_at_position(
        q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns
    )

    template = get_question_text(q_id)
    dataset_row.question = template.format(row=row_idx, col=col_idx)

    return dataset_row


def gen_question_q3_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q3 sample:
    focus: "color_at_position"
    """
    q_id = "Q3"
    min_turns = 50
    max_turns = 999

    dataset_row, row_idx, col_idx = _focus_color_at_position(
        q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns
    )

    template = get_question_text(q_id)
    dataset_row.question = template.format(row=row_idx, col=col_idx)

    return dataset_row


def gen_question_q4_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q4 sample:
    focus: "color_at_position"
    """
    q_id = "Q4"
    min_turns = 75
    max_turns = 999

    dataset_row, row_idx, col_idx = _focus_color_at_position(
        q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns
    )

    template = get_question_text(q_id)
    dataset_row.question = template.format(row=row_idx, col=col_idx)

    return dataset_row
