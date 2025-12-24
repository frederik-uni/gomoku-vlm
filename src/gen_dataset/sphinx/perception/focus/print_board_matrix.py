import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    persist_turn_game_state,
    get_question_text
)


def board_to_matrix_string(board: np.ndarray) -> str:
    """
    Convert a 2D board array into a plain text matrix:

    0 1 0 2
    0 0 1 0
    ...

    using NumPy's own formatting, then stripping brackets.
    """
    if board.ndim != 2:
        raise ValueError(f"Expected 2D board, got shape {board.shape}")

    # Formal np array to string
    s = np.array2string(board, separator=" ")

    # Remove brackets
    s = s.replace("[", "").replace("]", "")

    # Clean up each line (strip leading/trailing spaces)
    lines = [line.strip() for line in s.splitlines()]

    # Join back with newlines -> nice matrix
    return "\n".join(lines)


def _focus_print_board_matrix(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    min_turns: int = 0,
    max_turns: int = 999
) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: 'print_board_matrix'.
    """
    FOCUS = "print_board_matrix"
    FAMILY = QuestionFamily.PERCEPTION

    # Sample random turn
    turn_index = get_random_turn_index(game, min_turns, max_turns)
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id)
    # Persist game state for easier debugging
    persist_turn_game_state(board, turn_index, sim_id)

    # serialize the board as text matrix
    answer = board_to_matrix_string(board)
    valid_answers = [answer]

    return DatasetRow(
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


def gen_question_q500_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q500 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q500"
    dataset_row = _focus_print_board_matrix(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)
    return dataset_row


def gen_question_q501_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q501 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q501"
    dataset_row = _focus_print_board_matrix(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)
    return dataset_row


def gen_question_q502_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q502 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q502"
    dataset_row = _focus_print_board_matrix(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)
    return dataset_row


def gen_question_q503_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q503 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q503"
    dataset_row = _focus_print_board_matrix(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)
    return dataset_row