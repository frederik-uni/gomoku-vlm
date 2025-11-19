import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import QuestionFamily, build_basic_dataset_row, select_turn_and_store_image, get_question_meta


def _board_to_matrix_string(board: np.ndarray) -> str:
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


def _focus_encode_board_matrix(q_id: str, sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: 'encode_board_matrix'.
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_turn_and_store_image(family=family, q_id=q_id, sim_id=sim_id, simulated_game=simulated_game)

    # serialize the board as text matrix
    answer_text = _board_to_matrix_string(board)
    valid_answers = [answer_text]

    return build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer_text,
        valid_answers=valid_answers,
    )
