import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import QuestionFamily, build_basic_dataset_row, select_random_turn_and_store_image, get_question_meta


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


def _print_board_matrix(q_id: str, sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: 'print_board_matrix'.
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_random_turn_and_store_image(family=family, q_id=q_id, sim_id=sim_id, simulated_game=simulated_game)

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


def gen_question_q17_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q17 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q17"
    dataset_row = _print_board_matrix(q_id, sim_id, simulated_game)

    question_text = (
        "You are looking at a Gomoku board position. "
        "The board is a 15×15 grid. "
        "Encode the entire board as a 15×15 matrix of integers. "
        "Use the following encoding: 0 for empty intersections, "
        "1 for black stones (player 1), and 2 for white stones (player 2). "
        "Output exactly 15 rows, each row containing 15 integers separated by a single space. "
        "Do not include any extra text, explanation, brackets, or punctuation. "
        "Answer ONLY with the matrix."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q18_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q18 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q18"
    dataset_row = _print_board_matrix(q_id, sim_id, simulated_game)

    question_text = (
        "This image shows a Gomoku position on a 15×15 board. "
        "Represent the current board state as a numeric matrix. "
        "Use 0 for an empty cell, 1 for a black stone (player 1), and 2 for a white stone (player 2). "
        "Write the matrix row by row from top to bottom. "
        "Each row must contain exactly 15 integers separated by a single space. "
        "Return ONLY the 15 lines of numbers, with no additional commentary or symbols."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q19_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q19 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q19"
    dataset_row = _print_board_matrix(q_id, sim_id, simulated_game)

    question_text = (
        "Consider the displayed Gomoku game state on a 15×15 grid. "
        "Convert the full board into a textual 15×15 matrix. "
        "For each intersection, output 0 if it is empty, 1 if it contains a black stone (player 1), "
        "or 2 if it contains a white stone (player 2). "
        "Write one row per line, from the top row to the bottom row, with 15 space-separated integers in each row. "
        "Your answer must consist ONLY of these 15 numeric rows."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q20_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q20 sample:
    focus: "print_board_matrix"
    """
    q_id = "Q20"
    dataset_row = _print_board_matrix(q_id, sim_id, simulated_game)

    question_text = (
        "You are given an image of a Gomoku board (15×15). "
        "Encode the board configuration as a matrix of integers using this scheme: "
        "0 = empty intersection, 1 = black stone (player 1), 2 = white stone (player 2). "
        "Output the complete board as 15 lines, each line containing 15 integers separated by single spaces. "
        "Do not add brackets, commas, words, or any explanation. "
        "Respond strictly with the 15-line matrix."
    )

    dataset_row.question = question_text
    return dataset_row

