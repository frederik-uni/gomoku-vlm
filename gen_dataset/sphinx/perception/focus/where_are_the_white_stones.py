import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    build_basic_dataset_row,
    get_question_meta,
    select_random_turn_and_store_image,
)


def _focus_where_are_the_white_stones(
    q_id: str, sim_id: int, simulated_game: np.ndarray
) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "where_are_the_white_stones"
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_random_turn_and_store_image(
        family=family,
        q_id=q_id,
        sim_id=sim_id,
        simulated_game=simulated_game,
    )

    # count black stones (=1) as ground truth
    indices = np.argwhere(board == 2)
    answer_text = str(indices.tolist())

    return build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer_text,
    )


def gen_question_q108_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q1 sample:
    focus: "count_black_stones"
    """
    q_id = "Q1"
    question_text = (
        "“You are looking at a Gomoku game position."
        "Black is player 1 and white is player 2."
        "Identify ONLY the white stones already placed on the board."
        "List the coordinates of all white stones as index pairs. Return only the list of index pairs.”"
    )

    dataset_row = _focus_where_are_the_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    # Optionally, add additional valid answers here

    return dataset_row


def gen_question_q109_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q2 sample:
    focus: "count_black_stones"
    """
    q_id = "Q2"
    question_text = (
        "“You are examining a Gomoku board state."
        "Player 1 corresponds to black stones and Player 2 to white stones."
        "Determine the locations of all existing white stones on the grid."
        "Provide the indices of each white stone as (row, column) pairs, and output nothing else.”"
    )

    dataset_row = _focus_where_are_the_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    return dataset_row


def gen_question_q110_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q3 sample:
    focus: "count_black_stones"
    """
    q_id = "Q3"
    question_text = (
        "“You are given a Gomoku position."
        "Black stones are encoded as 1 and white stones as 2."
        "Scan the board and find every cell containing a white stone."
        "Return the complete set of indices for white stones in row-major order.”"
    )

    dataset_row = _focus_where_are_the_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    return dataset_row


def gen_question_q111_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q4 sample:
    focus: "count_black_stones"
    """
    q_id = "Q4"
    question_text = (
        "“You are analyzing a Gomoku board configuration."
        "Black = 1, White = 2."
        "Your task is to extract the positions of ALL white stones already on the board."
        "Output the indices of all white stones as an ordered list of (row, column) tuples only.”"
    )

    dataset_row = _focus_where_are_the_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    return dataset_row
