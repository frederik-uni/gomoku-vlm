import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    build_basic_dataset_row,
    select_turn_and_store_image,
    get_question_meta,
)


def _focus_count_black_stones(q_id: str, sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "count_black_stones"
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_turn_and_store_image(
        family=family,
        q_id=q_id,
        sim_id=sim_id,
        simulated_game=simulated_game,
    )

    # count black stones (=1) as ground truth
    num_black = int(np.count_nonzero(board == 1))
    answer_text = str(num_black)

    return build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer_text,
    )


def gen_question_q1_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q1 sample:
    focus: "count_black_stones"
    """
    q_id = "Q1"
    question_text = (
        "You are looking at a Gomoku game position. "
        "Black is player 1 and white is player 2. "
        "Count ONLY the black stones already placed on the board. "
        "How many black stones are there in total? Answer with a single integer."
    )

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    # Optionally, add additional valid answers here

    return dataset_row


def gen_question_q2_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q2 sample:
    focus: "count_black_stones"
    """
    q_id = "Q2"
    question_text = (
        "Your task is to analyze this Gomoku board position. "
        "Player 1 is represented by black stones, player 2 by white stones. "
        "Ignore all empty intersections and all white stones. "
        "How many black stones belonging to player 1 are currently on the board? "
        "Answer with a single integer (for example: 7)."
    )

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    return dataset_row


def gen_question_q3_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q3 sample:
    focus: "count_black_stones"
    """
    q_id = "Q3"
    question_text = (
        "This image shows a snapshot of a Gomoku game in progress. "
        "Black stones belong to player 1, white stones belong to player 2. "
        "Count how many board intersections are occupied by black stones only. "
        "What is the total number of black stones on the board? "
        "Respond using only one integer and no additional text."
    )

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    return dataset_row


def gen_question_q4_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q4 sample:
    focus: "count_black_stones"
    """
    q_id = "Q4"
    question_text = (
        "You are inspecting a partially played Gomoku position. "
        "Player 1 uses black stones, and player 2 uses white stones. "
        "Determine how many black stones player 1 has placed on the board so far. "
        "Do not count white stones or empty cells. "
        "Return only the total number of black stones as a single integer value."
    )

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text

    return dataset_row
