import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    build_basic_dataset_row,
    select_random_turn_and_store_image,
    get_question_meta,
)


def _focus_count_white_stones(q_id: str, sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "count_white_stones"
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_random_turn_and_store_image(
        family=family,
        q_id=q_id,
        sim_id=sim_id,
        simulated_game=simulated_game,
    )

    # count white stones (=2) as ground truth
    num_white = int(np.count_nonzero(board == 2))
    answer_text = str(num_white)

    return build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer_text,
    )


def gen_question_q5_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q5 sample:
    focus: "count_white_stones"
    """
    q_id = "Q5"
    question_text = (
        "You are looking at a Gomoku game position. "
        "Black stones belong to player 1, white stones belong to player 2. "
        "Count ONLY the white stones that are already on the board. "
        "How many white stones are there in total? Answer with a single integer."
    )

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text
    return dataset_row


def gen_question_q6_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q6 sample:
    focus: "count_white_stones"
    """
    q_id = "Q6"
    question_text = (
        "Analyze the current Gomoku position. "
        "Player 1 uses black stones, player 2 uses white stones. "
        "Ignore empty intersections and ignore all black stones. "
        "How many stones belonging to player 2 (white) are placed on the board? "
        "Answer with a single integer (for example: 6)."
    )

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text
    return dataset_row


def gen_question_q7_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q7 sample:
    focus: "count_white_stones"
    """
    q_id = "Q7"
    question_text = (
        "This image shows a snapshot of an ongoing Gomoku game. "
        "White stones represent player 2, black stones represent player 1. "
        "Count how many board intersections are occupied by white stones only. "
        "What is the total number of white stones currently on the board? "
        "Respond with exactly one integer and no extra words."
    )

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text
    return dataset_row


def gen_question_q8_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q8 sample:
    focus: "count_white_stones"
    """
    q_id = "Q8"
    question_text = (
        "You are inspecting a partially played Gomoku position. "
        "Player 2 is using white stones, player 1 is using black stones. "
        "Determine how many white stones player 2 has placed on the board so far. "
        "Do not count black stones or empty cells. "
        "Return only the total number of white stones as a single integer value."
    )

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game)
    dataset_row.question = question_text
    return dataset_row