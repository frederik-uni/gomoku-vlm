import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    build_basic_dataset_row,
    select_random_turn_and_store_image,
    get_question_meta,
)


def _focus_count_empty_intersections(q_id: str,sim_id: int,simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper for any question with the
    focus: "count_empty_intersections".
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_random_turn_and_store_image(
        family=family,
        q_id=q_id,
        sim_id=sim_id,
        simulated_game=simulated_game,
    )

    # empty cells are encoded as 0
    num_empty = int(np.count_nonzero(board == 0))
    answer_text = str(num_empty)

    return build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer_text,
    )


def gen_question_q9_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q9 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q9"
    question_text = (
        "You are looking at a Gomoku game position. "
        "Black stones belong to player 1 and white stones belong to player 2. "
        "Some intersections already contain stones, others are still empty. "
        "Count ONLY the empty intersections where no stone has been placed yet. "
        "How many empty intersections are there in total? "
        "Answer with a single integer."
    )

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = question_text
    return row


def gen_question_q10_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q10"
    question_text = (
        "Consider this snapshot of a Gomoku board. "
        "Player 1 uses black stones, player 2 uses white stones. "
        "Ignore all intersections that already contain a stone. "
        "Count how many intersections are still completely empty and "
        "available for future moves. "
        "Return the number of empty intersections as a single integer."
    )

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = question_text
    return row


def gen_question_q11_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q11 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q11"
    question_text = (
        "This image shows a Gomoku game in progress. "
        "Some grid intersections are occupied by black or white stones, "
        "while others remain empty. "
        "Your task is to determine how many intersections are still empty. "
        "Do NOT count any intersection that contains a stone. "
        "Provide only the total number of empty intersections as a single integer."
    )

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = question_text
    return row


def gen_question_q12_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q12 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q12"
    question_text = (
        "You are inspecting a partially filled Gomoku board. "
        "Black stones (player 1) and white stones (player 2) occupy some intersections, "
        "but many intersections are still empty. "
        "Focus ONLY on the intersections that contain no stone at all. "
        "How many empty intersections are on the board at this moment? "
        "Answer with a single integer and no additional text."
    )

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = question_text
    return row
