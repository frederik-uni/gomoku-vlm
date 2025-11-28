import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    persist_turn_game_state,
    get_question_text
)


def _focus_count_empty_intersections(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    min_turns: int = 0,
    max_turns: int = 999
) -> DatasetRow:
    """
    Helper for any question with the
    focus: "count_empty_intersections".
    """
    FOCUS = "count_empty_intersections"
    FAMILY = QuestionFamily.PERCEPTION

    # Sample random turn
    turn_index = get_random_turn_index(game, min_turns, max_turns)
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id)
    # Persist game state for easier debugging
    persist_turn_game_state(board, turn_index, sim_id)

    # empty cells are encoded as 0
    num_empty = int(np.count_nonzero(board == 0))
    answer = str(num_empty)
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


def gen_question_q300_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q300 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q300"

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row


def gen_question_q301_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q301 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q301"

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row


def gen_question_q302_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q302 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q302"

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row


def gen_question_q303_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q303 sample:
    focus: "count_empty_intersections"
    """
    q_id = "Q303"

    row = _focus_count_empty_intersections(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row
