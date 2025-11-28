import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    persist_turn_game_state,
    get_question_text
)

def _focus_count_black_stones(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    min_turns: int = 0,
    max_turns: int = 999
) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "count_black_stones"
    """
    FOCUS = "count_black_stones"
    FAMILY = QuestionFamily.PERCEPTION

    # Sample random turn
    turn_index = get_random_turn_index(game, min_turns, max_turns)
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id)
    # Persist game state for easier debugging
    persist_turn_game_state(board, turn_index, sim_id)

    # count black stones (=1) as ground truth
    num_black = int(np.count_nonzero(board == 1))
    answer = str(num_black)
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


def gen_question_q100_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q100 sample:
    focus: "count_black_stones"
    """
    q_id = "Q100"

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)

    return dataset_row


def gen_question_q101_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q101 sample:
    focus: "count_black_stones"
    """
    q_id = "Q101"

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)

    return dataset_row


def gen_question_q102_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q102 sample:
    focus: "count_black_stones"
    """
    q_id = "Q102"

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)

    return dataset_row


def gen_question_q103_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q103 sample:
    focus: "count_black_stones"
    """
    q_id = "Q103"

    dataset_row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    dataset_row.question = get_question_text(q_id)

    return dataset_row
