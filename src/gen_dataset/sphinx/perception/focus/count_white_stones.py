import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    persist_turn_game_state,
    get_question_text
)


def _focus_count_white_stones(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    non_rand_img: bool,
    min_turns: int = 0,
    max_turns: int = 999
) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "count_white_stones"
    """
    FOCUS = "count_white_stones"
    FAMILY = QuestionFamily.PERCEPTION

    # Sample random turn
    turn_index = get_random_turn_index(game, min_turns, max_turns)
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id, non_rand_img=non_rand_img)
    # Persist game state for easier debugging
    persist_turn_game_state(board, turn_index, sim_id)

    # count white stones (=2) as ground truth
    num_white = int(np.count_nonzero(board == 2))
    answer = str(num_white)
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


def gen_question_q201_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q201 sample:
    focus: "count_white_stones"
    """
    q_id = "Q201"
    min_turns = 0
    max_turns = 30

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns)
    dataset_row.question = get_question_text(q_id)

    return dataset_row


def gen_question_q202_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q202 sample:
    focus: "count_white_stones"
    """
    q_id = "Q202"
    min_turns = 31
    max_turns = 75

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns)
    dataset_row.question = get_question_text(q_id)

    return dataset_row


def gen_question_q203_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q203 sample:
    focus: "count_white_stones"
    """
    q_id = "Q203"
    min_turns = 76
    max_turns = 150

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns)
    dataset_row.question = get_question_text(q_id)

    return dataset_row


def gen_question_q204_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q204 sample:
    focus: "count_white_stones"
    """
    q_id = "Q204"
    min_turns = 151
    max_turns = 224

    dataset_row = _focus_count_white_stones(q_id, sim_id, simulated_game, non_rand_img, min_turns, max_turns)
    dataset_row.question = get_question_text(q_id)

    return dataset_row
