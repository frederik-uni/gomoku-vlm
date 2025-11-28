import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    persist_turn_game_state,
    get_question_text
)
from src import game_logic


def _focus_determine_who_won(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    min_turns: int = 0,
    max_turns: int = 999
) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "determine_who_won"
    """
    FOCUS = "determine_who_won"
    FAMILY = QuestionFamily.PERCEPTION

    # choose last turn, get board
    last_turn = game.shape[0] - 1
    board = game[last_turn]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, last_turn, sim_id)

    # Determine winner for ground truth answer
    winner = game_logic.get_winner(board, 5)
    match winner:
        case 1:
            answer = "black"
            valid_answers = ["black", "Player 1", "black won", "Player 1 won"]
        case 2:
            answer = "white"
            valid_answers = ["white", "Player 2", "white won", "Player 2 won"]
        case -1:
            answer = "draw"
            valid_answers = ["draw", "no winner", "tie"]
        case _:
            raise ValueError(f"Unexpected winner, game may still be in progress.")

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

def gen_question_q600_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q600 sample:
    focus: "determine_who_won"
    """
    q_id = "Q600"
    row = _focus_determine_who_won(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row


def gen_question_q601_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q601 sample:
    focus: "determine_who_won"
    """
    q_id = "Q601"
    row = _focus_determine_who_won(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row


def gen_question_q602_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q602 sample:
    focus: "determine_who_won"
    """
    q_id = "Q602"
    row = _focus_determine_who_won(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row


def gen_question_q603_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q603 sample:
    focus: "determine_who_won"
    """
    q_id = "Q603"
    row = _focus_determine_who_won(q_id, sim_id, simulated_game)
    row.question = get_question_text(q_id)
    return row
