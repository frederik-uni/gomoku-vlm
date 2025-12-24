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

def _focus_list_valid_moves(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    min_turns: int = 0,
    max_turns: int = 999
) -> tuple[str, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "list_valid_moves".

    It picks a random, not-final turn,
    and then returns all empty board positions as the answer.

    Answer format (example):
        "0 0, 0 3, 1 5, 2 2"

    Returns:
    tuple[str, DatasetRow]: A tuple containing:
        - str: The color of the player who has to perform this move.
        - DatasetRow: The pre-constructed dataset row for the dataset.
    """
    FOCUS = "list_valid_moves"
    FAMILY = QuestionFamily.STRATEGY

    num_turns = game.shape[0]
    if num_turns < 1:
        # miss configured question, game had no turns.
        raise ValueError(f"Need at least 1 turn for 'focus: list_valid_moves', got num_turns={num_turns}. ")

    # Random 0-based index of the board.
    turn_index = get_random_turn_index(game, min_turns, max_turns)

    # Make sure the game is not already finished at this position.
    board_at_turn = game[turn_index]
    winner = game_logic.get_winner(board_at_turn, 5)
    if winner != 0:
        # If the game already ended at this turn:
        if turn_index == 0:
            # No earlier state exists -> we can't ask for "valid next move".
            raise ValueError(
                "Game already ended at the first recorded state. "
                "Cannot generate 'perform_valid_move' sample."
            )
        turn_index -= 1

    # get board for this turn_index
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id)

    # Collect all valid moves = all empty cells (0's)
    mask = board == 0  # shape (15, 15)
    valid_moves = np.argwhere(mask)  # shape (N, 2), each row = [row_idx, col_idx]
    num_valid_moves = valid_moves.shape[0]
    if num_valid_moves < 1:
        raise ValueError("Expected at least one valid move, but found none (board has no empty cells).")

    # Build a single answer string like: "r0 c0, r0 c3, r1 c2, ..."
    move_strings = [f"{row_idx} {col_idx}" for row_idx, col_idx in valid_moves]
    answer = ", ".join(move_strings)

    # only one correct answer
    valid_answers = [answer]

    # get color of the player who must perform this turn
    if (turn_index + 1) % 2 == 0:
        color = "black"
    else:
        color = "white"

    return color, DatasetRow(
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


def gen_question_q10200_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10200 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q10200"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q10201_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10201 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q10201"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q10202_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10202 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q10202"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q10203_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10203 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q10203"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row
