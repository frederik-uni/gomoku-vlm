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
    non_rand_img: bool,
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
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id, non_rand_img=non_rand_img)

    # All valid moves are all empty cells (0)
    mask = board == 0
    valid_moves = np.argwhere(mask)  # [[row, col], ...]
    if valid_moves.size == 0:
        raise ValueError("Expected at least one valid move, but found none (board full).")

    # Row-major order
    valid_moves = valid_moves[np.lexsort((valid_moves[:, 1], valid_moves[:, 0]))]

    # Atomic answers: each move is its own answer unit
    valid_answers = [f"{r} {c}" for r, c in valid_moves]

    # Pick one valid move at random as the canonical "answer" for this sample.
    # valid_answers remains the full set for evaluation.
    answer = valid_answers[np.random.randint(len(valid_answers))]

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


def gen_question_q1001_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1001 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q1001"
    min_turn = 0
    max_turn = 30

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game, non_rand_img, min_turn, max_turn)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q1002_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1002 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q1002"
    min_turn = 31
    max_turn = 75

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game, non_rand_img, min_turn, max_turn)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q1003_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1003 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q1003"
    min_turn = 76
    max_turn = 150

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game, non_rand_img, min_turn, max_turn)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q1004_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1004 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q1004"
    min_turn = 151
    max_turn = 224

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game, non_rand_img, min_turn, max_turn)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row
