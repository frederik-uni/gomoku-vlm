import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    get_question_meta,
    QuestionFamily,
    select_fixed_turn_and_store_image,
    random_turn_index,
    build_basic_dataset_row
)
from src.game_logic import get_winner


def _focus_list_valid_moves(q_id: str, sim_id: int, simulated_game: np.ndarray) -> tuple[str, DatasetRow]:
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
    family, focus = get_question_meta(QuestionFamily.STRATEGY, q_id)

    num_turns = simulated_game.shape[0]
    if num_turns < 1:
        # miss configured config, game had no turns.
        raise ValueError(f"Need at least 1 turn for 'focus: perform_valid_move', got num_turns={num_turns}. "
                         f"Miss configured sphinx_config file, make sure the min_turns is at least 1.")

    # Random 0-based index of the board.
    turn_index = random_turn_index(QuestionFamily.STRATEGY, q_id, simulated_game)

    # Make sure the game is not already finished at this position.
    board_at_turn = simulated_game[turn_index]
    winner = get_winner(board_at_turn, 5)
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
    board, img_path, img_bytes = select_fixed_turn_and_store_image(
        sim_id=sim_id,
        simulated_game=simulated_game,
        turn_index=turn_index,
    )

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

    return color, build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer,
        valid_answers=valid_answers,
    )


def gen_question_q700_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q700 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q700"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    question_text = (
        "You are looking at a Gomoku board position in the middle of a game. "
        "Player 1 uses black stones and Player 2 uses white stones. "
        f"It is {color}'s turn to move. "
        "List ALL currently valid moves, i.e. all empty intersections where a stone can still be placed. "
        "Output them as a comma-separated list of 0-based coordinates in the format "
        "'row col, row col, row col, ...' and nothing else. "
        "For example: '0 0, 0 3, 1 5'."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q701_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q701 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q701"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    question_text = (
        "This image shows a Gomoku position partway through a game. "
        "Stones of one player are black, and stones of the other player are white. "
        f"It is {color}'s turn to move. "
        "Identify every empty board intersection where a stone could legally be played next. "
        "Output ALL such moves as a comma-separated list of 0-based coordinates in the format "
        "'row col, row col, row col, ...' with no extra text."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q702_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q702 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q702"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    question_text = (
        "You are given a snapshot of an ongoing Gomoku game. "
        "Player 1 places black stones and Player 2 places white stones. "
        f"In this position, it is {color}'s turn to play. "
        "Your task is to list every legal move, i.e. every empty intersection where a new stone "
        "can still be placed. "
        "Return your answer as a comma-separated list of 0-based 'row col' pairs, for example: "
        "'0 4, 1 3, 7 7'. Do not add any additional explanation."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q703_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q703 sample:
    focus: "list_valid_moves"
    """
    q_id = "Q703"

    color, dataset_row = _focus_list_valid_moves(q_id, sim_id, simulated_game)

    question_text = (
        "Consider the shown Gomoku board position. "
        f"It is {color}'s turn, and you must decide where a stone could legally be placed. "
        "Treat every currently empty intersection as a possible move, and every occupied "
        "intersection as invalid. "
        "List ALL valid moves as a comma-separated list of coordinates in 0-based '(row col)' "
        "format, written as 'row col, row col, row col, ...'. "
        "Only output this list of coordinates, nothing else."
    )

    dataset_row.question = question_text
    return dataset_row
