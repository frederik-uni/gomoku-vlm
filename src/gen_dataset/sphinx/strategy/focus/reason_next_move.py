import random

import numpy as np

import game_logic
from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    get_question_text
)
from gen_dataset.sphinx.perception.focus.print_board_matrix import board_to_matrix_string


def _focus_reason_next_move(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    non_rand_img: bool,
    min_turns: int = 0,
    max_turns: int = 999
) -> tuple[int, str, bool, bool, str, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "reason_next_move"

    Returns:
        tuple[int, str, bool, bool, str, DatasetRow]: A tuple containing:
            - int: The player as int (1 if black, 2 if white)
            - str: The color of the player.
            - bool: Can you win next turn?
            - bool: Can you lose next turn?
            - str: Next best move
            - DatasetRow: The pre-constructed dataset row for the dataset.
    """
    FOCUS = "reason_next_move"
    FAMILY = QuestionFamily.STRATEGY

    # Answer should look like this:
    # - Print the board matrix
    # - Can you win next turn?
    # - Can you lose next turn?
    # - What's the next best move

    # General:
    # Sanity check
    num_turns = game.shape[0]
    if num_turns < 2:
        # Impossible to compare two states, if less than two turns have been performed
        raise ValueError(f"Need at least 2 turns for 'focus: reason_next_move', got num_turns={num_turns}. "
                         f"Miss configured focus, make sure the min_turns is at least 2.")

    # turn_index is a random, 0-based index of the "before" board.
    turn_index = get_random_turn_index(game, min_turns, max_turns)

    # make sure this turn is not the last turn of the game
    # The largest valid "before" index (- 1 because 0-indexed, - 1 because another has to exist after, therefore = - 2)
    max_before_index = num_turns - 2
    if turn_index > max_before_index:
        turn_index = max_before_index

    next_turn = turn_index + 1

    # get color of the player who must perform this turn
    if next_turn % 2 == 0:
        player = 1
        color = "black"
        opposite_player = 2
    else:
        player = 2
        color = "white"
        opposite_player = 1

    # get board before performing the turn
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id, non_rand_img=non_rand_img)

    # also get the next board (to determine the actually performed move by the bot)
    board_after = game[next_turn]
    # persist the image for debugging only, not for the dataset
    persist_turn_image(board_after, next_turn, sim_id, non_rand_img=True)

    # Part 1) Print the board matrix
    board_as_matrix_string = board_to_matrix_string(board)

    # Part 2) Can you win
    if game_logic.count(board, 5, player, True) > 0:
        can_win = True
    else:
        can_win = False

    # Part 3) Can you lose
    if game_logic.count(board, 5, opposite_player, True) > 0:
        can_lose = True
    else:
        can_lose = False

    # Part 4) Best next move
    # compare boards to get target (row, col)
    mask = board != board_after  # shape (15, 15)
    changed_positions = np.argwhere(mask)  # shape (N, 2)
    num_changes = changed_positions.shape[0]
    if num_changes != 1:
        raise ValueError(
            f"Expected exactly 1 changed cell, but found {num_changes}. "
        )

    # (row, col) in 0-based indexing
    row_idx, col_idx = changed_positions[0]

    # Build the answer as a string.
    # space-separated "row col" in 0-based indices.
    best_next_move = f"{row_idx} {col_idx}"

    # Ground-truth answer
    answer = (
        f"{board_as_matrix_string}\n"
        f"CAN_WIN_NEXT_TURN: {'yes' if can_win else 'no'}\n"
        f"CAN_LOSE_NEXT_TURN: {'yes' if can_lose else 'no'}\n"
        f"BEST_NEXT_MOVE: {best_next_move}"
    )

    return player, color, can_win, can_lose, best_next_move, DatasetRow(
        img_path=str(img_path),
        img_bytes=img_bytes,

        family=FAMILY,
        q_id=q_id,
        focus=FOCUS,

        answer=answer,
        valid_answers=[answer],

        # Will be assigned later in the creation process
        question=None,
        split=None
    )


def gen_question_q1301_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1301 sample:
    focus: "reason_next_move"
    """
    q_id = "Q1301"

    player, color, can_win, can_lose, next_move, row = _focus_reason_next_move(q_id, sim_id, simulated_game, non_rand_img)

    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)

    return row

def gen_question_q1302_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1302 sample:
    focus: "reason_next_move"
    """
    q_id = "Q1302"

    player, color, can_win, can_lose, next_move, row = _focus_reason_next_move(q_id, sim_id, simulated_game, non_rand_img)

    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)

    return row

def gen_question_q1303_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1303 sample:
    focus: "reason_next_move"
    """
    q_id = "Q1303"

    player, color, can_win, can_lose, next_move, row = _focus_reason_next_move(q_id, sim_id, simulated_game, non_rand_img)

    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)

    return row

def gen_question_q1304_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1304 sample:
    focus: "reason_next_move"
    """
    q_id = "Q1304"

    player, color, can_win, can_lose, next_move, row = _focus_reason_next_move(q_id, sim_id, simulated_game, non_rand_img)

    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)

    return row
