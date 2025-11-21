import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    get_question_meta,
    QuestionFamily,
    select_fixed_turn_and_store_image,
    random_turn_index,
    build_basic_dataset_row
)


def _focus_best_next_move(q_id: str, sim_id: int, simulated_game: np.ndarray) -> tuple[str, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "best_next_move"

    Returns:
        tuple[str, DatasetRow]: A tuple containing:
            - str: The color of the player.
            - DatasetRow: The pre-constructed dataset row for the dataset.
    """
    family, focus = get_question_meta(QuestionFamily.STRATEGY, q_id)

    num_turns = simulated_game.shape[0]
    if num_turns < 2:
        # Impossible to compare two states, if less than two turns have been performed
        raise ValueError(f"Need at least 2 turns for 'focus: best_next_move', got num_turns={num_turns}. "
                         f"Miss configured sphinx_config file, make sure the min_turns is at least 2.")

    # Random 0-based index of the "before" board.
    turn_index = random_turn_index(QuestionFamily.STRATEGY, q_id, simulated_game)

    # make sure this turn is not the last turn of the game
    # The largest valid "before" index (- 1 because 0-indexed, - 1 because another has to exist after, therefore = - 2)
    max_before_index = num_turns - 2
    if turn_index > max_before_index:
        turn_index = max_before_index

    next_turn = turn_index + 1

    # get color of the player who must perform this turn
    if next_turn % 2 == 0:
        color = "black"
    else:
        color = "white"

    # get board before performing the turn, store image
    board_before, img_path, img_bytes = select_fixed_turn_and_store_image(
        sim_id=sim_id,
        simulated_game=simulated_game,
        turn_index=turn_index,
    )

    # also get the next board (to determine the actually performed move by the bot)
    # (save the image just for debugging, not for the dataset)
    board_after, _, _ = select_fixed_turn_and_store_image(
        sim_id=sim_id,
        simulated_game=simulated_game,
        turn_index=next_turn,
    )

    # compare boards to get target (row, col)
    mask = board_before != board_after  # shape (15, 15)
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
    answer = f"{row_idx} {col_idx}"
    # Different answer formats
    valid_answers = [
        answer,  # "r c"
        f"{row_idx},{col_idx}",  # "r,c"
        f"({row_idx}, {col_idx})"  # "(r, c)"
    ]
    return color, build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer,
        valid_answers=valid_answers,
    )


def gen_question_q600_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q600 sample:
    focus: "best_next_move"
    """
    q_id = "Q600"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    question_text = (
        "You are looking at a Gomoku board position in the middle of a game. "
        "Player 1 uses black stones and Player 2 uses white stones. "
        f"It is {color}'s turn to move. "
        f"You are {color}, what is the best next move for you to perform? "
        "Answer with two 0-based integers in the format 'row col', and nothing else."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q601_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q601 sample:
    focus: "best_next_move"
    """
    q_id = "Q601"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    question_text = (
        "This image shows a Gomoku position in the middle of a game. "
        "Player 1 places black stones and Player 2 places white stones. "
        f"It is {color}'s turn to move in this position. "
        "Select the best move for the current player. "
        "Give your answer as two 0-based integers in the format 'row col', and nothing else."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q602_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q602 sample:
    focus: "best_next_move"
    """
    q_id = "Q602"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    question_text = (
        "You are analyzing an ongoing Gomoku game. "
        f"You are playing as {color} in the position shown. "
        "Assume standard Gomoku rules on a 15x15 board. "
        "Choose the strongest available move for your side in this position. "
        "Answer with two 0-based coordinates in the form 'row col' (no extra text)."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q603_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q603 sample:
    focus: "best_next_move"
    """
    q_id = "Q603"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    question_text = (
        "Consider the displayed Gomoku board state. "
        f"It is {color}'s turn to move. "
        "Your goal is to play the move that gives your side the best continuation "
        "according to the position on the board. "
        "Respond with exactly two 0-based integers 'row col' indicating your chosen move."
    )

    dataset_row.question = question_text
    return dataset_row
