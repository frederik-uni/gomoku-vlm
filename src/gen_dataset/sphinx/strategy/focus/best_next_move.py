import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    get_random_turn_index,
    persist_turn_image,
    get_question_text
)


def _focus_best_next_move(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    min_turns: int = 0,
    max_turns: int = 999
) -> tuple[str, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "best_next_move"

    Returns:
        tuple[str, DatasetRow]: A tuple containing:
            - str: The color of the player.
            - DatasetRow: The pre-constructed dataset row for the dataset.
    """
    FOCUS = "best_next_move"
    FAMILY = QuestionFamily.STRATEGY

    num_turns = game.shape[0]
    if num_turns < 2:
        # Impossible to compare two states, if less than two turns have been performed
        raise ValueError(f"Need at least 2 turns for 'focus: best_next_move', got num_turns={num_turns}. "
                         f"Miss configured sphinx_config file, make sure the min_turns is at least 2.")

    # Random 0-based index of the "before" board.
    turn_index = get_random_turn_index(game, min_turns, max_turns)

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

    # get board before performing the turn
    board = game[turn_index]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, turn_index, sim_id)

    # also get the next board (to determine the actually performed move by the bot)
    board_after = game[next_turn]
    # persist the image for debugging only, not for the dataset
    persist_turn_image(board_after, turn_index, sim_id)

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
    answer = f"{row_idx} {col_idx}"
    # Different answer formats
    valid_answers = [
        answer,  # "r c"
        f"{row_idx},{col_idx}",  # "r,c"
        f"({row_idx}, {col_idx})"  # "(r, c)"
    ]

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


def gen_question_q10100_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10100 sample:
    focus: "best_next_move"
    """
    q_id = "Q10100"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q10101_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10101 sample:
    focus: "best_next_move"
    """
    q_id = "Q10101"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q10102_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10102 sample:
    focus: "best_next_move"
    """
    q_id = "Q10102"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row


def gen_question_q10103_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q10103 sample:
    focus: "best_next_move"
    """
    q_id = "Q10103"

    color, dataset_row = _focus_best_next_move(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(color=color)

    return dataset_row
