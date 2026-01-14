import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily,
    persist_turn_image,
    get_question_text
)
from src import game_logic


def _focus_win_next_turn(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    non_rand_img: bool,
    min_turns: int = 999,
    max_turns: int = 999
) -> tuple[int, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "win_next_turn"

    Returns:
        tuple[int, DatasetRow]: A tuple containing:
            - int: The winner of that game.
            - DatasetRow: The pre-constructed dataset row for the dataset.
    """
    FOCUS = "win_next_turn"
    FAMILY = QuestionFamily.STRATEGY

    # get last turn and second to last turn
    last_turn = game.shape[0] - 1
    second_to_last_turn = game.shape[0] - 2
    winner = game_logic.get_winner(game[last_turn], 5)
    if winner == 0:
        raise ValueError(
            f"The game did not end. winner={winner}."
        )

    # get board for second to last turn
    board = game[second_to_last_turn]
    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, last_turn, sim_id, non_rand_img=non_rand_img)

    # also get the last board, to compare against
    board_after = game[last_turn]

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

    return winner, DatasetRow(
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



def _build_outcome_context(winner: int, game: np.ndarray) -> dict[str, str]:
    """
    Compute the context used to format question texts for win_next_turn.
    Provides:
      - outcome_phrase
      - player_color
      - goal_phrase
    """
    last_turn = game.shape[0] - 1

    if winner == 1:
        # Black wins
        return {
            "outcome_phrase": "win for black",
            "player_color": "black",
            "color": "black",
            "goal_phrase": "win the game",
        }
    elif winner == 2:
        # White wins
        return {
            "outcome_phrase": "win for white",
            "player_color": "white",
            "color": "white",
            "goal_phrase": "win the game",
        }
    elif winner == -1:
        # Draw; work out whose move it was on the final move
        player_color = "black" if last_turn % 2 == 0 else "white"
        return {
            "outcome_phrase": "draw",
            "player_color": player_color,
            "color": player_color,
            "goal_phrase": "bring the game to a draw",
        }
    else:
        raise ValueError(f"Unexpected winner value: {winner}")


def gen_question_q1101_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1101 sample:
    focus: "win_next_turn"
    """
    q_id = "Q1101"
    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game, non_rand_img)
    context = _build_outcome_context(winner, simulated_game)

    template = get_question_text(q_id)  # from sphinx_questions.toml
    dataset_row.question = template.format(**context)

    return dataset_row


def gen_question_q1102_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1102 sample:
    focus: "win_next_turn"
    """
    q_id = "Q1102"
    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game, non_rand_img)
    context = _build_outcome_context(winner, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(**context)

    return dataset_row


def gen_question_q1103_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1103 sample:
    focus: "win_next_turn"
    """
    q_id = "Q1103"
    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game, non_rand_img)
    context = _build_outcome_context(winner, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(**context)

    return dataset_row


def gen_question_q1104_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    """
    Generate a single Q1104 sample:
    focus: "win_next_turn"
    """
    q_id = "Q1104"
    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game, non_rand_img)
    context = _build_outcome_context(winner, simulated_game)

    template = get_question_text(q_id)
    dataset_row.question = template.format(**context)

    return dataset_row
