import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    get_question_meta,
    QuestionFamily,
    select_fixed_turn_and_store_image,
    build_basic_dataset_row
)
from src.game_logic import get_winner


def _focus_win_next_turn(q_id: str, sim_id: int, simulated_game: np.ndarray) -> tuple[int, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "win_next_turn"

    Returns:
        tuple[int, DatasetRow]: A tuple containing:
            - int: The winner of that game.
            - DatasetRow: The pre-constructed dataset row for the dataset.
    """
    family, focus = get_question_meta(QuestionFamily.STRATEGY, q_id)

    last_turn = simulated_game.shape[0] - 1
    second_to_last_turn = simulated_game.shape[0] - 2
    winner = get_winner(simulated_game[last_turn], 5)
    if winner == 0:
        raise ValueError(
            f"The game did not end. winner={winner}."
        )

    # choose second to last turn, get board, store image for training
    board_before, img_path, img_bytes = select_fixed_turn_and_store_image(
        sim_id=sim_id,
        simulated_game=simulated_game,
        turn_index=second_to_last_turn,
    )
    # also get the last board, to compare against
    board_after = simulated_game[last_turn]

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
    return winner, build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer,
        valid_answers=valid_answers,
    )


def gen_question_q500_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q500 sample:
    focus: "win_next_turn"
    """
    q_id = "Q500"

    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game)
    match winner:
        case 1:
            outcome_phrase = "win for black"
            player_color = "black"
            goal_phrase = "win the game"
        case 2:
            outcome_phrase = "win for white"
            player_color = "white"
            goal_phrase = "win the game"
        case -1:
            outcome_phrase = "draw"
            last_turn = simulated_game.shape[0] - 1
            if last_turn % 2 == 0:
                player_color = "black"
            else:
                player_color = "white"
            goal_phrase = "bring the game to a draw"
        case _:
            raise ValueError(f"Unexpected winner, game may still be in progress.")

    question_text = (
        "You are looking at the Gomoku board position just before the final move of the game. "
        f"The game ended in a {outcome_phrase}. "
        f"You are currently playing as {player_color}. "
        f"What is your next, best move to {goal_phrase}? "
         "Answer with two 0-based integers in the format 'row col', and nothing else."
    )
    dataset_row.question = question_text

    return dataset_row


def gen_question_q501_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q501 sample:
    focus: "win_next_turn"
    """
    q_id = "Q501"

    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game)
    last_turn = simulated_game.shape[0] - 1

    match winner:
        case 1:
            outcome_phrase = "victory for the black player"
            player_color = "black"
            goal_phrase = "secure the win"
        case 2:
            outcome_phrase = "victory for the white player"
            player_color = "white"
            goal_phrase = "secure the win"
        case -1:
            outcome_phrase = "draw between both players"
            if last_turn % 2 == 0:
                player_color = "black"
            else:
                player_color = "white"
            goal_phrase = "force the game into a draw"
        case _:
            raise ValueError(f"Unexpected winner value: {winner}")

    question_text = (
        "You are viewing the Gomoku position just before the decisive last move. "
        f"In the final result, the game was a {outcome_phrase}. "
        f"Now imagine you are playing as {player_color} and it is your turn. "
        f"What move should you play to {goal_phrase}? "
        "Answer with two 0-based integers in the format 'row col' and nothing else."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q502_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q502 sample:
    focus: "win_next_turn"
    """
    q_id = "Q502"

    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game)
    last_turn = simulated_game.shape[0] - 1

    match winner:
        case 1:
            outcome_phrase = "won by the black player"
            player_color = "black"
            goal_phrase = "achieve this winning outcome"
        case 2:
            outcome_phrase = "won by the white player"
            player_color = "white"
            goal_phrase = "achieve this winning outcome"
        case -1:
            outcome_phrase = "ended in a draw"
            if last_turn % 2 == 0:
                player_color = "black"
            else:
                player_color = "white"
            goal_phrase = "reach this drawn result"
        case _:
            raise ValueError(f"Unexpected winner value: {winner}")

    question_text = (
        "This board shows a Gomoku position one move before the game ended. "
        f"The final game result was {outcome_phrase}. "
        f"Assume you are the player using {player_color} stones and it is your move. "
        f"Which single move must you play now to {goal_phrase}? "
        "Respond with exactly two 0-based coordinates in the form 'row col'."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q503_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q503 sample:
    focus: "win_next_turn"
    """
    q_id = "Q503"

    winner, dataset_row = _focus_win_next_turn(q_id, sim_id, simulated_game)
    last_turn = simulated_game.shape[0] - 1

    match winner:
        case 1:
            outcome_phrase = "a winning position for black"
            player_color = "black"
            goal_phrase = "convert this position into a win"
        case 2:
            outcome_phrase = "a winning position for white"
            player_color = "white"
            goal_phrase = "convert this position into a win"
        case -1:
            outcome_phrase = "a drawn final position"
            if last_turn % 2 == 0:
                player_color = "black"
            else:
                player_color = "white"
            goal_phrase = "force the final result to be a draw"
        case _:
            raise ValueError(f"Unexpected winner value: {winner}")

    question_text = (
        "You are given a Gomoku board state right before the last move of the game. "
        f"The final outcome of the game was {outcome_phrase}. "
        f"You are playing as {player_color}, and it is your turn in this position. "
        f"Choose the one move that will {goal_phrase}. "
        "Output your answer as two 0-based integers 'row col' with no extra text."
    )

    dataset_row.question = question_text
    return dataset_row
