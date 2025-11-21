import numpy as np

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    get_question_meta,
    QuestionFamily,
    select_fixed_turn_and_store_image,
    build_basic_dataset_row
)
from src.game_logic import get_winner


def _determine_who_won(q_id: str, sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "determine_who_won"
    """
    family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)

    last_turn = simulated_game.shape[0] - 1
    # choose last turn, get board, store image
    board, img_path, img_bytes = select_fixed_turn_and_store_image(
        sim_id=sim_id,
        simulated_game=simulated_game,
        turn_index=last_turn,
    )

    winner = get_winner(board, 5)
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

    return build_basic_dataset_row(
        img_path=img_path,
        img_bytes=img_bytes,
        family=family,
        q_id=q_id,
        focus=focus,
        answer=answer,
        valid_answers=valid_answers,
    )


def gen_question_q21_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q21 sample:
    focus: "determine_who_won"
    """
    q_id = "Q21"

    dataset_row = _determine_who_won(q_id, sim_id, simulated_game)

    question_text = (
        "You are looking at the final position of a completed Gomoku game. "
        "Player 1 uses black stones and Player 2 uses white stones. "
        "Based on this final board, determine the result of the game. "
        "Answer with exactly one word: 'black' if Player 1 won, "
        "'white' if Player 2 won, or 'draw' if neither player won."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q22_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q22 sample:
    focus: "determine_who_won"
    """
    q_id = "Q22"

    dataset_row = _determine_who_won(q_id, sim_id, simulated_game)

    question_text = (
        "This image shows the board at the end of a Gomoku game. "
        "Black stones belong to Player 1 and white stones belong to Player 2. "
        "The game is finished and no more moves will be played. "
        "Who won the game, or was it a draw? "
        "Respond with exactly one word: 'black', 'white', or 'draw'."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q23_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q23 sample:
    focus: "determine_who_won"
    """
    q_id = "Q23"

    dataset_row = _determine_who_won(q_id, sim_id, simulated_game)

    question_text = (
        "You see a finished Gomoku game. "
        "One player has placed black stones, the other player has placed white stones. "
        "Using only the final arrangement of black and white stones on the board, "
        "determine the outcome of the game. "
        "Answer with exactly one word: 'black', 'white', or 'draw'."
    )

    dataset_row.question = question_text
    return dataset_row


def gen_question_q24_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q24 sample:
    focus: "determine_who_won"
    """
    q_id = "Q24"

    dataset_row = _determine_who_won(q_id, sim_id, simulated_game)

    question_text = (
        "This board position represents the final state of a Gomoku game. "
        "Black stones and white stones have already been placed; the game is over. "
        "From the pattern of stones, decide who has won the game, or if it is a draw. "
        "Reply with exactly one word: 'black', 'white', or 'draw'."
    )

    dataset_row.question = question_text
    return dataset_row