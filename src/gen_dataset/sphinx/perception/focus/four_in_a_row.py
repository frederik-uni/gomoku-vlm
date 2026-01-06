import random

import numpy as np

import game_logic
from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily, persist_turn_image, get_random_turn_index, get_question_text
)


def _focus_four_in_a_row(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    non_rand_img: bool,
    *,
    min_turn = 0,
    max_turn = 999,
    player = -1
) -> tuple[int, str, int, DatasetRow]:
    """
    Helper function for any question that has the focus: "four_in_a_row_segments".

    Definition:
        A "four-in-a-row segment" is any contiguous window of length 4
        (horizontal, vertical, or diagonal) where all 4 cells are occupied
        by the given player's stones.

    Params:
        q_id: The q_id of the question.
        sim_id: The sim_id of the simulation.
        game: Sequence of board states (game[turn_idx] is a 15×15 board).
        min_turn: Minimum turn index to sample from.
        max_turn: Maximum turn index to sample from.
        player: 1 (black) or 2 (white). If -1, choose randomly.

    Returns:
        tuple[int, str, int, DatasetRow]: A tuple containing:
            - player: 1 if black, 2 if white.
            - color: "black" or "white".
            - num_four_in_a_row: Count of length-4 segments for player.
            - row: Pre-constructed DatasetRow (question still None).
    """
    FOCUS = "four_in_a_row"
    FAMILY = QuestionFamily.PERCEPTION

    if player == -1:
        player = 1 if random.random() < 0.5 else 2
    color = "black" if player == 1 else "white"

    idx = get_random_turn_index(game, min_turn, max_turn)
    board = game[idx]
    num_four_in_a_row = game_logic.count(board, 4, player, almost=False)
    answer = str(num_four_in_a_row)

    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(board, idx, sim_id, non_rand_img=non_rand_img)

    return player, color, num_four_in_a_row, DatasetRow(
        img_path=str(img_path),
        img_bytes=img_bytes,
        family=FAMILY,
        q_id=q_id,
        focus=FOCUS,
        answer=answer,
        valid_answers=[answer],
        question=None,
        split=None,
    )


def gen_question_q501_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    q_id = "Q501"
    min_turn = 0
    max_turn = 55

    player, color, _count, row = _focus_four_in_a_row(
        q_id, sim_id, simulated_game, non_rand_img, min_turn=min_turn, max_turn=max_turn
    )
    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)
    return row


def gen_question_q502_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    q_id = "Q502"
    min_turn = 55
    max_turn = 110

    player, color, _count, row = _focus_four_in_a_row(
        q_id, sim_id, simulated_game, non_rand_img, min_turn=min_turn, max_turn=max_turn
    )
    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)
    return row


def gen_question_q503_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    q_id = "Q503"
    min_turn = 110
    max_turn = 165

    player, color, _count, row = _focus_four_in_a_row(
        q_id, sim_id, simulated_game, non_rand_img, min_turn=min_turn, max_turn=max_turn
    )
    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)
    return row


def gen_question_q504_sample(sim_id: int, simulated_game: np.ndarray, non_rand_img: bool) -> DatasetRow:
    q_id = "Q504"
    min_turn = 165
    max_turn = 225

    player, color, _count, row = _focus_four_in_a_row(
        q_id, sim_id, simulated_game, non_rand_img, min_turn=min_turn, max_turn=max_turn
    )
    template = get_question_text(q_id)
    row.question = template.format(player=player, color=color)
    return row
