import random

import numpy as np

import game_logic
from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily, persist_turn_image, get_question_text, get_random_turn_index
)


def _focus_can_you_lose(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    *,
    max_resamples: int = 2000,
    late_game_bias_prob: float = 0.1,
    late_game_window: int = 24,
) -> tuple[int, str, bool, DatasetRow]:
    """
    focus: "can_you_lose"

    Label:
      can_lose == True  iff the opponent has an immediate winning move available
                         on their next turn (i.e., there exists a 4+1 threat now).

    Sampling:
      Always sample across the entire valid range:
        idx in [0, num_turns-2] (so idx+1 exists).
      Enforce 50/50 by choosing a target label and resampling until it matches.
    """
    FOCUS = "can_you_lose"
    FAMILY = QuestionFamily.PERCEPTION

    num_turns = game.shape[0]
    if num_turns < 2:
        raise ValueError(f"Need at least 2 turns for can_you_lose, got num_turns={num_turns}")

    lo = 0
    hi = num_turns - 2  # last valid "before board" index (must have a next turn)

    target_can_lose = (random.random() < 0.5)

    chosen_idx: int | None = None
    chosen_player: int | None = None
    chosen_board: np.ndarray | None = None
    chosen_can_lose: bool | None = None

    for i in range(max_resamples):
        # optional late-game bias to find more threat positions,
        # but still allow full-range sampling.
        if random.random() < late_game_bias_prob:
            late_lo = max(lo, hi - late_game_window)
            idx = random.randint(late_lo, hi)
        else:
            idx = get_random_turn_index(game, lo, hi)

        board = game[idx]

        next_turn_idx = idx + 1
        player = 1 if (next_turn_idx % 2 == 0) else 2
        opponent = 2 if player == 1 else 1

        can_lose = (game_logic.count(board, 5, opponent, almost=True) > 0)

        if can_lose == target_can_lose:
            chosen_idx = idx
            chosen_player = player
            chosen_board = board
            chosen_can_lose = can_lose
            break

    if chosen_idx is None:
        # Fallback: take the last possible before-board index
        idx = hi
        board = game[idx]

        next_turn_idx = idx + 1
        player = 1 if (next_turn_idx % 2 == 0) else 2
        opponent = 2 if player == 1 else 1

        can_lose = (game_logic.count(board, 5, opponent, almost=True) > 0)

        chosen_idx, chosen_player, chosen_board, chosen_can_lose = idx, player, board, can_lose

    color = "black" if chosen_player == 1 else "white"
    img_path, img_bytes = persist_turn_image(chosen_board, chosen_idx, sim_id)

    answer = "yes" if chosen_can_lose else "no"

    return chosen_player, color, chosen_can_lose, DatasetRow(
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


def gen_question_q800_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q800 sample:
    focus: "can_you_lose"
    """
    q_id = "Q800"

    player, color, can_lose, row = _focus_can_you_lose(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)

    row.question = template.format(
        player=f"Player {player}",
        color=color,              # safe even if template doesn't use it
    )

    return row
