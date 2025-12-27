import random

import numpy as np

import game_logic
from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import (
    QuestionFamily, persist_turn_image, get_question_text, get_random_turn_index
)


def _focus_can_you_win(
    q_id: str,
    sim_id: int,
    game: np.ndarray,
    *,
    can_win_bias = 0.01,
    max_resamples: int = 2000,
) -> tuple[int, str, bool, DatasetRow]:
    """
    Helper function for any question that has the
    focus: "can_you_win".

    Label definition:
        "can_win" is True iff the current player (the one to move next)
        has an immediate winning move available on this turn
        (i.e., there exists at least one 4+1 completion for this player).

    Sampling strategy:
        - ~0.5 (+ optional bias) to sample a "yes" label.
        The small bias exists because matches that ended with a draw are automatically labeled with a "no" by default
        (It is assumed that no winning position occurred for the bots under optimal play in the case of a draw).

    Returns:
        tuple[int, str, bool, DatasetRow]: A tuple containing:
            - int: The player to move as int (1 if black, 2 if white).
            - str: The color of that player ("black" or "white").
            - bool: Whether the player can win immediately with one move.
            - DatasetRow: The pre-constructed dataset row for the dataset (question still None).
    """
    FOCUS = "can_you_win"
    FAMILY = QuestionFamily.PERCEPTION

    # Sanity Check
    num_turns = game.shape[0]
    if num_turns < 3:
        raise ValueError(f"Need at least 3 turns for {FOCUS}, got num_turns={num_turns} instead.")

    last_turn = game.shape[0] - 1
    end_board = game[last_turn]

    winner = game_logic.get_winner(end_board, 5)

    if winner == -1 or winner == 0:
        can_win = False # Game ended in a draw, or is somehow still in progress, assume no winning opportunity occurred
    else:
        # ~50/50 + user defined bias, to offset the fact that matches can end in a draw
        p_yes = min(1.0, max(0.0, 0.5 + float(can_win_bias)))
        can_win = random.random() < p_yes

    if can_win:
        # Under optimal play the only time when a potential winning position occurs is on the last turn
        after_idx = last_turn
        before_idx = after_idx - 1  # turn that has to be played
        after_board = game[after_idx]
        before_board = game[before_idx]
        player = 1 if (after_idx % 2 == 0) else 2
    else:
        found = False
        for i in range (max_resamples):
            # Just sample a random turn otherwise, assume no winning position did occur for that player,
            # otherwise the bot would have taken advantage of it
            after_idx = get_random_turn_index(game, 1, last_turn - 1) # exclude last turn
            before_idx = after_idx - 1
            after_board = game[after_idx]
            before_board = game[before_idx]
            player = 1 if (after_idx % 2 == 0) else 2
            # Break out of the for-loop if no winning position is possible
            # (i.e. guaranteed that no miss play by the bots occurred)
            if game_logic.count(before_board, 5, player, almost=True) <= 0:
                found = True
                break

        if not found:
            # Fallback, at least assign the correct label
            can_win = game_logic.count(before_board, 5, player, almost=True) > 0

    # Raise runtime error if labels are incorrect
    if (not can_win and game_logic.count(before_board, 5, player, almost=True)
    or can_win and not game_logic.count(before_board, 5, player, almost=True)):
        raise RuntimeError(f"Invalid lables in focus {FOCUS}."
                           f"Answer to the question \"can_you_win\" is: {game_logic.count(before_board, 5, player, almost=True)},"
                           f"but has been classified as {can_win} for turn {before_idx} -> {after_idx}.")

    # Persist the image and get img_bytes
    img_path, img_bytes = persist_turn_image(before_board, before_idx, sim_id)
    # Persist after board for debugging only
    _, _ = persist_turn_image(after_board, after_idx, sim_id)
    color = "black" if player == 1 else "white"
    answer = "yes" if can_win else "no"

    return player, color, can_win, DatasetRow(
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

def gen_question_q700_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q700 sample:
    focus: "can_you_win"
    """
    q_id = "Q700"

    player, color, can_win, row = _focus_can_you_win(q_id, sim_id, simulated_game)

    template = get_question_text(q_id)
    row.question = template.format(
        player=f"Player {player}",
        color=color,
    )

    return row
