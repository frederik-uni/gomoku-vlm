import numpy as np
import numpy.typing as npt
import random
from utils.generic import get_random
from ..game_logic import make_move


def _get_random_empty_position(
    board: npt.NDArray[np.int8], rng: random.Random
) -> tuple[int, int]:
    """
    Return a random (y, x) that is currently empty.
    """

    empties = np.argwhere(board == 0)  # shape: (k, 2) rows of empty [y, x] points
    if len(empties) == 0:
        raise RuntimeError("board is full")

    y, x = rng.choice(empties)
    return y, x


def generate_next_move_random(
    board: npt.NDArray[np.int8], player: int
) -> tuple[int, int]:
    """
    Perform a random, but valid move for the given player.
    Returns the (y, x) position where the move was performed.
    """

    if player not in [1, 2]:
        raise RuntimeError("player must be either 1 or 2")

    y, x = _get_random_empty_position(board, get_random())
    make_move(board, y, x, player)
    return y, x
