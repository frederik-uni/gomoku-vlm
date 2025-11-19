import random

import numpy as np
import numpy.typing as npt

from utils.generic import get_random


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


def generate_next_move_random(board: npt.NDArray[np.int8]) -> tuple[int, int]:
    """
    Perform a random, but valid move for the given player.
    Returns the (y, x) position where the move was performed.
    """
    y, x = _get_random_empty_position(board, get_random())
    return y, x
