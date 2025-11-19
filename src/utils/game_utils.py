import numpy as np


def in_bounds(board: np.ndarray, y: int, x: int) -> bool:
    return 0 <= y < board.shape[0] and 0 <= x < board.shape[1]
