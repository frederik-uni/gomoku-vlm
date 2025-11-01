import numpy as np
from ..utils.bot_utils import PATTERN_WEIGHTS, board_key


def evaluate(board: np.ndarray, us: int, them: int) -> int:
    raise NotImplementedError("evaluate is not implemented")


def minimax_alpha_beta(
    board: np.ndarray,
    depth: int,
    radius: int,
    alpha: int,
    beta: int,
    maximizing: bool,
    us: int,
    them: int,
    transposition,
) -> tuple[int, tuple[int, int] | None]:
    key = (board_key(board), depth, maximizing)
    if key in transposition:
        return transposition[key]
    raise NotImplementedError("minimax_alpha_beta is not implemented")


def find_best_move(
    board: np.ndarray,
    us: int,
    max_depth: int,
    radius: int,
    alpha: int = -(10**12),
    beta: int = 10**12,
) -> tuple[tuple[int, int] | None, int]:
    them = (us % 2) + 1
    best_overall_move = None
    best_overall_score = 0
    transposition = {}

    for depth in range(1, max_depth + 1):
        score, move = minimax_alpha_beta(
            board, depth, radius, alpha, beta, True, us, them, transposition
        )

        if move is not None and score >= best_overall_score:
            best_overall_move, best_overall_score = move, score

        if abs(score) >= PATTERN_WEIGHTS[5]:
            break
    return best_overall_move, best_overall_score
