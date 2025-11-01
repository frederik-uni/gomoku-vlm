import numpy as np

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
PATTERN_WEIGHTS = {5: 1_000_000, 4: 10_000, 3: 1000, 2: 100, 1: 10, 0: 0}


def neighbors_exist(board: np.ndarray, r: int, c: int, radius: int) -> bool:
    R, C = board.shape
    r0 = max(0, r - radius)
    r1 = min(R, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(C, c + radius + 1)
    return bool(np.count_nonzero(board[r0:r1, c0:c1]) > 0)


def generate_moves(board: np.ndarray, radius: int = 2) -> list[tuple[int, int]]:
    R, C = board.shape
    moves = []
    if not np.any(board != 0):
        return [(R // 2, C // 2)]  # first move always center
    for r in range(R):
        for c in range(C):
            if board[r, c] != 0:
                continue
            if neighbors_exist(board, r, c, radius):
                moves.append((r, c))
    return moves


def board_key(board: np.ndarray) -> bytes:
    # todo: improve
    return board.tobytes()
