import numpy as np

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
PATTERN_WEIGHTS = {5: 1_000_000, 4: 10_000, 3: 1000, 2: 100, 1: 10, 0: 0}


def neighbors_exist(board: np.ndarray, y: int, x: int, radius: int) -> bool:
    R, C = board.shape
    y0 = max(0, y - radius)
    y1 = min(R, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(C, x + radius + 1)
    return bool(np.count_nonzero(board[y0:y1, x0:x1]) > 0)


def generate_moves(board: np.ndarray, radius: int = 2) -> list[tuple[int, int]]:
    R, C = board.shape
    moves = []
    if not np.any(board != 0):
        return [(R // 2, C // 2)]  # first move always center
    for y in range(R):
        for x in range(C):
            if board[y, x] != 0:
                continue
            if neighbors_exist(board, y, x, radius):
                moves.append((y, x))
    return moves


def board_key(board: np.ndarray) -> bytes:
    # todo: improve
    return board.tobytes()
