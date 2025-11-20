import numpy as np
import numpy.typing as npt

from .utils.game_utils import in_bounds


def create_board(size: int = 15):
    return np.zeros((size, size), dtype=np.int8)


def position_is_empty(board: npt.NDArray[np.int8], y: int, x: int) -> bool:
    return board[y, x] == 0


def is_board_full(board: npt.NDArray[np.int8]) -> bool:
    return not np.any(board == 0)


def is_board_empty(board: npt.NDArray[np.int8]) -> bool:
    return not np.any(board != 0)


def make_move(board: npt.NDArray[np.int8], y: int, x: int, player: int):
    if not in_bounds(board, x, y):
        raise ValueError("x/y value out of board range")
    if player not in [1, 2]:
        raise ValueError("player must be either 1 or 2")
    if not position_is_empty(board, y, x):
        raise RuntimeError("position already occupied")
    board[y, x] = player


def count_almost_wins(board: npt.NDArray[np.int8], n: int, player: int) -> int:
    win_windows = np.lib.stride_tricks.sliding_window_view(board, n)
    player_count = (win_windows == player).sum(axis=1)
    empty_count = (win_windows == 0).sum(axis=1)
    mask = (player_count == n - 1) & (empty_count == 1)
    return int(mask.sum())


def count_wins(board: npt.NDArray[np.int8], n: int, player: int):
    mask = (board == player).astype(int)
    kernel = np.ones(n, dtype=int)

    conv = np.convolve(mask, kernel, mode="valid")
    return np.count_nonzero(conv == n)


def _has_player_won_helper(board: npt.NDArray[np.int8], n: int, player: int):
    mask = (board == player).astype(int)
    kernel = np.ones(n, dtype=int)  # dim of win condition

    conv = np.convolve(mask, kernel, mode="valid")
    return np.any(conv == n)


def count(board: npt.NDArray[np.int8], n: int, player: int, almost: bool) -> int:
    """
    returns true if the win condition is satisfied by the given player (n in a row), otherwise false
    """
    func = count_almost_wins if almost else count_wins
    counter = 0
    size = board.shape[0]
    for i in range(0, size):
        col = board[:, i]
        counter += func(col, n, player)
        row = board[i, :]
        counter += func(row, n, player)

    for offset in range(-size + n, size - n + 1):
        diag1 = np.diag(board, k=offset)
        counter += func(diag1, n, player)

        diag2 = np.diag(np.fliplr(board), k=offset)
        counter += func(diag2, n, player)
    return counter


def has_player_won(board: npt.NDArray[np.int8], n: int, player: int) -> bool:
    """
    returns true if the win condition is satisfied by the given player (n in a row), otherwise false
    """

    # Check horizontals
    size = board.shape[0]
    for i in range(0, size):
        col = board[:, i]
        if _has_player_won_helper(col, n, player):
            return True
        row = board[i, :]
        if _has_player_won_helper(row, n, player):
            return True

    # Check diagonals
    for offset in range(-size + n, size - n + 1):
        diag1 = np.diag(board, k=offset)
        if _has_player_won_helper(diag1, n, player):
            return True
        diag2 = np.diag(np.fliplr(board), k=offset)
        if _has_player_won_helper(diag2, n, player):
            return True
    return False


def get_winner(board: npt.NDArray[np.int8], n: int) -> int:
    """
    returns the winner of the given board.
    1 = player1 won
    2 = player2 won
    -1 = no winner, board is full
    0 = no winner, game still in progress
    """

    for player in (1, 2):
        if has_player_won(board, n, player):
            return player
    return -1 if is_board_full(board) else 0
