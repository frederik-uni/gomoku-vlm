from os.path import join
from pathlib import Path

import numpy as np
import numpy.typing as npt
from gobang.algorithm.ai import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
from gobang.algorithm.policy import CNNPolicy
from gobang.game import GameState

bot1 = None
bot2 = None
current_file = Path(__file__)

home_dir = current_file.parent.parent.parent
policy = CNNPolicy.load_model(join(home_dir, "models", "gobang.pth"))


def convert_board(board: npt.NDArray[np.int8]):
    shape, _ = board.shape
    # black is 1
    # white is -1
    return GameState(shape, np.where(board == 2, -1, board).T, False, False)


def convert_back(game: GameState):
    board = game.board
    return np.where(board == -1, 2, board).T


def generate_next_move_greedy(board: npt.NDArray[np.int8]) -> tuple[int, int]:
    global bot1
    if bot1 is None:
        bot1 = GreedyPolicyPlayer(policy)
    x, y = bot1.get_move(convert_board(board))
    return (y, x)


def generate_next_move_probabilistic(board: npt.NDArray[np.int8]) -> tuple[int, int]:
    global bot2
    if bot2 is None:
        bot2 = ProbabilisticPolicyPlayer(policy, temperature=0.1)
    x, y = bot2.get_move(convert_board(board))
    return (y, x)


if __name__ == "__main__":
    move = generate_next_move_probabilistic(np.zeros((15, 15), dtype=np.int8))
    print(move)
