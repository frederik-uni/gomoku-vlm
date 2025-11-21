from typing import Callable

import numpy as np
from PIL import Image

from bots.ai_bot import generate_next_move_greedy, generate_next_move_probabilistic
from bots.random_bot import generate_next_move_random
from game_logic import create_board, get_winner, make_move
from gomoku_renderer import calc_coords_gomoku, create_gomoku_board, create_pieces
from renderer import render

Func = Callable[[np.ndarray], tuple[int, int]]


def get_func(f: Func | tuple[Func, Func], index: int = 0) -> Func:
    if isinstance(f, tuple):
        return f[index]
    return f


def simulate_game(
    function: Func | tuple[Func, Func],
    size: int = 15,
    n: int = 5,
) -> np.ndarray:
    """
    Play a random Gomoku game with 2 random actors.
    Returns a 3D array of shape (num_moves, size, size) representing the board after each move.
    """
    board = create_board(size)
    game_states = []
    current_player = 1
    while True:
        y, x = get_func(function, ((current_player) - 1 % 2))(board)
        make_move(board, y, x, current_player)
        # print(f"Player {current_player} placed at (y={y}, x={x})")

        game_states.append(board.copy())

        winner = get_winner(board, n)
        if winner != 0:
            if winner == -1:
                print("Game ended in a draw.")
            else:
                print(f"Player {winner} wins!")
            break

        current_player = (current_player % 2) + 1

    return np.stack(game_states)


def render_game_step(state: np.ndarray) -> Image.Image:
    size = 68
    board_img = create_gomoku_board(
        size=15,  # fields
        cell_size=size,  # pixel for cell
        margin=size//2,  # margin on all sides in px
        line_width=2,  # line width
        color=(238, 178, 73),  # board color
        line_color=(0, 0, 0),  # line color
    )
    pieces = create_pieces(size)  # 40 cell size in px

    def calc_coords_gomoku_wrapper(i: int, j: int):
        return calc_coords_gomoku(
            i, j, size, (size//2, size//2)
        )  # 40 cell size in px, (20, 20) margin in px

    return render(
        board_img,
        pieces,
        state,
        calc_coords=calc_coords_gomoku_wrapper,
    )


def sim_game_with_images(size: int = 15, n: int = 5) -> np.ndarray:
    game_states = simulate_game(generate_next_move_random, size, n)
    game_states = simulate_game(
        (generate_next_move_greedy, generate_next_move_probabilistic), size, n
    )
    img = render_game_step(
        game_states[index]
    )  # <- example function/needs styling customization

    return game_states
