from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .bots.ai_bot import generate_next_move_greedy, generate_next_move_probabilistic
from .bots.random_bot import generate_next_move_random
from .game_logic import create_board, get_winner, make_move
from .renderer import calc_coords_gomoku, render

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


def create_gomoku_board(
    size: int = 15,
    cell_size: int = 40,
    margin: int = 20,
    line_width: int = 2,
    color=(238, 178, 73),
    line_color=(0, 0, 0),
):
    size = size - 1
    board_px = size * cell_size + 2 * margin
    img = Image.new("RGB", (board_px, board_px), color=color)
    draw = ImageDraw.Draw(img)

    for i in range(size + 1):
        offset = margin + i * cell_size
        draw.line(
            (margin, offset, board_px - margin, offset),
            width=line_width,
            fill=line_color,
        )
        draw.line(
            (offset, margin, offset, board_px - margin),
            width=line_width,
            fill=line_color,
        )

    if size == 15:
        star_positions = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        star_radius = max(2, cell_size // 8)
        for r, c in star_positions:
            cx = margin + c * cell_size
            cy = margin + r * cell_size
            draw.ellipse(
                (
                    cx - star_radius,
                    cy - star_radius,
                    cx + star_radius,
                    cy + star_radius,
                ),
                fill=line_color,
            )

    return img


def create_gomoku_stone(
    color: str = "black",
    size: int = 40,
    outline="black",
    shadow_color=(255, 255, 255, 100),
) -> Image.Image:
    scale = 4
    large_size = size * scale

    img = Image.new("RGBA", (large_size, large_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.ellipse((0, 0, large_size - 1, large_size - 1), fill=color, outline=outline)

    highlight = Image.new("RGBA", (large_size, large_size), (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(highlight)
    hdraw.ellipse(
        (large_size * 0.1, large_size * 0.1, large_size * 0.6, large_size * 0.6),
        fill=shadow_color,
    )
    img = Image.alpha_composite(img, highlight)

    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def create_pieces(cell_size=40):
    black_piece = create_gomoku_stone("black", cell_size)
    white_piece = create_gomoku_stone("white", cell_size)

    return [black_piece, white_piece]


def render_game_step(state: np.ndarray) -> Image.Image:
    board_img = create_gomoku_board(
        size=15,  # fields
        cell_size=40,  # pixel for cell
        margin=20,  # margin on all sides in px
        line_width=2,  # line width
        color=(238, 178, 73),  # board color
        line_color=(0, 0, 0),  # line color
    )
    pieces = create_pieces(40)  # 40 cell size in px

    def calc_coords_gomoku_wrapper(i: int, j: int):
        return calc_coords_gomoku(
            i, j, 40, (20, 20)
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
