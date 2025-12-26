import colorsys
import math
import random
from typing import Callable

import numpy as np
from PIL import Image

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
            # if winner == -1:
            #     print("Game ended in a draw.")
            # else:
            #     print(f"Player {winner} wins!")
            break

        current_player = (current_player % 2) + 1

    return np.stack(game_states)


def rotation_safe_scale(w: int, h: int, deg: float) -> float:
    theta = math.radians(deg)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))

    w_rot = w * c + h * s
    h_rot = w * s + h * c

    return min(w / w_rot, h / h_rot)


def _generate_distinct_colors(n: int = 22):
    colors = []
    for i in range(n):
        h = i / n
        s = random.uniform(0.35, 0.55)
        v = random.uniform(0.70, 0.88)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _contrasting_line_color(board_color: tuple[int, int, int]):
    r, g, b = board_color
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if luminance > 140 else (240, 240, 240)


def render_game_step(
    state: np.ndarray,
    lcolor: tuple[int, int, int] = (238, 178, 73),
    color: tuple[int, int, int] = (0, 0, 0),
    rotate_deg: int | None = None,
) -> Image.Image:
    size = 68
    board_img = create_gomoku_board(
        size=15,  # fields
        cell_size=size,  # pixel for cell
        margin=size // 2,  # margin on all sides in px
        line_width=2,  # line width
        color=color,  # board color
        line_color=lcolor,  # line color
    )
    pieces = create_pieces(size)  # 40 cell size in px

    def calc_coords_gomoku_wrapper(i: int, j: int):
        return calc_coords_gomoku(
            i, j, size, (size // 2, size // 2)
        )  # 40 cell size in px, (20, 20) margin in px

    img = render(board_img, pieces, state, calc_coords=calc_coords_gomoku_wrapper)
    w, h = img.size
    if rotate_deg is not None and rotate_deg != 0:
        scale = rotation_safe_scale(w, h, rotate_deg)

        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)

        img = img.rotate(
            rotate_deg,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=color,
        )

        canvas = Image.new("RGB", (w, h), color)
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        img = canvas
    return img


def render_game_step_rand(state: np.ndarray, non_rand: bool = False) -> Image.Image:
    if non_rand:
        return render_game_step(state=state)
    palette = _generate_distinct_colors(22)
    board_color = random.choice(palette)
    line_color = _contrasting_line_color(board_color)

    rotate_deg = random.randint(-30, 30)

    return render_game_step(
        state=state,
        lcolor=board_color,
        color=line_color,
        rotate_deg=rotate_deg,
    )
