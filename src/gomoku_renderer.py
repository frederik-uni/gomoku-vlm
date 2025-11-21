from PIL import Image, ImageDraw


def calc_coords_gomoku(
    i: int, j: int, cell_size: int = 40, board_origin: tuple[int, int] = (0, 0)
):
    """
    Calculate the coordinates of a piece on the game board.
    """
    x0, y0 = board_origin
    w = h = cell_size

    x = x0 + j * cell_size
    y = y0 + i * cell_size
    return x, y, w, h, "c", "c"


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
