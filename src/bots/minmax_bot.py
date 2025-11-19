import sys


def evaluate(board: np.ndarray, us: int, them: int) -> int:
    raise NotImplementedError("evaluate is not implemented")


def minimax_alpha_beta(
    board: np.ndarray,
    depth: int,
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

    # stop cond
    if has_player_won(board, 5, us):
        return PATTERN_WEIGHTS[5], None
    if has_player_won(board, 5, them):
        return -PATTERN_WEIGHTS[5], None

    if depth == 0:
        val = evaluate(board, us, them)
        return val, None

    moves = generate_moves(board, radius=2)
    if not moves:
        return 0, None

    # find recursive
    scored_moves = []
    for mv in moves:
        y, x = mv
        board[y, x] = us if maximizing else them
        s = evaluate(board, us, them)
        board[y, x] = 0
        scored_moves.append((s, mv))

    scored_moves.sort(reverse=maximizing, key=lambda x: x[0])
    ordered_moves = [m[1] for m in scored_moves]

    best_move = None

    value = sys.maxsize * (-1 if maximizing else 1)
    cur = us if maximizing else them
    for y, x in ordered_moves:
        board[y, x] = cur
        child_score, _ = minimax_alpha_beta(
            board,
            depth - 1,
            alpha,
            beta,
            not maximizing,
            us,
            them,
            transposition,
        )
        board[y, x] = 0
        vc = value
        if maximizing:
            value = max(value, child_score)
            alpha = max(alpha, value)
        else:
            value = min(value, child_score)
            beta = min(beta, value)
        best_move = (y, x) if vc != value else best_move
        if alpha >= beta:
            break

    transposition[key] = (value, best_move)
    return value, best_move


def find_best_move(
    board: np.ndarray,
    us: int,
    max_depth: int,
    alpha: int = -(10**12),
    beta: int = 10**12,
) -> tuple[tuple[int, int] | None, int]:
    if sys.maxsize <= 2**32:
        raise RuntimeError("This implementation is not supported on 32-bit systems")

    them = (us % 2) + 1
    best_overall_move = None
    best_overall_score = 0
    transposition = {}

    # iterative deepening
    for depth in range(1, max_depth + 1):
        score, move = minimax_alpha_beta(
            board, depth, alpha, beta, True, us, them, transposition
        )

        if move is not None and score >= best_overall_score:
            best_overall_move, best_overall_score = move, score

        if abs(score) >= PATTERN_WEIGHTS[5]:
            break
    return best_overall_move, best_overall_score


def generate_next_move_minimax(board: npt.NDArray[np.int8]) -> tuple[int, int]:
    player = 1 if (board != 0).sum() % 2 == 0 else 2

    return find_best_move(board, player, 3)[0] or (0, 0)
