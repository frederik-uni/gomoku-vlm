import random
import shutil
import warnings
from enum import Enum
import numpy as np
from pathlib import Path
import json

from src import sim_game
from gen_dataset.dataset_schema import DatasetRow

SPHINX_CONFIG_PATH = Path(__file__).with_name("sphinx_config.json")
# core.py -> /sphinx -> /generate_dataset -> /PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load sphinx_config.json
with open(SPHINX_CONFIG_PATH, "r", encoding="utf-8") as config_file:
    SPHINX_CONFIG = json.load(config_file)

# base dir from config, e.g. "dataset/sphinx/out/"
SPHINX_BASE_OUT = PROJECT_ROOT / SPHINX_CONFIG["general"].get("output_dir")

# These will be filled later by the runner, calling core.init_output_dirs
SPHINX_IMG_OUT_PATH: Path | None = None
SPHINX_GAME_STATES_OUT_PATH: Path | None = None
SPHINX_PARQUET_OUT_PATH: Path | None = None


class QuestionFamily(str, Enum):
    PERCEPTION = "perception"
    STRATEGY = "strategy"


def get_question_meta(family: QuestionFamily, q_id: str) -> tuple:
    """
    Return the question metadata
    """
    family = family # a bit redundant, may change later
    focus = SPHINX_CONFIG["questions"][family.value][q_id]["focus"]

    return family, focus


def random_turn_index(q_family: QuestionFamily, q_id: str, game_states: np.ndarray) -> int:
    """
    returns a random turn number to sample from the simulated game.
    The values are bound by the user defined settings in the sphinx_config.json file,
    as well as the total number of turns performed during the simulation.
    """

    # if someone deletes those keys from the JSON file, still assign defaults.
    general_config = SPHINX_CONFIG["general"]
    default_min_turns = general_config.get("default_min_turns", 1)
    default_max_turns = general_config.get("default_max_turns", 50)

    question_config = SPHINX_CONFIG["questions"][q_family.value][q_id]
    min_turns = question_config.get("min_turns", default_min_turns)
    max_turns = question_config.get("max_turns", default_max_turns)

    min_turns = max(min_turns, 1)
    # Ensure max turns < total played turns
    num_turns = game_states.shape[0]
    max_turns = min(max_turns, num_turns)

    # Possibly a miss configured sphinx_config file,
    # edge case where the game ended earlier than expected,
    # or a forced last_turn
    if min_turns > max_turns:
        min_turns, max_turns = num_turns, num_turns

    # Return zero based index
    return random.randint(min_turns, max_turns) - 1


def _get_sim_image_dir(sim_id: int) -> Path:
    """
    Return the directory where images for a single simulation are stored,
    e.g.  dataset_NNN/images/sim_0000
    """
    if SPHINX_IMG_OUT_PATH is None:
        raise RuntimeError("init_output_dirs() must be called first")

    sim_dir = SPHINX_IMG_OUT_PATH / f"sim_{sim_id:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir


def _get_sim_game_state_dir(sim_id: int) -> Path:
    """
    Return the directory where game states for a single simulation are stored,
    e.g. dataset_NNN/game_states/sim_0000
    """
    if SPHINX_GAME_STATES_OUT_PATH is None:
        raise RuntimeError("init_output_dirs() must be called first")

    sim_dir = SPHINX_GAME_STATES_OUT_PATH / f"sim_{sim_id:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir


def store_turn_image(board: np.ndarray, turn_index: int, sim_id: int) -> tuple[Path, bytes]:
    """
    Copy the image for a given turn from /tmp to /out
    and return its new location and bytes.

    Args:
        board: (np.ndarray): The 2D-board of the turn.
        turn_index (int): Zero-based turn index used to locate the image in the /tmp folder.
        sim_id (int): Zero-based index of the simulated game (episode).

    Returns:
        tuple[Path, bytes]: A tuple containing:
            - Path: The path of the copied image in the output directory.
            - bytes: The PNG-encoded image bytes.
    """
    filename = f"turn_{turn_index:03d}.png"
    img_path = _get_sim_image_dir(sim_id) / filename
    img = sim_game.render_game_step(board)
    img.save(img_path / filename)

    # read image bytes (PNG-encoded)
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    return img_path, img_bytes


def store_turn_game_state(turn_index: int, sim_id: int, board: np.ndarray) -> Path:
    """
    Store the numpy board for a given turn into the debug folder and
    return the path to the .npy file.

    File layout:
      dataset_NNN/game_states/sim_0000/turn_007.npy
    """
    sim_dir = _get_sim_game_state_dir(sim_id)
    filename = f"turn_{turn_index:03d}.npy"
    out_path = sim_dir / filename

    # Save as .npy for easy debugging with np.load
    np.savetxt(out_path, board, fmt="%d", delimiter=" ")

    return out_path


def build_basic_dataset_row(img_path: Path, img_bytes: bytes, family: QuestionFamily, q_id: str, focus: str, answer: str, valid_answers = None):
    """
    Helper function to build a basic DatasetRow,
    where question and split is filled in later.
    """
    if valid_answers is None:
        valid_answers = [answer]

    row = DatasetRow(
        img_path=str(img_path.relative_to(PROJECT_ROOT)),
        img_bytes=img_bytes,

        family=family.value,
        q_id=q_id,
        focus=focus,

        answer=answer,
        valid_answers=valid_answers,

        question=None,
        split=None
    )
    return row

def select_random_turn_and_store_image(
    family: QuestionFamily,
    q_id: str,
    sim_id: int,
    simulated_game: np.ndarray,
) -> tuple[int, np.ndarray, Path, bytes]:
    """
    Helper function to select a random turn for a given question, fetch the board for that turn,
    and store the corresponding image permanently.

    Args:
        family (QuestionFamily): Question family (e.g. PERCEPTION or STRATEGY).
        q_id (str): Question ID (e.g. "Q1", "Q10").
        sim_id (int): Zero-based index of the simulated game (episode).
        simulated_game (np.ndarray): 3D array of shape (num_turns, size, size)
            containing the board state after each turn.

    Returns:
        tuple[int, np.ndarray, Path, bytes]: A tuple containing:
            - int: The chosen turn index (0-based).
            - np.ndarray: The board state at that turn.
            - Path: The path of the copied image in the output directory.
            - bytes: The PNG-encoded image bytes.
    """
    # pick a random turn index (0-based) according to sphinx_config bounds
    turn_index = random_turn_index(family, q_id, simulated_game)
    board = simulated_game[turn_index]

    # permanently store the image for that turn from /tmp
    img_path, img_bytes = store_turn_image(board, turn_index, sim_id)

    # store the board state for debugging
    store_turn_game_state(turn_index, sim_id, board)

    return turn_index, board, img_path, img_bytes


def select_fixed_turn_and_store_image(
    sim_id: int,
    simulated_game: np.ndarray,
    turn_index: int
) -> tuple[np.ndarray, Path, bytes]:
    """
    Helper function to select a fixed turn for a given question, fetch the board for that turn,
    and store the corresponding image permanently.

    Args:
        sim_id (int): Zero-based index of the simulated game (episode).
        simulated_game (np.ndarray): 3D array of shape (num_turns, size, size)
            containing the board state after each turn.
        turn_index (int): Desired 0-based turn index.

    Returns:
        tuple[np.ndarray, Path, bytes]: A tuple containing:
            - np.ndarray: The board state at that turn.
            - Path: The path of the copied image in the output directory.
            - bytes: The PNG-encoded image bytes.
    """
    if simulated_game.shape[0] == 0:
        raise ValueError(f"Simulated game has not turns.")
    last_index = simulated_game.shape[0] - 1
    if turn_index > last_index or turn_index < 0:
        warnings.warn(
            f"Selected turn_index in select_fixed_turn_and_store_image is out of range. "
            f"Provided: {turn_index}, valid range: 0..{last_index}. "
            f"Using last turn {last_index} instead.",
            RuntimeWarning
        )
        turn_index = last_index

    board = simulated_game[turn_index]

    # permanently store the image for that turn from /tmp
    img_path, img_bytes = store_turn_image(board, turn_index, sim_id)

    # store the board state for debugging
    store_turn_game_state(turn_index, sim_id, board)

    return board, img_path, img_bytes


def init_output_dirs() -> None:
    """
    Create dataset_<NNN>/images and dataset_<NNN>/parquet under SPHINX_BASE_OUT
    and write the paths into SPHINX_IMG_OUT_PATH, SPHINX_PARQUET_OUT_PATH.
    """
    global SPHINX_IMG_OUT_PATH, SPHINX_PARQUET_OUT_PATH, SPHINX_GAME_STATES_OUT_PATH

    SPHINX_BASE_OUT.mkdir(parents=True, exist_ok=True)

    # find first free index of existing dataset_XXX dirs
    next_idx = 0
    while True:
        dataset_root = SPHINX_BASE_OUT / f"dataset_{next_idx:03d}"
        if not dataset_root.exists():
            break
        next_idx += 1

    dataset_root = SPHINX_BASE_OUT / f"dataset_{next_idx:03d}"
    img_dir = dataset_root / "images"
    game_states_dir = dataset_root / "game_states"
    parquet_dir = dataset_root / "parquet"

    img_dir.mkdir(parents=True, exist_ok=True)
    game_states_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    SPHINX_IMG_OUT_PATH = img_dir
    SPHINX_GAME_STATES_OUT_PATH = game_states_dir
    SPHINX_PARQUET_OUT_PATH = parquet_dir

    print(f"Created dataset dir: {dataset_root}")
