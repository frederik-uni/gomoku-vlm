import random
import shutil
import warnings
from enum import Enum
import numpy as np
from pathlib import Path
import json

from gen_dataset import game_simulator
from gen_dataset.dataset_schema import DatasetRow

SPHINX_CONFIG_PATH = Path(__file__).with_name("sphinx_config.json")
# core.py -> /sphinx -> /generate_dataset -> /PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load sphinx_config.json
with open(SPHINX_CONFIG_PATH, "r", encoding="utf-8") as config_file:
    SPHINX_CONFIG = json.load(config_file)

# root dir for permanent files generated via sphinx
SPHINX_OUT_PATH = PROJECT_ROOT / SPHINX_CONFIG["general"].get("output_dir")
SPHINX_OUT_PATH.mkdir(parents=True, exist_ok=True)

# dir for permanent images that are needed for training
SPHINX_IMG_OUT_PATH = SPHINX_OUT_PATH / "images"
SPHINX_IMG_OUT_PATH.mkdir(parents=True, exist_ok=True)

# dir for parquet files
SPHINX_PARQUET_OUT_PATH = SPHINX_OUT_PATH / "parquet"
SPHINX_PARQUET_OUT_PATH.mkdir(parents=True, exist_ok=True)

def get_output_path_for(q_id: str) -> Path:
    return SPHINX_OUT_PATH / f"{q_id}.parquet"

class QuestionFamily(str, Enum):
    PERCEPTION = "perception"
    STRATEGY = "strategy"

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

    # Possibly miss configured sphinx_config file, or edge case where the game ended much earlier than expected
    if min_turns > max_turns:
        warnings.warn(
            f"Possibly inconsistent config for {q_family.value} {q_id}: "
            f"min_turns={min_turns}, max_turns={max_turns}, num_turns={num_turns}. "
            "Falling back to sensible defaults.",
            RuntimeWarning,
        )
        min_turns, max_turns = default_min_turns, min(default_max_turns, num_turns)
        # sphinx_config file must have miss configured defaults
        if min_turns > max_turns:
            raise ValueError(
                f"Inconsistent turn range for {q_family.value} {q_id}: "
                f"min_turns={min_turns}, max_turns={max_turns}, num_turns={num_turns}"
            )

    # Return zero based index
    return random.randint(min_turns, max_turns) - 1


def store_turn_image(turn_index: int, sim_id: int) -> tuple[Path, bytes]:
    """
    Copy the image for a given turn from /tmp to /out
    and return its new location and bytes.

    Args:
        turn_index (int): Zero-based turn index used to locate the image in the /tmp folder.
        sim_id (int): Zero-based index of the simulated game (episode).

    Returns:
        tuple[Path, bytes]: A tuple containing:
            - Path: The path of the copied image in the output directory.
            - bytes: The PNG-encoded image bytes.
    """
    tmp_filename = f"turn_{turn_index:03d}.png"
    tmp_path = game_simulator.IMG_TMP_PATH / tmp_filename

    filename = f"sim_{sim_id:04d}_turn_{turn_index:03d}.png"
    img_path = SPHINX_IMG_OUT_PATH / filename

    shutil.copy2(tmp_path, img_path)

    # read image bytes (PNG-encoded)
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    return img_path, img_bytes


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

def select_turn_and_store_image(
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
    img_path, img_bytes = store_turn_image(turn_index, sim_id)

    return turn_index, board, img_path, img_bytes