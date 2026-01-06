import random
import shutil
import tomllib
import warnings
from enum import Enum
from typing import Dict

import numpy as np
from pathlib import Path
import json

import sim_game
from gen_dataset.dataset_schema import DatasetRow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPHINX_CONFIG_PATH = PROJECT_ROOT / "sphinx_config.toml"
DEFAULT_SPHINX_QUESTION_PATH = PROJECT_ROOT / "sphinx_questions.toml"
DEFAULT_SPHINX_OUT_ROOT_PATH = PROJECT_ROOT / "out"

# dynamic, will be set via init_sphinx_environment() in runner.py
SPHINX_CONFIG: dict | None = None
SPHINX_CONFIG_PATH: Path | None = None

SPHINX_QUESTIONS: dict | None = None
SPHINX_QUESTIONS_PATH: Path | None = None

SPHINX_OUT_ROOT_PATH: Path | None = None

SPHINX_IMG_PATH: Path | None = None
SPHINX_GAME_STATES_PATH: Path | None = None
SPHINX_PARQUET_PATH: Path | None = None

class QuestionFamily(str, Enum):
    PERCEPTION = "perception"
    STRATEGY = "strategy"


def get_random_turn_index(game: np.ndarray,min_idx: int = 0, max_idx: int | None = None) -> int:
    """
    Return a random 0-based turn index.

    Args:
        game: np.ndarray of shape (num_turns, board_h, board_w)
        min_idx: minimum allowed turn index (0-based, inclusive).
        max_idx: maximum allowed turn index (0-based, inclusive).
                 If None, it defaults to the last index (num_turns - 1).

    Behavior:
        - Clamps [min_idx, max_idx] to the valid range [0, num_turns-1].
        - If after clamping, min_idx > max_idx, use last index only
    """
    num_turns = game.shape[0]

    if num_turns == 0:
        raise ValueError("get_random_turn_index: game has no turns (num_turns = 0).")

    # Default upper bound = last available index
    if max_idx is None:
        max_idx = num_turns - 1

    # Clamp into [0, num_turns-1]
    min_idx = max(min_idx, 0)
    max_idx = min(max_idx, num_turns - 1)

    # If min_idx > max_idx, use last index only
    if min_idx > max_idx:
        min_idx = max_idx = num_turns - 1

    return random.randint(min_idx, max_idx)


def _get_sim_image_dir(sim_id: int) -> Path:
    """
    Return the directory where images for a single simulation are stored,
    e.g.  dataset_NNN/images/sim_0000
    """
    if SPHINX_IMG_PATH is None:
        raise RuntimeError("init_output_dirs() must be called first")

    sim_dir = SPHINX_IMG_PATH / f"sim_{sim_id:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir


def _get_sim_game_state_dir(sim_id: int) -> Path:
    """
    Return the directory where game states for a single simulation are stored,
    e.g. dataset_NNN/game_states/sim_0000
    """
    if SPHINX_GAME_STATES_PATH is None:
        raise RuntimeError("init_output_dirs() must be called first")

    sim_dir = SPHINX_GAME_STATES_PATH / f"sim_{sim_id:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir


def persist_turn_image(
        board: np.ndarray,
        turn_index: int,
        sim_id: int,
        *,
        non_rand_img: bool) -> tuple[Path, bytes]:
    """
    Renders the provided board as a .png image and then saves it.

    Args:
        board: (np.ndarray): The 2D-board of the turn.
        turn_index (int): Zero-based turn index used to correctly name the file.
        sim_id (int): Zero-based sim id index (episode) to correctly locate the folder.
        non_rand_img: (bool) Whether to exclude image alterations, like rotation or discoloration from the image rendering process.

    Returns:
        tuple[Path, bytes]: A tuple containing:
            - Path: The path of the copied image in the output directory.
            - bytes: The PNG-encoded image bytes.
    """
    filename = f"turn_{turn_index:03d}.png"
    img_path = _get_sim_image_dir(sim_id) / filename
    # img = sim_game.render_game_step(board)
    img = sim_game.render_game_step_rand(board, non_rand=non_rand_img)
    img.save(img_path)

    # read image bytes (PNG-encoded)
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    return img_path, img_bytes


def persist_turn_game_state(board: np.ndarray, turn_index: int, sim_id: int) -> Path:
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


def _init_output_dirs(gen_subfolder: bool = True) -> None:
    """
    Create dataset_<NNN>/images, game_states, parquet under SPHINX_BASE_OUT_PATH
    and write the paths into SPHINX_IMG_OUT_PATH, SPHINX_PARQUET_OUT_PATH, etc.
    """
    global SPHINX_IMG_PATH, SPHINX_PARQUET_PATH, SPHINX_GAME_STATES_PATH

    if SPHINX_OUT_ROOT_PATH is None:
        raise RuntimeError("SPHINX_BASE_OUT_PATH is not set. Call init_sphinx_environment() first.")

    base = SPHINX_OUT_ROOT_PATH
    base.mkdir(parents=True, exist_ok=True)

    if gen_subfolder:
        # find first free index of existing dataset_XXX dirs
        next_idx = 0
        while True:
            dataset_root = base / f"dataset_{next_idx:03d}"
            if not dataset_root.exists():
                break
            next_idx += 1
        dataset_root = base / f"dataset_{next_idx:03d}"
    else:
        dataset_root = base

    img_dir = dataset_root / "images"
    game_states_dir = dataset_root / "game_states"
    parquet_dir = dataset_root / "parquet"

    img_dir.mkdir(parents=True, exist_ok=True)
    game_states_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    SPHINX_IMG_PATH = img_dir
    SPHINX_GAME_STATES_PATH = game_states_dir
    SPHINX_PARQUET_PATH = parquet_dir

    print(f"Created dataset dir: {dataset_root}")


def init_sphinx_environment(
    config_path: Path | str = DEFAULT_SPHINX_CONFIG_PATH,
    questions_path: Path | str = DEFAULT_SPHINX_QUESTION_PATH,
    output_path: Path | str = DEFAULT_SPHINX_OUT_ROOT_PATH,
    gen_subfolder: bool = True,
) -> None:
    """
    Load the sphinx_config TOML, initialize all SPHINX_* paths,
    create a fresh dataset_NNN structure.

    Intended to be called once from runner.py.
    """
    global SPHINX_CONFIG, SPHINX_CONFIG_PATH, SPHINX_QUESTIONS, SPHINX_QUESTIONS_PATH, SPHINX_OUT_ROOT_PATH

    config_path = Path(config_path)
    questions_path = Path(questions_path)
    output_path = Path(output_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        SPHINX_CONFIG = tomllib.load(f)

    if not questions_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with questions_path.open("rb") as f:
        SPHINX_QUESTIONS = tomllib.load(f)

    SPHINX_CONFIG_PATH = config_path
    SPHINX_QUESTIONS_PATH = questions_path
    SPHINX_OUT_ROOT_PATH = output_path

    _init_output_dirs(gen_subfolder)


def is_question_configured(q_id: str) -> bool:
    """
    Return True if q_id is present in sphinx_questions.toml under [questions.QXXX].

    Used to decide whether a question should be generated at all.
    """
    if SPHINX_QUESTIONS is None:
        raise RuntimeError(
            "SPHINX_QUESTIONS is not loaded. Call init_sphinx_environment() first."
        )

    questions = SPHINX_QUESTIONS.get("questions", {}) or {}
    return q_id in questions


def get_question_text(q_id: str) -> str:
    """
    Return the question text for a given q_id from sphinx_questions.toml.

    If [general].context exists and is non-empty, it is prepended to the
    per-question text with a blank line separator.
    """
    if SPHINX_QUESTIONS is None:
        raise RuntimeError(
            "SPHINX_QUESTIONS is not loaded. Call init_sphinx_environment() first."
        )

    questions = SPHINX_QUESTIONS.get("questions", {}) or {}
    q_cfg = questions.get(q_id)

    if q_cfg is None or "text" not in q_cfg:
        raise KeyError(f"No question text found for q_id={q_id} in {SPHINX_QUESTIONS_PATH}")

    base_text = q_cfg["text"].strip("\n")

    general = SPHINX_QUESTIONS.get("general", {}) or {}
    ctx = (general.get("context") or "").strip("\n")

    # Build final text with optional global context
    if ctx:
        result = ctx + "\n\n" + base_text.lstrip("\n")
    else:
        result = base_text

    return result

def num_required_samples(q_id: str) -> int:
    """
    Return required sample count for a specific q_id.
    Uses [questions.<q_id>].num_samples if set, otherwise [general].num_samples_per_question.
    """
    if SPHINX_QUESTIONS is None:
        raise RuntimeError("SPHINX_QUESTIONS is not loaded. Call init_sphinx_environment() first.")

    qcfg = SPHINX_QUESTIONS

    general = qcfg.get("general", {}) or {}
    raw_default = general.get("num_samples_per_question")
    if raw_default is None:
        raise ValueError('Missing [general].num_samples_per_question in sphinx_questions.toml')

    try:
        default_n = int(raw_default)
    except (TypeError, ValueError):
        raise ValueError(f"num_samples_per_question must be an int >= 0, got {raw_default!r}")
    if default_n < 0:
        raise ValueError(f"num_samples_per_question must be >= 0, got {default_n}")

    questions = qcfg.get("questions", {}) or {}
    qc = questions.get(q_id) or {}

    raw_n = qc.get("num_samples")
    if raw_n is None:
        return default_n

    try:
        n = int(raw_n)
    except (TypeError, ValueError):
        raise ValueError(f"num_samples for {q_id} must be an int >= 0, got {raw_n!r}")
    if n < 0:
        raise ValueError(f"num_samples for {q_id} must be >= 0, got {n}")
    return n

def should_generate_question(qid: str, generated_questions_count: Dict[str, int]) -> bool:
    """
    Returns for a specific q_id if the question should be generated based on the config.toml file.
    """
    generated_questions_count.setdefault(qid, 0)
    if is_question_configured(qid) and generated_questions_count[qid] < num_required_samples(qid):
        return True
    else:
        return False