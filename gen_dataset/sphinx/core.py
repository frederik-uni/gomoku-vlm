import random
import warnings
from enum import Enum
import numpy as np
from pathlib import Path
import json

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
