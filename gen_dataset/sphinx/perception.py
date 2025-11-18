import shutil
from typing import List

import numpy as np

from .core import SPHINX_CONFIG, QuestionFamily, random_turn_index, SPHINX_IMG_OUT_PATH, PROJECT_ROOT
from gen_dataset import game_simulator
from gen_dataset.dataset_schema import DatasetRow


def _focus_count_black_stones(q_id: str, sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Helper function for any question that has the
    focus: "count_black_stones"
    """
    family = QuestionFamily.PERCEPTION
    q_id = q_id
    focus = SPHINX_CONFIG["questions"][family.value][q_id]["focus"]

    # pick a random turn index (0-based) according to sphinx_config bounds
    turn_index = random_turn_index(family, q_id, simulated_game)
    board = simulated_game[turn_index]

    # permanently copy the image from /tmp to /out
    tmp_filename = f"turn_{turn_index:03d}.png"
    tmp_path = game_simulator.IMG_TMP_PATH / tmp_filename

    filename = f"sim_{sim_id:04d}_turn_{turn_index:03d}.png"
    img_path = SPHINX_IMG_OUT_PATH / filename

    shutil.copy2(tmp_path, img_path)

    # read image bytes (PNG-encoded)
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # count black stones (=1) as ground truth
    num_black = int(np.count_nonzero(board == 1))
    answer_text = str(num_black)

    # build DatasetRow (question and split is filled later)
    row = DatasetRow(
        img_path=str(img_path.relative_to(PROJECT_ROOT)),
        img_bytes=img_bytes,

        family=family.value,
        q_id=q_id,
        focus=focus,

        answer=answer_text,
        valid_answers=[answer_text],

        question=None,
        split=None
    )

    return row

def gen_question_q1_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
    """
    Generate a single Q1 sample:
    focus: "count_black_stones"
    """
    q_id = "Q1"
    question_text = "How many black stones (player 1) are on the board?"

    row = _focus_count_black_stones(q_id, sim_id, simulated_game)
    row.question = question_text

    return row


def generate_perception_questions_for_episode(sim_id: int, simulated_game: np.ndarray) -> List[DatasetRow]:
    """
    Generates all the perception questions for a single simulated game
    """
    rows: List[DatasetRow] = []

    # Q1
    row = gen_question_q1_sample(sim_id, simulated_game)
    rows.append(row)
    # Q2
    # ...

    return rows
