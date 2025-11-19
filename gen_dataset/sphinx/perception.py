import shutil
from typing import List

import numpy as np

from .core import SPHINX_CONFIG, QuestionFamily, random_turn_index, SPHINX_IMG_OUT_PATH, PROJECT_ROOT, store_turn_image, \
    build_basic_dataset_row, select_turn_and_store_image
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

    # choose turn, get board, store image
    turn_index, board, img_path, img_bytes = select_turn_and_store_image(
        family=family,
        q_id=q_id,
        sim_id=sim_id,
        simulated_game=simulated_game,
    )

    # count black stones (=1) as ground truth
    num_black = int(np.count_nonzero(board == 1))
    answer_text = str(num_black)

    return build_basic_dataset_row(img_path, img_bytes, family, q_id, focus, answer_text)


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
