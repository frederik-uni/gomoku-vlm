import shutil
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from core import SPHINX_CONFIG
import core
from gen_dataset import game_simulator
from gen_dataset.dataset_schema import DatasetRow


def gen_q1_sample(num_sample: int) -> DatasetRow:
    """
    Generate a single Q1 sample:
    focus: "count_black_pieces"
    """
    question_text = "How many black stones (player 1) are on the board?"

    family = core.QuestionFamily.PERCEPTION
    q_id = "Q1"
    focus = SPHINX_CONFIG["questions"][family.value][q_id]["focus"]

    # TODO: One Sim Run -> Multiple Questions
    # simulate one random game and render PNGs into tmp/sim_game
    game_states = game_simulator.sim_game_with_images()

    # pick a random turn index (0-based) according to sphinx_config bounds
    turn_index = core.random_turn_index(family, q_id, game_states)
    board = game_states[turn_index]

    # permanently copy the image from /tmp to /out
    tmp_filename = f"move_{turn_index:03d}.png"
    tmp_path = game_simulator.IMG_TMP_PATH / tmp_filename

    filename = f"sample_{num_sample}_move_{turn_index:03d}.png"
    image_path = core.SPHINX_IMG_OUT_PATH / filename

    shutil.copy2(tmp_path, image_path)

    # count black stones (=1) as ground truth
    num_black = int(np.count_nonzero(board == 1))
    answer_text = str(num_black)

    # build DatasetRow (split is filled later)
    row = DatasetRow(
        family=family.value,
        q_id=q_id,
        focus=focus,
        question=question_text,
        answer=answer_text,
        image_path=str(image_path.relative_to(core.PROJECT_ROOT)),
        split=None,
    )

    return row


def gen_q1_dataset(num_samples: int) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    for i in range(num_samples):
        row = gen_q1_sample(num_sample=i)
        rows.append(row)
    return rows


def write_q1_parquet(num_samples: int) -> None:
    # 1) generate DatasetRow objects
    rows = gen_q1_dataset(num_samples)

    # 2) convert dataclasses to dicts
    data = [asdict(row) for row in rows]

    # 3) build DataFrame
    df = pd.DataFrame(data)

    # 4) get output path for Q1
    out_path = core.SPHINX_PARQUET_OUT_PATH / "Q1.parquet"  # e.g. PROJECT_ROOT/dataset/sphinx/out/Q1.parquet

    # 5) write parquet
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    write_q1_parquet(num_samples=10)