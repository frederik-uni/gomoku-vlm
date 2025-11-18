from dataclasses import asdict
from typing import List

import pandas as pd

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import SPHINX_CONFIG
from gen_dataset import sphinx, game_simulator
from gen_dataset.sphinx import core as sphinx_core


def _determine_num_of_required_episodes() -> int:
    """
        Return how many full game simulations we need.

        Logic:
          num_episodes = max(required_samples over all questions)
          where required_samples = per-question "num_samples" or global default.
        """
    general = SPHINX_CONFIG.get("general", {})
    max_required = general.get("default_num_samples", 0)

    question_families = SPHINX_CONFIG.get("questions", {})

    for family in question_families.values():
        for question in family.values():
            n = question.get("num_samples", 0)
            max_required = max(max_required, n)

    return max_required

def generate_question_dataset() -> List[DatasetRow]:
    """
    Simulates multiple episodes of the entire game
    and generates one of each question per episode,
    slowly creating the entire dataset
    """
    num_required_episodes = _determine_num_of_required_episodes()
    rows: List[DatasetRow] = []
    for sim_id in range(num_required_episodes):
        simulated_game = game_simulator.sim_game_with_images()

        perception_rows: List[DatasetRow] = sphinx.perception.generate_perception_questions_for_episode(sim_id, simulated_game)
        rows.extend(perception_rows)

        strategy_rows: List[DatasetRow] = sphinx.strategy.generate_strategy_questions_for_episode(sim_id, simulated_game)
        rows.extend(strategy_rows)

    return rows


def assemble_parquet_file(rows: List[DatasetRow]) -> None:
    # convert dataclasses to dicts
    data = [asdict(row) for row in rows]

    # build DataFrame
    df = pd.DataFrame(data)

    # get output path for Q1
    out_path = sphinx_core.SPHINX_PARQUET_OUT_PATH / "dataset.parquet"  # e.g. PROJECT_ROOT/dataset/sphinx/out/dataset.parquet

    # write parquet
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    rows = generate_question_dataset()
    assemble_parquet_file(rows)