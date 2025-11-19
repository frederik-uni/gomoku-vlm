import math
from collections import defaultdict
from dataclasses import asdict
from typing import List

import pandas as pd

from gen_dataset.dataset_schema import DatasetRow
from gen_dataset.sphinx.core import SPHINX_CONFIG
from gen_dataset import sphinx, game_simulator
from gen_dataset.sphinx import core as sphinx_core
from gen_dataset.sphinx.perception.per_simulation import generate_perception_questions_for_episode
from gen_dataset.sphinx.strategy.per_simulation import generate_strategy_questions_for_episode


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
        print(f"Simulating {sim_id} / {num_required_episodes}")
        simulated_game = game_simulator.sim_game_with_images()

        perception_rows: List[DatasetRow] = generate_perception_questions_for_episode(sim_id, simulated_game)
        rows.extend(perception_rows)

        strategy_rows: List[DatasetRow] = generate_strategy_questions_for_episode(sim_id, simulated_game)
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


def _check_train_eval_split_config(train_r: float, eval_r: float, test_r: float) -> None:
    # basic sanity: non-negative
    if train_r < 0 or eval_r < 0 or test_r < 0:
        raise ValueError(
            f"Invalid split_ratios (must be >= 0): "
            f"train={train_r}, eval={eval_r}, test={test_r}"
        )

    total_r = train_r + eval_r + test_r

    # require sum ≈ 1.0, with float tolerance
    if not math.isclose(total_r, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Invalid split_ratios: train+eval+test={total_r} (expected 1.0). "
            "Please fix sphinx_config.json."
        )


def assign_splits(rows: List[DatasetRow]) -> None:
    """
    In-place assignment of row.split based on split_ratios in sphinx_config.json.
    """
    ratios = SPHINX_CONFIG["general"].get("split_ratios", {})
    train_r = float(ratios.get("train", 0.8))
    eval_r  = float(ratios.get("eval", 0.1))
    test_r  = float(ratios.get("test", 0.1))

    _check_train_eval_split_config(train_r, eval_r, test_r)

    # Group indices by q_id
    by_qid: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_qid[row.q_id].append(idx) # map q_id -> idx, where the element is found in the list

    for q_id, idxs in by_qid.items():
        num_samples_for_this_question = len(idxs)

        # Compute counts for this question
        n_train = int(round(num_samples_for_this_question * train_r))
        n_eval = int(round(num_samples_for_this_question * eval_r))
        n_test = num_samples_for_this_question - n_train - n_eval # rest goes to test
        # Fail if any of the resulting splits is < 0 (probably sample size too small for that question)
        if n_train < 0 or n_eval < 0 or n_test < 0:
            raise ValueError(
                f"Bad split for q_id={q_id}: "
                f"num_samples_for_this_question={num_samples_for_this_question}, n_train={n_train}, n_eval={n_eval}, n_test={n_test}. "
                "Check your \"num_samples\" for that questions and \"split_ratios\" in sphinx_config.json."
            )

        # Assign splits for this question
        for local_idx, row_idx in enumerate(idxs):
            row = rows[row_idx]
            if local_idx < n_train:
                row.split = "train"
            elif local_idx < n_train + n_eval:
                row.split = "eval"
            else:
                row.split = "test"


if __name__ == "__main__":
    sphinx.core.init_output_dirs() # sets SPHINX_IMG_OUT_PATH / SPHINX_PARQUET_OUT_PATH

    rows = generate_question_dataset()
    # If you run Hydra, then here, before assigning the splits

    assign_splits(rows)
    assemble_parquet_file(rows)