# import random
#
# import numpy as np
#
# from gen_dataset.dataset_schema import DatasetRow
# from gen_dataset.sphinx.core import (
#     QuestionFamily,
#     build_basic_dataset_row,
#     get_question_meta,
# )
#
#
# def _focus_can_you_lose(
#     q_id: str, sim_id: int, simulated_game: np.ndarray
# ) -> tuple[int, DatasetRow]:
#     """
#     Helper function for any question that has the
#     focus: "can_you_lose"
#     """
#     family, focus = get_question_meta(QuestionFamily.PERCEPTION, q_id)
#
#     win = random.random() < 0.5
#     if win:
#         board = simulated_game[-1]
#         player = 1 if (board != 0).sum() % 2 == 0 else 2
#     else:
#         N = simulated_game.shape[0]
#         idx = np.random.randint(0, N - 1)
#         board = simulated_game[idx]
#         player = 1 if random.random() < 0.5 else 2
#
#     # count black stones (=1) as ground truth
#     num_black = int(np.count_nonzero(board == 1))
#     answer_text = str(num_black)
#
#     return player, build_basic_dataset_row(
#         img_path=None,
#         img_bytes=img_bytes,
#         family=family,
#         q_id=q_id,
#         focus=focus,
#         answer=answer_text,
#     )
#
#
# def gen_question_q104_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q1 sample:
#     focus: "count_black_stones"
#     """
#     q_id = "Q1"
#     player, dataset_row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#     question_text = (
#         "“Analyze the position in this Gomoku game."
#         "Player 1 is black; Player 2 is white."
#         f"Determine whether a victory for {player} remains achievable under optimal play."
#         "Answer strictly with ‘yes’ or ‘no’.”"
#     )
#
#     dataset_row.question = question_text
#
#     # Optionally, add additional valid answers here
#
#     return dataset_row
#
#
# def gen_question_q105_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q2 sample:
#     focus: "count_black_stones"
#     """
#     q_id = "Q2"
#
#     player, dataset_row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#     question_text = (
#         "“You are analyzing the current Gomoku game state."
#         "Player 1 uses black stones and Player 2 uses white stones."
#         f"Evaluate the position and determine whether ${player} has a forced win from this position, assuming optimal play by both sides."
#         "Answer with either ‘yes’ or ‘no’.”"
#     )
#     dataset_row.question = question_text
#
#     return dataset_row
#
#
# def gen_question_q106_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q3 sample:
#     focus: "count_black_stones"
#     """
#     q_id = "Q3"
#
#     player, dataset_row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#     question_text = (
#         "“Consider the following Gomoku position."
#         "Black represents Player 1 and white represents Player 2."
#         f"Your task is to assess the position and state whether {player} can still win the game from here."
#         "Respond with a single word: ‘yes’ or ‘no’.”"
#     )
#
#     dataset_row.question = question_text
#
#     return dataset_row
#
#
# def gen_question_q107_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q4 sample:
#     focus: "count_black_stones"
#     """
#     q_id = "Q4"
#
#     player, dataset_row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#     question_text = (
#         "“You are reviewing a Gomoku board position."
#         "Player 1 (black stones) and Player 2 (white stones) have already played several moves."
#         f"Based solely on the current configuration, determine if a winning sequence for {player} is still possible."
#         "Provide a one-word answer: ‘yes’ or ‘no’.”"
#     )
#     dataset_row.question = question_text
#
#     return dataset_row
