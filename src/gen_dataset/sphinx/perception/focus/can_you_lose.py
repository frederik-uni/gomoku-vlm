# import random
#
# import numpy as np
#
# from gen_dataset.dataset_schema import DatasetRow
# from gen_dataset.sphinx.core import (
#     QuestionFamily, persist_turn_image, persist_turn_game_state, get_question_text
# )
#
#
# def _focus_can_you_lose(
#         q_id: str,
#         sim_id: int,
#         game: np.ndarray,
#         min_turns: int = 0,
#         max_turns: int = 999
# ) -> tuple[int, DatasetRow]:
#     """
#     Helper function for any question that has the
#     focus: "can_you_lose"
#
#     Returns:
#         tuple[int, DatasetRow]:
#             - int: 1 if player 1 should be determined, 2 if player 2 should be determined.
#             - DatasetRow with answer + valid_answers filled (question still None)
#     """
#     FOCUS = "can_you_lose"
#     FAMILY = QuestionFamily.PERCEPTION
#
#     win = random.random() < 0.5
#     if win:
#         idx = game.shape[0] - 1
#         board = game[idx]
#         player = 1 if (board != 0).sum() % 2 == 0 else 2
#     else:
#         N = game.shape[0]
#         idx = np.random.randint(0, N - 1)
#         board = game[idx]
#         player = 1 if random.random() < 0.5 else 2
#
#     # WRONG LOGIC!
#     # count black stones (=1) as ground truth
#     num_black = int(np.count_nonzero(board == 1))
#     answer = str(num_black)
#
#     # Persist the image and get img_bytes
#     img_path, img_bytes = persist_turn_image(board, idx, sim_id)
#     # Persist game state for easier debugging
#     persist_turn_game_state(board, idx, sim_id)
#
#     return player, DatasetRow(
#         img_path=str(img_path),
#         img_bytes=img_bytes,
#
#         family=FAMILY,
#         q_id=q_id,
#         focus=FOCUS,
#
#         answer=answer,
#         valid_answers=[answer],
#
#         # Will be assigned later in the creation process
#         question=None,
#         split=None
#     )
#
#
# def gen_question_q800_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q800 sample:
#     focus: "can_you_lose"
#     """
#     q_id = "Q800"
#     player, row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#
#     template = get_question_text(q_id)
#     row.question = template.format(player=player)
#
#     return row
#
# def gen_question_q801_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q801 sample:
#     focus: "can_you_lose"
#     """
#     q_id = "Q801"
#     player, row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#
#     template = get_question_text(q_id)
#     row.question = template.format(player=player)
#
#     return row
#
# def gen_question_q802_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q802 sample:
#     focus: "can_you_lose"
#     """
#     q_id = "Q802"
#     player, row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#
#     template = get_question_text(q_id)
#     row.question = template.format(player=player)
#
#     return row
#
# def gen_question_q803_sample(sim_id: int, simulated_game: np.ndarray) -> DatasetRow:
#     """
#     Generate a single Q803 sample:
#     focus: "can_you_lose"
#     """
#     q_id = "Q803"
#     player, row = _focus_can_you_lose(q_id, sim_id, simulated_game)
#     player = "Player 1" if player == 2 else "Player 2"
#
#     template = get_question_text(q_id)
#     row.question = template.format(player=player)
#
#     return row
