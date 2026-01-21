import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from bots.ai_bot import generate_next_move_greedy
from game_logic import position_is_empty
from sim_game import render_game_step, simulate_game


def init(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id)
    return processor, model


def ask(model, processor, question, img, max_new_tokens):
    inputs = processor.apply_chat_template(
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a vision-language model analyzing Gomoku game positions.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {
                        "type": "text",
                        "text": question.replace(
                            "You are a vision-language model analyzing Gomoku game positions.",
                            "",
                        ).strip(),
                    },
                ],
            },
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=inputs,
        images=[img],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    prompt_text = processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)[
        0
    ]

    pred = decoded[len(prompt_text) :].strip()
    return pred


def get_next_move_(model, processor, state, player: int):
    img = render_game_step(state)
    color = "Black" if player == 1 else "White"
    q = f"""
    It is {color}'s turn.

    Task:
    What is the best move to play now for {color}?

    Answer format: "row col" (0-based). Output nothing else.
    """
    pred = ask(model, processor, q, img, 64)
    y, x = map(int, pred.split())
    return (y, x)


_MODEL = None
_PROC = None


def get_next_move(state, player: int):
    global _MODEL
    global _PROC
    if _MODEL is None or _PROC is None:
        p, m = init("Qwen/Qwen3-VL-2B-Instruct")
        _PROC = p
        _MODEL = m
    while True:
        try:
            y, x = get_next_move_(_MODEL, _PROC, state, player)
            if position_is_empty(state, y, x):
                return (y, x)
            print("invalid move retry")
        except:
            print("invalid format retry")
            pass


if __name__ == "__main__":
    import numpy as np

    game = simulate_game(generate_next_move_greedy)
    idx = np.random.randint(game.shape[0])
    random_board = game[idx]

    pred = get_next_move(random_board, idx % 2 + 1)
    print(pred)
