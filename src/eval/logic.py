import difflib
import io
import json
import os
from collections import defaultdict
from typing import Literal, Optional, cast

import numpy as np
import pandas as pd
import regex as re
import requests
from PIL import Image

from ask import ask
from datasets import concatenate_datasets, load_dataset
from utils.ai_utils import get_device


def normalize(s: str):
    return s.lower().strip()


def is_yes(data):
    s = str(data)
    s = s.lower().strip()
    s = re.sub(r"[^a-z]+", " ", s)
    s = s.strip()
    return s.startswith("yes")


def last_line(s: str):
    lines = [line.strip().lower() for line in s.splitlines() if line.strip()]

    if not lines:
        raise ValueError("Empty output")

    return lines[-1]


def ask_lisa(question1: str, question2: str) -> tuple[bool, str]:
    api_token = os.getenv("API_TOKEN")

    if not api_token:
        raise RuntimeError("API_TOKEN is not set")
    try:
        url = "https://chat-1.ki-awz.iisys.de/api/chat/completions"

        model: str = "lisa-v40-rc1-qwen3235b-a22b-instruct"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    ROLE:
                    You are a strict binary classifier.

                    DEFINITIONS:
                    - ANSWER1 is a JSON array of strings. Each string is a complete valid answer unit. ANSWER1 is the ground-truth.
                    - ANSWER2 may contain one or more answer units.

                    TASK:
                    Decide if ANSWER2 is an EXACT match to ANY element in ANSWER1.

                    INPUTS:
                    <ANSWER1>
                    {question1}
                    </ANSWER1>

                    <ANSWER2>
                    {question2}
                    </ANSWER2>

                    RULES (strict):
                    - If the answer content in ANSWER2 equals one or more complete elements from ANSWER1 exactly, character-for-character IT IS A MATCH.
                    - Do NOT accept substrings, partial matches or “close” answers.
                    - If ANSWER2 contains multiple answer contents, all of them must exactly match elements in ANSWER1 and they must not contradict each other; if any included answer content fails to exactly match ANSWER1, return no.
                    - If ANSWER2 is empty or whitespace-only, return no.
                    - If anything is unclear, return no.

                    OUTPUT FORMAT:
                    [Your Reasoning Process]
                    [----------------------]
                    [On the final line output ONLY one word ‘yes’ or ‘no’]
                    """,
                }
            ],
        }
        response = requests.post(url, json=data, headers=headers)

        data = response.json()["choices"][0]["message"]["content"]

        return is_yes(last_line(str(data))), str(data)
    except:
        print("error")
        return ask_lisa(question1, question2)


def contains_match(pred: str, valid_answers: list[str]) -> bool:
    pred_norm = normalize(pred)

    for ans in valid_answers:
        ans_norm = normalize(ans)
        if not ans_norm:
            continue

        if ans_norm in pred_norm:
            return True

    return False


def fuzzy_match(pred: str, valid_answers: list[str], threshold: float = 0.75) -> bool:
    pred_norm = normalize(pred)
    for ans in valid_answers:
        ratio = difflib.SequenceMatcher(a=pred_norm, b=normalize(ans)).ratio()
        if ratio >= threshold:
            return True
    return False


def match_answer(
    pred: str,
    valid_answers: list[str],
    regex: Optional[str],
    mode: Literal["exact", "contains", "fuzzy", "regex", "lisa"] = "exact",
):
    if regex is None and mode == "regex":
        mode = "exact"
    if mode == "lisa":
        v, t = ask_lisa(str(valid_answers), pred)
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(
                f"=======================\nGround truth:\n{valid_answers}\n\nResp:\n{pred}\n\nResult:\n{v}\n\nLisa:\n{t}\n\n"
            )
        return v
    if mode == "exact":
        return pred in valid_answers
    if mode == "contains":
        return contains_match(pred, valid_answers)
    if mode == "fuzzy":
        return fuzzy_match(pred, valid_answers)
    if mode == "regex" and regex is not None:
        pattern = re.compile(regex)
        return any(pattern.fullmatch(ans) for ans in valid_answers)

    raise ValueError("Unknown match mode")


def eval_vlm_on_parquet(
    processor,
    model,
    match_mode: Literal["exact", "contains", "fuzzy", "regex", "lisa"] = "exact",
    max_new_tokens=64,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    dataset: str = "eval",
) -> dict[str, float]:
    device = get_device(device)
    model = model.to(device)

    model.eval()

    ds = load_dataset("eganscha/gomoku_vlm_ds", dataset)
    df = concatenate_datasets(list(ds.values())).to_pandas()
    df = cast(pd.DataFrame, df)
    # df = pd.read_parquet(parquet_path)

    def ensure_list(x):
        if isinstance(x, np.ndarray):
            return x.astype(str).tolist()
        if isinstance(x, list):
            if not all(isinstance(item, str) for item in x):
                raise ValueError("valid_answers must be a list of strings")
            return x
        if isinstance(x, str):
            try:
                parsed = json.loads(x)
            except json.JSONDecodeError:
                raise ValueError("valid_answers JSON string is invalid")

            if not isinstance(parsed, list):
                raise ValueError("valid_answers JSON must decode to a list")

            if not all(isinstance(item, str) for item in parsed):
                print("[WARNING] casted valid_answers", parsed)
                return [str(item) for item in parsed]
            return parsed
        raise ValueError("valid_answers must be list or JSON string")

    df["valid_answers"] = df["valid_answers"].apply(ensure_list)

    correct = defaultdict(int)
    total = defaultdict(int)
    from tqdm import tqdm

    for _, row in tqdm(df.iterrows(), desc="Processing", unit="item"):
        question = cast(str, row["question"])
        valid_answers = cast(list[str], row["valid_answers"])
        img_bytes = cast(bytes, row["img_bytes"])
        family = cast(str, row["family"])
        q_id = cast(str, row["q_id"])
        focus = cast(str, row["focus"])
        img_bytes = cast(bytes, row["img_bytes"])

        regex = cast(Optional[str], row.get("regex"))
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        pred = ask(model, processor, question, img, max_new_tokens)

        total[focus] += 1
        total[q_id] += 1
        total[family] += 1
        total["all"] += 1

        pred_v = match_answer(pred, valid_answers, regex, mode=match_mode)

        with open("logfile.txt", "a", encoding="utf-8") as f:
            f.write(
                f"=====\nGround truth :{valid_answers}\n Resp:{pred}\n:Result:{pred_v}\n\n"
            )
        if pred_v:
            correct[focus] += 1
            correct[q_id] += 1
            correct[family] += 1
            correct["all"] += 1
    accuracy = {}
    for key in correct:
        if total[key] > 0:
            accuracy[key] = correct[key] / total[key]
        else:
            print("[WARNING] Division by zero")
            accuracy[key] = 0.0

    return accuracy
