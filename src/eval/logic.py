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
import torch
from PIL import Image

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


def ask_lisa(question1: str, question2: str) -> bool:
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
                    "content": """You are an LLM judge. You are supposed to evaluate the performance of another LLM model.
                    Your task is to decide whether answer2 matches ANY valid ground-truth answer in answer1.

                    You will be given:
                    - answer1: a list of valid ground-truth answers (strings). If answer2 matches ANY one item, the result is yes.
                    - answer2: the candidate model answer (string)
                    First, evaluate carefully, step by step, whether answer2 corresponds to any valid answer in answer1. Use this reasoning to reach your decision.

                    In the final line, output ONLY ONE word: 'yes' if answer2 corresponds to any of the answers in the ground truth, or 'no' otherwise. Do not include anything else in the final line.

                    Matching policy (be strict to avoid false positives):
                    1) Exact match always counts.
                    2) Do NOT count as a match if answer2:
                       - is merely related, plausible, but not equivalent;
                       - mixes multiple answers where any part conflicts with the matched ground-truth item;


                    Decision rule:
                    - If there exists at least one ground-truth item that answer2 matches under the policy above, output "yes".
                    - If you are not highly confident that answer2 matches at least one ground-truth item, output "no".

                    Process:
                    - Compare answer2 against each item in answer1.

                    Output format:
                    On the final line output ONLY one word:
                    yes
                    or
                    no

              """
                    + question1
                    + "\n\n"
                    + question2,
                }
            ],
        }
        response = requests.post(url, json=data, headers=headers)
        print(response.json())

        data = response.json()["choices"][0]["message"]["content"]

        return is_yes(last_line(str(data)))
    except:
        print("error")
        return ask_lisa(question1, question2)


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
    mode: Literal["exact", "fuzzy", "regex", "lisa"] = "exact",
):
    if regex is None and mode == "regex":
        mode = "exact"
    if mode == "lisa":
        return ask_lisa(str(valid_answers), pred)
    if mode == "exact":
        return pred in valid_answers
    if mode == "fuzzy":
        return fuzzy_match(pred, valid_answers)
    if mode == "regex" and regex is not None:
        pattern = re.compile(regex)
        return any(pattern.fullmatch(ans) for ans in valid_answers)

    raise ValueError("Unknown match mode")


def eval_vlm_on_parquet(
    processor,
    model,
    parquet_path: str,
    match_mode: Literal["exact", "fuzzy", "regex"] = "exact",
    max_new_tokens=64,
    device: Literal["cpu", "cuda", "auto"] = "auto",
) -> dict[str, float]:
    device = get_device(device)
    model = model.to(device)

    model.eval()

    df = pd.read_parquet(parquet_path)

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
        question = "<start_of_image>\n" + row["question"]
        valid_answers = cast(list[str], row["valid_answers"])
        img_bytes = cast(bytes, row["img_bytes"])
        family = cast(str, row["family"])
        q_id = cast(str, row["q_id"])
        focus = cast(str, row["focus"])
        img_bytes = cast(bytes, row["img_bytes"])

        regex = cast(Optional[str], row.get("regex"))

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        inputs = processor(images=[img], text=question, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            prompt_len = inputs["input_ids"].shape[1]

            generated_only = output_ids[:, prompt_len:]
            pred = processor.batch_decode(generated_only, skip_special_tokens=True)[0]

        total[focus] += 1
        total[q_id] += 1
        total[family] += 1
        total["all"] += 1

        if match_answer(pred, valid_answers, regex, mode=match_mode):
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
