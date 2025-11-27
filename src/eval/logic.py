import difflib
import io
import json
from typing import Literal, cast

import pandas as pd
import torch
from PIL import Image

from utils.ai_utils import get_device


def normalize(s: str):
    return s.lower().strip()


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
    mode: Literal["exact", "fuzzy"] = "exact",
):
    # todo: add regex mode

    if mode == "exact":
        return pred in valid_answers

    if mode == "fuzzy":
        return fuzzy_match(pred, valid_answers)

    raise ValueError("Unknown match mode")


def eval_vlm_on_parquet(
    processor,
    model,
    parquet_path: str,
    match_mode: Literal["exact", "fuzzy"] = "exact",
    max_new_tokens=64,
    device: Literal["cpu", "cuda", "auto"] = "auto",
):
    device = get_device(device)
    model = model.to(device)

    model.eval()

    df = pd.read_parquet(parquet_path)

    def ensure_list(x):
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
                raise ValueError("valid_answers JSON list must contain only strings")
            return parsed
        raise ValueError("valid_answers must be list or JSON string")

    df["valid_answers.list"] = df["valid_answers.list"].apply(ensure_list)

    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        question = row["question"]
        valid_answers = cast(list[str], row["valid_answers.list"])
        img_bytes = cast(bytes, row["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        inputs = processor(images=img, text=question, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            pred = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        if match_answer(pred, valid_answers, mode=match_mode):
            correct += 1

    accuracy = correct / total
    return accuracy
