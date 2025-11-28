import difflib
import io
import json
from typing import Literal, Optional, cast

import pandas as pd
import regex as re
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
    regex: Optional[str],
    mode: Literal["exact", "fuzzy", "regex"] = "exact",
):
    if regex is None:
        mode = "exact"

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

    for _, row in df.iterrows():
        question = row["question"]
        valid_answers = cast(list[str], row["valid_answers"])
        img_bytes = cast(bytes, row["img_bytes"])
        family = cast(str, row["family"])
        q_id = cast(str, row["q_id"])
        focus = cast(str, row["focus"])
        img_bytes = cast(bytes, row["img_bytes"])

        regex = cast(Optional[str], row.get("regex"))
        regex = cast(Optional[str], row["regex"])

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        inputs = processor(images=img, text=question, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            pred = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

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
