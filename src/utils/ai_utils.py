from typing import Literal

import torch


def get_device(preferred: Literal["cuda", "cpu", "auto"] = "auto"):
    if preferred == "cpu" or preferred == "cuda":
        return preferred

    if preferred == "auto":
        print("Using ", "cuda" if torch.cuda.is_available() else "cpu")
        return "cuda" if torch.cuda.is_available() else "cpu"

    raise ValueError(f"Invalid device preference: {preferred}")
