import os
from pathlib import Path

from transformers import AutoModelForImageTextToText


def init_model(model_id):
    return AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )


def init_save(out):
    base_path = Path(out)
    base_path.mkdir(parents=True, exist_ok=True)
    existing_runs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("lora_")
    ]
    next_run_num = 0
    if existing_runs:
        max_run_num = max(
            [int(d.name.split("_")[-1]) for d in existing_runs if "_" in d.name]
        )
        next_run_num = max_run_num + 1
    new_run_dir = base_path / f"lora_{next_run_num}"
    new_run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"
    return str(new_run_dir)
