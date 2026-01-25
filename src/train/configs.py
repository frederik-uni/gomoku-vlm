import os
from typing import Literal

from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig


def init_lora(r, target_modules, modules_to_save):
    return LoraConfig(
        r=r,
        lora_alpha=int(r * 1.5),
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    )


def init_train(
    out,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
):
    return SFTConfig(
        output_dir=out,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        weight_decay=0.01,
        num_train_epochs=epochs,
        save_strategy="steps",
        save_steps=200, # 200 for vision due to dataset-size and iteration speed, 50 for strategy makes more sense
        eval_strategy="steps",
        eval_steps=200, # 200 for vision due to dataset-size and iteration speed, 50 for strategy makes more sense
        save_total_limit=4,
        logging_dir=os.path.join(out, "logs"),
        logging_steps=20,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        packing=False,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        max_length=1024 * 4,
        per_device_eval_batch_size=1
    )


Mode = Literal["visual", "logic"]


def target(mode: Mode):
    if mode == "visual":
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    else:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


def modules(mode: Mode):
    if mode == "visual":
        return ["multi_modal_projector"]
    else:
        return []
