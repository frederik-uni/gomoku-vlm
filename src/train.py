import os
from io import BytesIO
from pathlib import Path

from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer
from typing_extensions import Literal

from datasets import Dataset, load_dataset


def init_model(model_id):
    return AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )


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
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        weight_decay=0.01,
        num_train_epochs=epochs,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=4,
        logging_dir=os.path.join(out, "logs"),
        logging_steps=20,
        bf16=True,
        optim="adamw_torch",
        packing=False,
        report_to="none",
    )


def get_sorted_adapter_paths(root: str) -> list[str]:
    p = Path(root)
    if not p.exists():
        return []

    lora_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("lora_")]
    if not lora_dirs:
        return []

    # Sort lora_dirs numerically by the index X in "lora_X"
    sorted_lora_dirs = sorted(lora_dirs, key=lambda d: int(d.name.split("_")[-1]))

    adapter_paths = []
    for lora_dir in sorted_lora_dirs:
        adapter_path = lora_dir / "final-adapter"
        if adapter_path.exists() and adapter_path.is_dir():
            adapter_paths.append(str(adapter_path))
    return adapter_paths


def load_our_dataset(file_path: str) -> tuple[Dataset, Dataset]:
    ds = load_dataset(
        "eganscha/gomoku_vlm_ds",
        data_files={
            "train": file_path,
            "eval": "eval/*.parquet",
        },
    )
    ds = ds.select_columns(["question", "img_bytes", "answer"])

    def preprocess_batch(batch):
        formatted_messages = []
        imgs = []
        for i in range(len(batch["question"])):
            images = []

            try:
                img_entries = batch["img_bytes"][i]
                if isinstance(img_entries, list):
                    for b in img_entries:
                        img = Image.open(BytesIO(b)).convert("RGB")
                        imgs.append([img])
                        images.append(
                            {
                                "type": "image",
                                "image": img,
                            }
                        )
                else:
                    img = Image.open(BytesIO(img_entries)).convert("RGB")
                    imgs.append([img])
                    images.append(
                        {
                            "type": "image",
                            "image": img,
                        }
                    )
            except Exception as e:
                print(f"Error decoding images at row {i}: {e}")

            formatted_messages.append(
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
                        "content": images
                        + [
                            {
                                "type": "text",
                                "text": batch["question"][i]
                                .replace(
                                    "You are a vision-language model analyzing Gomoku game positions.",
                                    "",
                                )
                                .strip(),
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": batch["answer"][i]}],
                    },
                ]
            )
        return {"messages": formatted_messages, "images": imgs}

    dst = ds["train"].map(
        preprocess_batch,
        batched=True,
        batch_size=8,
        num_proc=4,
        remove_columns=["question", "img_bytes", "answer"],
        desc="Formatting dataset with messages",
    )
    dse = ds["eval"].map(
        preprocess_batch,
        batched=True,
        batch_size=8,
        num_proc=4,
        remove_columns=["question", "img_bytes", "answer"],
        desc="Formatting dataset with messages",
    )

    return (dst, dse)


Mode = Literal["visual", "logic"]


def target(mode: Mode):
    if mode == "visual":
        return [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    else:
        return [
            "q_proj",
            "v_proj",
            "up_proj",
        ]


def modules(mode: Mode):
    if mode == "visual":
        return ["multi_modal_projector"]
    else:
        return ["vision_tower"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a LoRA SFT model.")

    parser.add_argument("--model_id", type=str, required=True, help="Model ID to load")
    parser.add_argument(
        "--peft", type=Path, required=False, default=None, help="Model ID to load"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./train_output",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--mode",
        default="visual",
        choices=["visual", "logic"],
        help="Target modules for LoRA",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    adapter_paths = get_sorted_adapter_paths(args.output_dir)
    model = init_model(args.model_id)

    if adapter_paths:
        print(f"Found {len(adapter_paths)} adapters to apply sequentially.")
        for adapter_path in adapter_paths:
            print(f"  -> Applying adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
    else:
        print("No previous adapters found. Starting from base model.")

    if args.peft:
        model = PeftModel.from_pretrained(model, args.peft, is_trainable=False)

    new_output_dir = init_save(args.output_dir)
    print(f"New adapter will be saved to: {new_output_dir}")

    final_dir = os.path.join(new_output_dir, "final-adapter")

    dst, dse = load_our_dataset(args.data_file)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dst,
        eval_dataset=dse,
        args=init_train(
            new_output_dir,
            args.num_epochs,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.learning_rate,
        ),
        peft_config=init_lora(args.lora_r, target(args.mode), modules(args.mode)),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    print("Saved to:", final_dir)
