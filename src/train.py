import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from typing_extensions import Literal


def init_model(model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
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
    Path(out).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DISABLED"] = "true"


def init_train(out, epochs: int, batch_size: int):
    return SFTConfig(
        output_dir=out,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=4,
        logging_dir=os.path.join(out, "logs"),
        logging_steps=20,
        bf16=True,
        optim="adamw_torch",
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )


from io import BytesIO


def latest_valid_checkpoint(root: str):
    p = Path(root)
    if not p.exists():
        return None
    ckpts = [
        d
        for d in p.iterdir()
        if d.is_dir()
        and d.name.startswith("checkpoint-")
        and (d / "trainer_state.json").exists()
    ]
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda d: int(d.name.split("-")[-1]))
    return str(ckpts[-1])


def load_our_dataset(parquet_path: str) -> Dataset:
    df = pd.read_parquet(parquet_path)

    df = df[["question", "img_bytes", "answer"]]

    def preprocess_row(row):
        prompt_content = f"{row['question']}"

        def load_image_from_bytes(img_bytes):
            return Image.open(BytesIO(img_bytes)).convert("RGB")

        return {
            "images": [load_image_from_bytes(row["img_bytes"])],
            "prompt": [{"role": "user", "content": prompt_content}],
            "completion": [{"role": "assistant", "content": row["answer"]}],
        }

    processed_data = df.apply(preprocess_row, axis=1)

    sft_dataset = Dataset.from_list(processed_data.tolist())
    return sft_dataset


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
        return ["lm_head"]


def freeze_lora(model, modules: list[str]):
    for name, param in model.named_parameters():
        if "lora_" in name and any([module in name for module in modules]):
            param.requires_grad = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a LoRA SFT model.")

    parser.add_argument("--model_id", type=str, required=True, help="Model ID to load")
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
        "--num_epochs", type=int, default=6, help="Number of training epochs"
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

    args = parser.parse_args()
    resume_path = latest_valid_checkpoint(args.output_dir)

    model = init_model(args.model_id)

    final_dir = os.path.join(args.output_dir, "final-adapter")
    if os.path.isdir(final_dir) and resume_path:
        model = PeftModel.from_pretrained(
            model, resume_path, is_trainable=True, ignore_mismatched_sizes=True
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=load_our_dataset(args.data_file),
        args=init_train(args.output_dir, args.num_epochs, args.batch_size),
        peft_config=init_lora(args.lora_r, target(args.mode), modules(args.mode)),
    )

    trainer.train()  # resume_from_checkpoint=resume_path)

    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    print("Saved to:", final_dir)
