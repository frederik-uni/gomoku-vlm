import os
from io import BytesIO
from pathlib import Path

from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
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


processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="right",
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "right"


def load_our_dataset(file_path: str) -> Dataset:
    ds = load_dataset(
        "eganscha/gomoku_vlm_ds",
        data_files=file_path,
    )["train"]

    ds = ds.select_columns(["question", "img_bytes", "answer"])

    def preprocess_batch(batch):
        formatted_messages = []

        for i in range(len(batch["question"])):
            images = []
            try:
                img_entries = batch["img_bytes"][i]
                if isinstance(img_entries, list):
                    for b in img_entries:
                        images.append(
                            {
                                "type": "image",
                                "image": Image.open(BytesIO(b)).convert("RGB"),
                            }
                        )
                else:
                    images.append(
                        {
                            "type": "image",
                            "image": Image.open(BytesIO(img_entries)).convert("RGB"),
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
        return {"messages": formatted_messages}

    ds = ds.map(
        preprocess_batch,
        batched=True,
        batch_size=8,
        num_proc=4,
        remove_columns=["question", "img_bytes", "answer"],
        desc="Formatting dataset with messages",
    )

    return ds


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
    parser.add_argument("--peft", type=Path, default=None, help="Path to PEFT model")

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

    model = init_model(resume_path)
    if args.peft:
        model.load_adapter(args.peft, adapter_name="visual", is_trainable=False)
        model.set_adapter("visual")

    final_dir = os.path.join(args.output_dir, "final-adapter")
    ##if os.path.isdir(final_dir) and resume_path:
    #    model = PeftModel.from_pretrained(
    #        model, resume_path, is_trainable=True, ignore_mismatched_sizes=True
    #    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=load_our_dataset(args.data_file),
        args=init_train(args.output_dir, args.num_epochs, args.batch_size),
        peft_config=init_lora(args.lora_r, target(args.mode), modules(args.mode)),
        processing_class=processor.tokenizer,
    )
    print(resume_path)

    trainer.train()

    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    print("Saved to:", final_dir)
