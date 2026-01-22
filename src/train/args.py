import argparse
from pathlib import Path


def parse_args():
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
        "--eval_path",
        type=str,
        required=True,
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
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument(
        "--mode",
        default="visual",
        choices=["visual", "logic"],
        help="Target modules for LoRA",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()
    return args
