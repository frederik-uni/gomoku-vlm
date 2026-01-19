import argparse
from pathlib import Path

from peft import PeftModel

from ask import init
from eval.logic import eval_vlm_on_parquet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM model on a parquet dataset."
    )

    parser.add_argument(
        "--model-id",
        default="google/gemma-3-4b-it",
        help="HuggingFace model identifier (e.g., 'google/gemma-3-4b-it').",
    )
    parser.add_argument(
        "--match-mode",
        default="exact",
        choices=["exact", "contains", "fuzzy", "lisa", "regex"],
        help="Answer-matching mode. Default: exact.",
    )
    parser.add_argument(
        "--peft",
        default=None,
        type=str,
        help="Path to the PEFT model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--ds",
        default="eval",
        choices=["eval", "eval_reduced", "eval_mini", "test"],
        help="Dataset config in eganscha/gomoku_vlm_ds to evaluate. Default: eval.",
    )
    return parser


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


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    processor, model = init(args.model_id)

    adapter_paths = get_sorted_adapter_paths("./train_output")

    if adapter_paths:
        print(f"Found {len(adapter_paths)} adapters to apply sequentially.")
        for adapter_path in adapter_paths:
            print(f"  -> Applying adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
    else:
        print("No previous adapters found. Starting from base model.")

    result = eval_vlm_on_parquet(
        processor=processor,
        model=model,
        match_mode=args.match_mode,
        max_new_tokens=args.max_new_tokens,
        device="auto",
        dataset=args.ds,
    )

    print(result)


if __name__ == "__main__":
    main()
