import argparse

from peft import PeftModel

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
        choices=["exact", "fuzzy", "lisa", "regex"],
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
        choices=["eval", "eval_reduced", "test"],
        help="Dataset config in eganscha/gomoku_vlm_ds to evaluate. Default: eval.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    # todo: apply lora

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id)

    if args.peft:
        model = PeftModel.from_pretrained(model, args.peft)

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
