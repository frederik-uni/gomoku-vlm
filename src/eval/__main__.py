import argparse

from eval.logic import eval_vlm_on_parquet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM model on a parquet dataset."
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model identifier (e.g., 'google/paligemma-3b').",
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Path to the parquet file to evaluate.",
    )
    parser.add_argument(
        "--match-mode",
        default="exact",
        choices=["exact", "fuzzy"],
        help="Answer-matching mode. Default: exact.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    # todo: apply lora

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id)

    # todo: download parquet_path

    result = eval_vlm_on_parquet(
        processor=processor,
        model=model,
        parquet_path=args.parquet_path,
        match_mode=args.match_mode,
        max_new_tokens=args.max_new_tokens,
        device="auto",
    )

    print(result)


if __name__ == "__main__":
    main()
