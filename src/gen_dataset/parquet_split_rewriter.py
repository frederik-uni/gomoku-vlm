import argparse
from pathlib import Path

import pandas as pd


def _validate_split(split: str) -> None:
    if split not in ("train", "eval", "test"):
        raise ValueError(f'Invalid split "{split}". Must be one of: train, eval, test')


def rewrite_parquet_split(in_path: Path, out_path: Path, *, split: str) -> None:
    """
    Read a parquet file, set split for all rows to the provided value,
    and write a new parquet.
    """
    _validate_split(split)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    df = pd.read_parquet(in_path)

    # Ensure split column exists
    if "split" not in df.columns:
        df["split"] = None

    # Rewrite split
    df["split"] = split

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")


def parse_args() -> argparse.Namespace:
    """cli"""
    parser = argparse.ArgumentParser(
        description='Rewrite the entire "split" column of an existing parquet dataset.'
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=str,
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        type=str,
        help="Path to output parquet file",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        help='Split value to write into every row (train|eval|test)',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Turn strings into absolute Paths
    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()

    rewrite_parquet_split(in_path=in_path, out_path=out_path, split=args.split)