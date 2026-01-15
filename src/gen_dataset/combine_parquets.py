import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _list_parquet_files(in_dir: Path) -> List[Path]:
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {in_dir}")

    files = sorted(in_dir.rglob("*.parquet"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .parquet files found in: {in_dir}")

    return files


def combine_parquet_files(in_dir: Path, out_path: Path) -> None:
    """
    Combine all .parquet files inside in_dir (recursive) into a single parquet file.
    """
    parquet_files = _list_parquet_files(in_dir)

    dfs: List[pd.DataFrame] = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    print(f"Found {len(parquet_files)} parquet files in {in_dir}")
    print(f"Wrote {len(combined)} rows to {out_path}")


def parse_args() -> argparse.Namespace:
    """cli"""
    parser = argparse.ArgumentParser(
        description="Combine all .parquet files in a folder into one parquet file."
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        type=str,
        help="Folder containing parquet files (searched recursively).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output parquet file path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_path = Path(args.out).resolve()

    combine_parquet_files(in_dir=in_dir, out_path=out_path)