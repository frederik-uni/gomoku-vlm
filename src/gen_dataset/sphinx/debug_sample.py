# gen_dataset/debug_sample.py
import argparse
import io
from typing import Optional
from pathlib import Path

import pandas as pd
from PIL import Image

from gen_dataset.sphinx.core import DEFAULT_SPHINX_OUT_ROOT_PATH, DATASET_PATH_SUFFIX


def _find_latest_dataset_idx(dataset_base_dir: Path) -> Optional[int]:
    """
    Scan dataset_base_dir for subdirectories named 'dataset_XXX'
    and return the largest XXX as an int. If none exist, return None.
    """
    if not dataset_base_dir.exists():
        return None

    candidates: list[int] = []

    for d in dataset_base_dir.iterdir():
        if not d.is_dir():
            continue

        name = d.name  # e.g. "dataset_001"
        if not name.startswith("dataset_"):
            continue

        # Extract the numeric part after "dataset_"
        suffix = name[len("dataset_"):]  # "001"
        try:
            idx = int(suffix)
        except ValueError:
            # skip weird names like "dataset_backup"
            continue

        candidates.append(idx)

    if not candidates:
        return None

    return max(candidates)


def parse_args():
    """cli"""
    parser = argparse.ArgumentParser(
        description="Evaluate a single question sample."
    )
    parser.add_argument(
        "--dataset_root",
        default=str(DEFAULT_SPHINX_OUT_ROOT_PATH / DATASET_PATH_SUFFIX),
        type=str,
        help="root dir where the dataset files are stored. If none is provided the DEFAULT_SPHINX_OUT_ROOT_PATH will be used.",
    )
    parser.add_argument(
        # default assigned later
        "--dataset_idx",
        default=None,
        type=str,
        help="index of the dataset to use. If none is provided the latest one will be used.",
    )
    parser.add_argument(
        "--qid",
        default=str("Q100"),
        type=str,
        help="qid of the question to be evaluated. If none is provided Q100 will be used.",
    )
    parser.add_argument(
        "--sample_idx",
        default=0,
        type=str,
        help="the sample of the particular question to be evaluated. If none is provided the first sample (idx = 0) will be used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Turn strings into absolute Paths
    dataset_root = Path(args.dataset_root).resolve()
    if not args.dataset_idx:
        args.dataset_idx = _find_latest_dataset_idx(dataset_root)
    parquet_path = dataset_root / f"dataset_{args.dataset_idx:03d}" / f"parquet/dataset.parquet"

    # Print the Metadata
    print(f"\nEvaluating dataset:")
    print(f"==============================")
    print(f"Dataset root:   {dataset_root}")
    print(f"Parquet file:   {parquet_path}")
    print(f"------------------------------")
    print(f"Dataset idx:    {args.dataset_idx}")
    print(f"QID:            {args.qid}")
    print(f"Sample idx:     {args.sample_idx}")
    print(f"------------------------------")

    if not parquet_path.is_file():
        print("Parquet file does not exist.")
        exit(1)

    # Load parquet
    df = pd.read_parquet(parquet_path)

    # Filter by q_id
    subset = df[df["q_id"] == args.qid].reset_index(drop=True)
    if subset.empty:
        print(f"No Questions found for q_id={args.qid}")
        exit(1)

    if args.sample_idx < 0 or args.sample_idx >= len(subset):
        print(f"sample_idx {args.sample_idx} out of range for q_id={args.qid}, "
              f"we only have {len(subset)} samples for this question.")
        exit(1)

    # Pick the row we want
    row = subset.iloc[args.sample_idx]

    # Decode and show image from img_bytes
    img_bytes = row["img_bytes"]
    if img_bytes is None:
        print("\nimg_bytes is None, cannot display image.")
        exit(1)

    img = Image.open(io.BytesIO(img_bytes))
    img.show(title=f"q_id={args.qid}, sample={args.sample_idx}")

    # Print parquet row contents (but only size for img_bytes)
    for key, value in row.items():
        dtype = type(value).__name__  # e.g. 'bytes', 'str', 'numpy.ndarray', 'int', etc.

        if key == "img_bytes":
            if value is not None:
                print(f"{key} ({dtype}): <{len(value)} bytes>")
            else:
                print(f"{key} ({dtype}): NONE")
        else:
            print(f"{key} ({dtype}): {value}")
