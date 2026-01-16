# gen_dataset/debug_sample.py
import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image

def parse_args():
    """cli"""
    parser = argparse.ArgumentParser(
        description="Evaluate a single question sample for a parquet file"
    )
    parser.add_argument(
        "--in", dest="input",
        required=True,
        type=str,
        help="path to the parquet file",
    )
    parser.add_argument(
        "--qid",
        required=True,
        type=str,
        help="qid of the question to be evaluated",
    )
    parser.add_argument(
        "--idx",
        default=0,
        type=int,
        help="idx of the sample to be evaluated for the provided qid. If none is provided the first sample (idx = 0) will be used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Turn strings into absolute Paths
    path = Path(args.input).resolve()

    # Print the Metadata
    print(f"\nEvaluating dataset:")
    print(f"==============================")
    print(f"Parquet file:   {path}")
    print(f"------------------------------")
    print(f"QID:            {args.qid}")
    print(f"idx:            {args.idx}")
    print(f"------------------------------")

    if not path.is_file():
        print("Parquet file does not exist.")
        exit(1)

    # Load parquet
    df = pd.read_parquet(path)

    # Filter by q_id
    subset = df[df["q_id"] == args.qid].reset_index(drop=True)
    if subset.empty:
        print(f"No Questions found for q_id={args.qid}")
        exit(1)

    if args.idx < 0 or args.idx >= len(subset):
        print(f"idx {args.idx} out of range for q_id={args.qid}, "
              f"we only have {len(subset)} samples for this question.")
        exit(1)

    # Pick the row we want
    row = subset.iloc[args.idx]

    # Decode and show image from img_bytes
    img_bytes = row["img_bytes"]
    if img_bytes is None:
        print("\nimg_bytes is None, cannot display image.")
        exit(1)

    img = Image.open(io.BytesIO(img_bytes))
    img.show(title=f"qid={args.qid}, sample={args.idx}")

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
