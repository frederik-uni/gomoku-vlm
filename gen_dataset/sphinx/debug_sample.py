# gen_dataset/debug_sample.py
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gen_dataset.sphinx.core import PROJECT_ROOT, SPHINX_BASE_OUT


def main(dataset_idx: int, q_id: str, q_row_idx: int = 0) -> None:
    dataset_root = SPHINX_BASE_OUT / f"dataset_{dataset_idx:03d}"
    parquet_path = dataset_root / "parquet" / "dataset.parquet"

    print(f"Dataset root:   {dataset_root}")
    print(f"Parquet file:   {parquet_path}")

    if not parquet_path.is_file():
        print("Parquet file does not exist.")
        return

    # Load parquet
    df = pd.read_parquet(parquet_path)

    # Filter by q_id
    subset = df[df["q_id"] == q_id].reset_index(drop=True)
    if subset.empty:
        print(f"No rows found for q_id={q_id}")
        return

    if q_row_idx < 0 or q_row_idx >= len(subset):
        print(f"q_row_idx {q_row_idx} out of range for q_id={q_id}, "
              f"we only have {len(subset)} rows.")
        return

    # Pick the row we want
    row = subset.iloc[q_row_idx]

    # Print parquet row contents (but only size for img_bytes)
    print("\n=== DatasetRow (from parquet) ===")
    for key, value in row.items():
        if key == "img_bytes":
            if value is not None:
                print(f"{key}: <{len(value)} bytes>")
            else:
                print(f"{key}: NONE")
        else:
            print(f"{key}: {value}")

    # Load and print board from board_path
    if "board_path" in row and row["board_path"] is not None:
        board_rel = Path(row["board_path"])
        board_abs = (PROJECT_ROOT / board_rel).resolve()

        print(f"\nBoard path: {board_abs}")
        board = np.loadtxt(board_abs, dtype=int, delimiter=" ")
        print("\n=== Board state ===")
        print(board)
        print(f"\nShape: {board.shape}")
        print(f"Black stones (1): {int((board == 1).sum())}")
        print(f"White stones (2): {int((board == 2).sum())}")
    else:
        print("\nNo 'board_path' column or value is None – cannot show board.")

    # Decode and show image from img_bytes
    img_bytes = row["img_bytes"]
    if img_bytes is None:
        print("\nimg_bytes is None, cannot display image.")
        return

    print("\nOpening image decoded from img_bytes...")
    img = Image.open(io.BytesIO(img_bytes))
    img.show(title=f"q_id={q_id}, sample={q_row_idx}")


if __name__ == "__main__":
    dataset_idx = 20
    q_id = "Q500"
    q_row_idx = 0

    main(dataset_idx, q_id, q_row_idx)
