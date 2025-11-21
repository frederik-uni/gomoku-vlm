# gen_dataset/debug_sample.py
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from gen_dataset.sphinx.core import PROJECT_ROOT, SPHINX_BASE_OUT


def _find_latest_dataset_idx() -> Optional[int]:
    """
    Scan SPHINX_BASE_OUT for subdirectories named 'dataset_XXX'
    and return the largest XXX as an int. If none exist, return None.
    """
    if not SPHINX_BASE_OUT.exists():
        return None

    candidates: list[int] = []

    for d in SPHINX_BASE_OUT.iterdir():
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
    import sys

    # -------------------------------------------------
    # 1) No arguments -> "IDE mode" (Run in PyCharm)
    # -------------------------------------------------
    if len(sys.argv) == 1:
        # Hard-coded defaults for quick debugging
        q_id = "Q700"
        q_row_idx = 0
        dataset_idx = _find_latest_dataset_idx()
        if dataset_idx is None:
            print(f"No dataset_* directories found under {SPHINX_BASE_OUT}")
            sys.exit(1)

        print(
            f"[IDE mode] Using latest dataset_{dataset_idx:03d}, "
            f"q_id={q_id}, q_row_idx={q_row_idx}"
        )
        main(dataset_idx=dataset_idx, q_id=q_id, q_row_idx=q_row_idx)
        sys.exit(0)

    # -------------------------------------------------
    # 2) CLI mode with arguments
    # -------------------------------------------------
    # Usage help
    if sys.argv[1] in ("-h", "--help"):
        print("Usage:")
        print("  python -m gen_dataset.debug_sample QID [ROW_IDX] [DATASET_IDX]")
        print()
        print("Examples:")
        print("  python -m gen_dataset.debug_sample Q600")
        print("  python -m gen_dataset.debug_sample Q600 3")
        print("  python -m gen_dataset.debug_sample Q600 3 23")
        sys.exit(0)

    # First arg: q_id (required)
    q_id = sys.argv[1]

    # Second arg: q_row_idx (optional, default 0)
    if len(sys.argv) >= 3:
        try:
            q_row_idx = int(sys.argv[2])
        except ValueError:
            print(f"Invalid q_row_idx: {sys.argv[2]!r}, must be an integer")
            sys.exit(1)
    else:
        q_row_idx = 0

    # Third arg: dataset_idx (optional, default = latest)
    if len(sys.argv) >= 4:
        try:
            dataset_idx = int(sys.argv[3])
        except ValueError:
            print(f"Invalid dataset_idx: {sys.argv[3]!r}, must be an integer")
            sys.exit(1)
    else:
        dataset_idx = _find_latest_dataset_idx()
        if dataset_idx is None:
            print(f"No dataset_* directories found under {SPHINX_BASE_OUT}")
            sys.exit(1)
        print(f"No dataset_idx provided, using latest dataset_{dataset_idx:03d}")

    main(dataset_idx=dataset_idx, q_id=q_id, q_row_idx=q_row_idx)