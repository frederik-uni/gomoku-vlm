import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from gen_dataset.sphinx.core import DEFAULT_SPHINX_CONFIG_PATH, DEFAULT_SPHINX_OUT_ROOT_PATH


def parse_args():
    """cli"""
    parser = argparse.ArgumentParser(
        description="Runs the dataset runner for all question.toml files in a folder."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SPHINX_CONFIG_PATH),
        type=str,
        help="Path to the sphinx config file.",
    )
    parser.add_argument(
        "--questions_dir",
        required=True,
        type=str,
        help="Folder with question files (e.g. *.toml)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to the base output folder.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    if importlib.util.find_spec("gen_dataset.runner") is None:
        raise SystemExit("Please run `pip install -e .` (in your active venv) from the repo root first.")

    args = parse_args()
    # Turn strings into absolute Paths
    config_path = Path(args.config).resolve()
    questions_dir_path = Path(args.questions_dir).resolve()
    out_base_path = Path(args.output).resolve()

    if not questions_dir_path.is_dir():
        raise SystemExit(f"--questions_dir must be a directory, got: {questions_dir_path}")

    files = sorted(questions_dir_path.glob("*.toml"))
    if not files:
        raise SystemExit(f"No \"*.toml\" files in {questions_dir_path}")

    for f in files:
        # / out /
        #       q1 /
        #           parquet / dataset.parquet
        #           images / ...
        #           states / ...
        #       q2 /
        run_out = out_base_path / f.stem
        run_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running for {f.name} -> {run_out} ===")
        subprocess.run(
            [sys.executable, "-m", "gen_dataset.runner",
             "--config", str(config_path),
             "--questions", str(f),
             "--output", str(run_out)],
            check=True
        )