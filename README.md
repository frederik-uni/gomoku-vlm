# Gomoku VLM Training Project

## Installation
```bash
pip install -e .
```
or 
```bash
pip install "git+https://github.com/frederik-uni/gomoku-vlm.git"
```

## Usage
### Game
```bash
usage: python -m game [-h] [--size SIZE] [--bot {random,ai,none}]

configurable settings

options:
  -h, --help            show this help message and exit
  --size SIZE           Board size (default: 15)
  --bot {random,ai,none}
```

### Dataset Generator
```bash
usage: python -m gen_dataset.runner [-h] [--output OUTPUT] [--config CONFIG]

Simulate multiple Gomoku games, generate all configured perception and strategy questions, assign train/eval/test splits, and write a single dataset.parquet
file plus images.

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Path where parquet file will be stored
  --config CONFIG  Path where the config file is stored
```

### Evaluation
```bash
python -m eval --model-id "Qwen/Qwen2.5-VL-7B-Instruct" --parquet-path ./datasets/initial_eval.parquet
usage: python -m eval [-h] --model-id MODEL_ID --parquet-path PARQUET_PATH [--match-mode {exact,fuzzy}]
                   [--max-new-tokens MAX_NEW_TOKENS]

Evaluate a VLM model on a parquet dataset.

options:
  -h, --help            show this help message and exit
  --model-id MODEL_ID   HuggingFace model identifier (e.g., 'google/paligemma-3b').
  --parquet-path PARQUET_PATH
                        Path to the parquet file to evaluate.
  --match-mode {exact,fuzzy}
                        Answer-matching mode. Default: exact.
  --max-new-tokens MAX_NEW_TOKENS
                        Max new tokens to generate.
```

## PyGame Controls

- **Mouse click**: Place a stone (as white player)
- **Any key**: Restart game (after game ends)
- **ESC**: Quit game

**Note**: Game states > automatically exported as .npy files to `game_data/` folder > when the game ends.
