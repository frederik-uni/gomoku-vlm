# Gomoku VLM Training Project

## Installation
```bash
git clone https://github.com/frederik-uni/gomoku-vlm.git
cd gomoku-vlm
pip install -e .
```
or 
```bash
pip install "git+https://github.com/frederik-uni/gomoku-vlm.git"
```
```
or 
```bash
pip install "git+https://github.com/frederik-uni/gomoku-vlm.git"
```

## Setup
```bash
git clone https://github.com/frederik-uni/gomoku-vlm.git
cd gomoku-vlm
python3 -m venv venv && source venv/bin/activate
pip install -e .
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

## Downloading the Dataset
### Make sure the hf CLI is installed
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```
### Source bashrc
```
source ~/.bashrc
```
### Download Dataset
```
hf download eganscha/gomoku_vlm_ds --repo-type=dataset
```

## Dataset Generation

### Single questions.toml file
```bash
usage: python -m gen_dataset.runner [-h] [--config CONFIG] [--questions QUESTIONS] [--output OUTPUT] [--no_gen_subfolder] [--no_rand_img]

Simulate multiple Gomoku games, generate all configured perception and strategy questions, assign train/eval/test splits, and write a single dataset.parquet file plus images.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path where the config file is stored
  --questions QUESTIONS
                        Path where the question text for the questions is stored.
  --output OUTPUT       Path where parquet file will be stored
  --no_gen_subfolder    Do not generate dataset_NNNN subfolder under output folder.
  --no_rand_img         Do not add randomness to the images (e.g. discoloration, rotation, etc.).
```

### Batch question generation for folder containing *.toml files
```bash
usage: python -m gen_dataset.runner [-h] [--config CONFIG] --questions_dir QUESTIONS_DIR --output OUTPUT [--no_gen_subfolder] [--no_rand_img]
example: python -m gen_dataset.batch_runner --config sphinx_config.toml --questions_dir question_datasets/basic_visual_strategy_split/ --output parquets/ --no_gen_subfolder --no_rand_img

Runs the dataset runner for all question.toml files in a folder.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the sphinx config file.
  --questions_dir QUESTIONS_DIR
                        Folder with question files (e.g. *.toml)
  --output OUTPUT       Path to the base output folder.
  --no_gen_subfolder    Do not generate dataset_NNNN subfolder under output folder.
  --no_rand_img         Do not add randomness to the images (e.g. discoloration, rotation, etc.).
```

### Rewrite the entire split column of an existing parquet dataset
```bash
python -m gen_dataset.parquet_split_rewriter -h
usage: parquet_split_rewriter.py [-h] --in IN_PATH --out OUT_PATH --split SPLIT
example: python -m gen_dataset.parquet_split_rewriter --in out/batch/run_1/strategy_questions/parquet/dataset.parquet --out out/batch/run_1b/strategy.parquet --split train

Rewrite the entire "split" column of an existing parquet dataset.

options:
  -h, --help      show this help message and exit
  --in IN_PATH    Path to input parquet file
  --out OUT_PATH  Path to output parquet file
  --split SPLIT   Split value to write into every row (train|eval|test)
```

### Combine Parquets inside a folder into one .parquet file
```bash
python -m gen_dataset.combine_parquets -h
usage: combine_parquets.py [-h] --in_dir IN_DIR --out OUT
example: python -m gen_dataset.combine_parquets --in_dir out/batch/run_1b/ --out out/batch/run_1c/combined.parquet

Combine all .parquet files in a folder into one parquet file.

options:
  -h, --help       show this help message and exit
  --in_dir IN_DIR  Folder containing parquet files (searched recursively).
  --out OUT        Output parquet file path.
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

## Adjusting the Dataset

### sphinx_config.toml
- **Avoiding draws**: Due to the fact that the algorithmic bots are too good, matches often end in draws. To avoid this adjust the `max_simulation_attempts` variable. This adjusts how many times to re-simulate per round to get a game that did not end in a draw.


- **Adjusting split-ratios**: Adjust the `split_ratios` variable. `train, eval, test` need to sum = 1.

### sphinx_question.toml
- **Adjusting the number of samples per question**: Adjust the `num_samples_per_question` variable.


- **Adding global context**: To add global context, which will be pre-pended to every question adjust the `context` variable.


- **Adjusting question text**: To change the question text adjust the `text` variable in the question `[questions.QNNN]`.


- **Removing Questions**: Simply delete the entire `[questions.QNNN]` and associated `text`. It will be excluded from the dataset generation process.

## Adding new Focuses & Questions
### Adding a new Focus
- Add a new focus to either `src/gen_dataset/perception/focus`, or `src/gen_dataset/strategy/focus`.


- Give it a valid `qid`. `qid's` for **perception: Q100 - Q9999, strategy: Q10000 - Q19999**.


- `qid's` for a new focus should be incremented by **+100** from the last focus. e.g. `focus_count_black_stones qid = "Q100"`, `focus_count_white_stones qid = "Q200"`.


- Individual `qid's` **for question variations inside the focus** should be incremented by **+1**.


- Append the **Question Variations** to the `per_simulation.py` file.


- Add the corresponding `question text` via the associated `qid` to the `sphinx_questions.toml` file.