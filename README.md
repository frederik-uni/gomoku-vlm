# Gomoku VLM Training Project

## Installation
```bash
pip install -e .
```
```bash
pip install "git+https://github.com/frederik-uni/gomoku-vlm.git"
```

## Usage
```bash
# Standard (15x15, bot vs. human)
python -m game

# custom board size
python -m game --size 9

# 2-Player mode
python -m game --bot none
```

## PyGame Controls

- **Mouse click**: Place a stone (as white player)
- **Any key**: Restart game (after game ends)
- **ESC**: Quit game

**Note**: Game states > automatically exported as .npy files to `game_data/` folder > when the game ends.
