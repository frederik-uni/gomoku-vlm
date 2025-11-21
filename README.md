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

## PyGame Controls

- **Mouse click**: Place a stone (as white player)
- **Any key**: Restart game (after game ends)
- **ESC**: Quit game

**Note**: Game states > automatically exported as .npy files to `game_data/` folder > when the game ends.
