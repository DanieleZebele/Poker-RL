# pokerlab

A No-Limit Texas Hold'em engine, built as the foundation for a future
reinforcement-learning poker project.

The long-term plan has four parts:

1. **Game engine** — configurable players, stacks and blinds, with a full
   hand-history log of every action. *(done)*
2. **Players** — a manual (human, terminal-driven) player and scripted bots
   with a few preset strategies. *(done)*
3. **Reinforcement learning** — training scripts where an agent learns to
   play against the bots or against itself (self-play). *(not started —
   see `src/pokerlab/rl/env.py` for the planned interface)*
4. **Live table recognition** — identifying cards and players from a
   screen capture of a poker application. *(not started — see
   `src/pokerlab/vision/reader.py` for the planned interface)*

A simple Tkinter GUI for playing/testing live (instead of the terminal) is
also available — see "Playing with the GUI" below.

## Setup

Requires Python 3.13+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If activation fails with `... esecuzione di script è disabilitata` /
`running scripts is disabled on this system`, PowerShell's default
execution policy is blocking it. Either allow it for just this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

or skip activation entirely and call the venv's executables directly:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pokerlab.cli.play --list-bots
```

## Playing with the GUI

```powershell
poker-gui
```

Opens a setup screen (players, stack, blinds, hands, bot selection --
accepts the same catalog keys and `custom:...` spec as `--bots` below) and
then a live table screen with clickable action buttons for your seat.

## Playing a session (terminal)

```powershell
# see the available bot personalities and their difficulty (1 = weakest, 5 = strongest)
poker-play --list-bots

# 6 players, one of them you (seat 0), the rest bots
poker-play --players 6 --stack 200 --sb 1 --bb 2 --hands 10 --human-seats 1

# pick which bots fill the non-human seats (cycled if there are more seats than keys)
poker-play --players 4 --human-seats 1 --bots rock,shark,maniac

# or tune a one-off bot's own probabilities instead of using a preset
poker-play --players 2 --bots "custom:tightness=0.1;aggression=0.9;bluff_frequency=0.4;size_variance=0.7"

# fully unattended bot-only run, reproducible via --seed
poker-play --players 9 --hands 500 --human-seats 0 --seed 42
```

Each session writes a hand-by-hand log to `hand_histories/session_<id>.jsonl`
(one JSON object per line, one line per hand).

## Running the tests

```powershell
pytest -v
pytest --cov=pokerlab   # with coverage
ruff check .            # lint
```

## Project layout

```
src/pokerlab/
  cards/       Card, Suit, Rank, Deck
  evaluator/   hand-strength evaluation (5-7 cards -> best 5-card hand)
  engine/      GameConfig, betting rules, side pots, Table orchestration,
               hand-history read/write
  players/     Player interface, ManualPlayer, ScriptedBot strategies, GuiPlayer
  cli/         `poker-play` command-line entrypoint
  gui/         `poker-gui` Tkinter desktop app
  rl/          (stub) future self-play training environment
  vision/      (stub) future live screen-capture card/player recognition
tests/
  unit/        engine, evaluator, players, hand history
  integration/ full multi-hand sessions with a chip-conservation fuzz test
```

See `CLAUDE.md` for the architectural decisions behind this layout and
pointers on where each future section should be built.
