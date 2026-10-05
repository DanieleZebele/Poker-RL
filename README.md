# pokerlab

A No-Limit Texas Hold'em engine, built as the foundation for a future
reinforcement-learning poker project.

The project has five parts:

1. **Game engine** — configurable players, stacks and blinds, with a full
   hand-history log of every action. *(done)*
2. **Players** — a manual (human, terminal-driven) player and trained
   models as opponents (`model:<path>`). *(done)*
3. **Reinforcement learning** — self-play PPO against a pool of previously
   trained models, with a population-wide Elo ranking. *(done — see
   `src/pokerlab/rl/`)*
4. **Live table recognition** — identifying cards, the dealer button, seat
   states, bets, the pot and stacks from a screen capture of a poker
   application. *(done — see `src/pokerlab/vision/`)*

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
accepts the same `model:<path>` specs as `--bots` below) and
then a live table screen with clickable action buttons for your seat.

## Playing a session (terminal)

```powershell
# see the best trained models found
poker-play --list-bots

# 6 players, one of them you (seat 0), the rest bots
poker-play --players 6 --stack 200 --sb 1 --bb 2 --hands 10 --human-seats 1

# pick which models fill the non-human seats (cycled if there are more seats than specs)
poker-play --players 4 --human-seats 1 --bots model:checkpoints/models/<a-model>.pt

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
  players/     Player interface, ManualPlayer, RLAgentPlayer, GuiPlayer
  cli/         `poker-play` command-line entrypoint
  gui/         `poker-gui` Tkinter desktop app
  rl/          self-play PPO, opponent pool, Elo ranking, continuous training loop
  vision/      live screen-capture recognition (cards, dealer, seats, amounts)
tests/
  unit/        engine, evaluator, players, hand history
  integration/ full multi-hand sessions with a chip-conservation fuzz test
```

See `CLAUDE.md` for the architectural decisions behind this layout and
pointers on where each future section should be built.
