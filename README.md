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
```

## Playing with the GUI

```powershell
poker-gui
```

Opens a setup screen (players, stack, blinds, hands, bot selection --
accepts `model:<path>` specs for trained checkpoints) and
then a live table screen with clickable action buttons for your seat.

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
  players/     Player interface, RLAgentPlayer, GuiPlayer
  cli/         model discovery and bot building shared by the GUI
  gui/         `poker-gui` Tkinter desktop app
  rl/          self-play PPO, opponent pool, Elo ranking, continuous training loop
  vision/      live screen-capture recognition (cards, dealer, seats, amounts)
tests/
  unit/        engine, evaluator, players, hand history
  integration/ full multi-hand sessions with a chip-conservation fuzz test
```

See `CLAUDE.md` for the architectural decisions behind this layout and
pointers on where each future section should be built.
