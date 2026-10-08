# Studies of agents

Tools that measure or study trained models. Nothing in `src/` imports them; they import
`pokerlab` and each other. Run from the project root:

```bash
# how one model plays, against itself (preflop by hand, bluffs, slowplays, examples)
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/agent_study.py --model '#1' --hands 20000 --jobs 16
#   --model <label | path/to/agent.pt | #N>   --players 6 --stack-bb 100   --save hands.jsonl / --load hands.jsonl
#   the report compares the groups of table sizes and of effective stack; --by giocatori|stack
#   adds the whole report for each group

# does it recognise straights (the wheel A-2-3-4-5 too) and flushes? fixed heads-up spots, each straight/flush next to
# a twin with the combination broken, leading and facing a bet; also the equity network's own estimate and a probe of its encoder
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/agent_study.py --made-hands --model '#1' --per-class 400 --jobs 16

# does it play differently against different HUDs? the same states, only the opponents' statistics changed
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/hud_study.py --model '#1' --decisions 20000 --jobs 16

# against opponents pushed into archetypes (nit, maniaco, calling station): does it read their real HUD, does it pay?
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/archetype_study.py --model '#1' --strength 2 --hands 60000 --jobs 16

# how many hands a session needs before the stronger of two models wins it
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/duel_power.py --model-a '#1' --model-b '#100'

# bb/100 per Elo gap, every anchor against every other
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/elo_bb_grid.py --hands 10000 --jobs 20
```

Tests: `tests/unit/test_study_*.py` (`pytest -m study`).
