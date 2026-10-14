# Player embedder (study)

Does a network that reads what a player did in the hands already played, and turns it into a few
numbers, tell a policy more about that player than the nine hand-picked statistics of
`engine/stats.py`? This folder is the **offline gate**: build the data, train the embedder, and see
whether its embedding predicts a player's actions on models it never saw, before any reinforcement
learning is spent on it. Nothing here touches the training.

```
hands 0..t-1 of one player -> HandEncoder -> GRU (one step per hand) -> head -> e_t   (before hand t)
```

- `embed_records.py` (pure Python): a played hand -> what an observer sees of each seat: a vector
  per decision (what it faced, what it did), the **showdown it was seen to take** (nothing / shown /
  mucked, and whether it took the pot), and, kept apart as targets, the action class and the strength
  of the player's own cards. The muck is simulated here: winners show, each loser shows with the
  probability of its own (`p_show`).
- `embed_data.py` (numpy): the arrays, shards on disk, the split of models for validation, and the
  three showdown modes (`full`, `winners`, `none`).
- `embed_model.py` (torch): `HandEncoder`, `PlayerEmbedder`, and the `Predictor` that trains it.
- `embed_generate.py`: plays mixed tables of the store's models (random styles on a share of the
  seats) and writes the shards. Forked copies, the result does not depend on `--jobs`.
- `embed_train.py`: trains the embedder and the same predictor without it, and prints the gain by
  number of hands seen and by showdown mode, on **models held out by name**.

```bash
# 1. the hands (20 copies; ~1 s per session of 100 hands)
OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents:studies/player_embedding \
    python studies/player_embedding/embed_generate.py --sessions 2000 --jobs 24 \
    --out checkpoints/studies/player_embedding            # --top 100: only the best rated models

# 2. the embedder, and whether it helps
PYTHONPATH=src:studies/agents:studies/player_embedding \
    python studies/player_embedding/embed_train.py --data checkpoints/studies/player_embedding --steps 5000
```

## What to read in the table

- **gain** is the cross-entropy of the action without the embedding minus with it: positive is
  good. It should be ~0 at hand 0 (nothing was seen yet) and grow with the hands seen.
- **mode `none`** is the worth of the embedding when no card is ever shown: the case of a screen that
  does not read the opponents' cards. If the gain is there only under `full`, the embedder lives off
  the showdowns and will not survive a real table.
- Only the models held out by name count: a gain on the training models is memorising who is who.

## Design notes

- **The embedding before hand t sees hands 0..t-1 only** (`PlayerEmbedder.embeddings` shifts the GRU's
  states by one; `test_the_embedding_before_hand_t_does_not_see_hand_t...`). The hand in progress is
  read by the policy from the actions in front of it.
- **The predictor is given the player's own cards** (private, known in the data), so the embedding is
  only rewarded for what the cards do not explain: the style. At use only the embedding goes to the
  policy.
- **The models see no opponent statistics while the data is generated** (the HUD slots of their
  observation are zero; `--with-opponent-stats` turns them back on). What is measured is each
  model's own style, not how it reacts to the others. The hands are played one by one
  (`TableBank.play_hand` has no tracker) instead of as a session. Pinned by
  `test_by_default_the_models_play_without_seeing_any_opponent_statistics`.
- **The models play with zero opponent statistics** in the generated hands (`--with-opponent-stats`
  turns them back on): the observation carries zeros where the HUD would be, so what the embedder reads
  is how each model plays on its own and not how it reacts to the others. The hands are played one by
  one (`TableBank.play_hand`, no tracker) instead of as a session. Pinned by
  `test_by_default_the_models_play_without_seeing_any_opponent_statistics`.
- **Muck:** every player gets a `p_show` per session (a quarter never show a loss, a quarter always,
  the rest in between) and the training draws the showdown mode per sequence, so no rule of any
  client is assumed.
- **First reading, with the models reading their opponents' statistics** (the generator's old
  behaviour; rerun with the default to measure models that do not react) (800 sessions of 100 hands from the best 150 models, 3,390 sequences, 2,500 steps,
  one seed; the gain is the action cross-entropy without the embedding minus with it, on models held
  out by name): the gain is ~0 at hand 0 and grows with the hands seen, +0.05 to +0.07 nats from
  hand 10 on (about 7% of 0.79), for the **GRU** (+0.057 over all hands) and the **running mean**
  (+0.055) alike, and about half of that for the **transformer** (+0.025). It is the same under
  `full`, `winners` and `none`: the embedding lives off the actions, the showdowns add nothing
  measurable. The noise is ~0.01 (the transformer shows +0.01 at hand 0, where nothing is known).
  Not yet a result: every model was still improving at 2,500 steps, there is one seed, and the
  baseline is *no* embedding, not the nine statistics.

## Not done yet

- The comparison against the existing nine statistics as the baseline (this one compares with
  nothing). It needs the `StatsTracker` fed over the same sequences.
- The held-out *archetypes* (nit, maniac, calling station) of `archetype_study.py`.
- The tracker that carries the state at a table and the new feature version
  (see the TODO in `CLAUDE.md`): only after the gate is passed.
