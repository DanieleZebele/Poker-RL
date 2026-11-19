# Belief network: the range of every opponent, from what the table saw

The plan for version 4 of the project (branch `v4`): the policy is to read, for every opponent, a
vector describing the range of cards it believes that opponent holds, built from public information
only. This study is the offline gate: does a network trained on self-play (where every hole card is
known) place the real hand better than "any two cards", and by how much?

## Pipeline

    # 1. play mixed tables of store models (training-like: top 100 + random, styles, HUD on)
    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/belief python studies/belief/belief_generate.py \
        --sessions 4000 --jobs 44 --out checkpoints/studies/belief/data
    # 2. hands -> arrays
    PYTHONPATH=src:studies/belief python studies/belief/belief_data.py --data checkpoints/studies/belief/data --jobs 40
    # 3a. range from the current hand (+ HUD): arms full / no-tokens / no-hud
    PYTHONPATH=src:studies/belief python studies/belief/belief_train.py --arm full --out checkpoints/studies/belief/runs/full
    # 3b. with a memory of the target's past hands: arms hud / memory / both / none
    PYTHONPATH=src:studies/belief python studies/belief/belief_history.py --arm memory --out checkpoints/studies/belief/history/memory

- `belief_records.py` (pure Python): a hand as one token per action record (street, actor's
  position, action class, eight numbers on the spot), the samples (at every voluntary action, every
  other player still in: what does it hold?), the statistics every seat had (a session replayed
  through a fresh `StatsTracker`, pinned equal to what the table showed).
- `belief_model.py`: `BeliefNet` scores the 1326 combos in the space of the equity encoder `phi`
  (`logit(c) = <A h, psi(phi(c, board), class(c))>`), the observer's known cards masked;
  `PlayerMemory` (a GRU per past hand, the player's cards only when shown at a showdown, its result;
  a GRU over the hands in order); `range_embedding` (the range's mean and spread in `phi` space:
  the vector the policy is to read; on the true hand alone it is `phi` of that hand, the oracle).
- `belief_export.py`: a `memory`-arm run as the file models of feature version 5 are built with
  (`belief_evaluator/belief-memory-v2.pt`, `[network] belief_model`); the network in play is
  `src/pokerlab/rl/belief.py`, and the tokens are `src/pokerlab/rl/belief_inputs.py`, which
  `belief_records.py` imports, so the network is fed in play what it was trained on.
- Tests: `tests/unit/test_study_belief.py` (the memory's leak test -- a hand's state knows nothing of
  that hand -- is the one to keep).

## Reading the report

`CE` the cross-entropy of the true combo (nats), `unif` the uniform range over the combos the
observer cannot rule out, `bit` the gain over it, `range` the size the network believes (`exp` of
its entropy, in combos), `rank` where the true combo sits in it (0 the likeliest, 0.5 chance).
`ha agito` / `non ancora`: the target has / has not chosen anything yet in this hand (a player who
has not spoken is "any two cards": 0 bits, rightly). `visto N`: hands of the target the table has
seen in the session.

## Results

See the "Belief network" section of `CLAUDE.md`.
