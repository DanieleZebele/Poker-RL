# pokerlab — notes for future Claude Code sessions

## What this project is

A No-Limit Texas Hold'em engine built as the foundation for a future
reinforcement-learning poker project. Four planned sections:

1. **Game engine** (`src/pokerlab/engine/`, `cards/`, `evaluator/`) — **implemented and tested.**
   Configurable players (2-9), stacks, blinds; full betting logic including
   side pots and the min-raise/short-all-in edge case; JSONL hand history.
2. **Players** (`src/pokerlab/players/`) — **implemented and tested.**
   `ManualPlayer` (terminal input). There is no hand-coded bot catalog any
   more (removed — see "Heuristic bots, removed" below): every non-human
   seat, in the CLI and the GUI alike, is a trained model given by
   `model:<path>`.
3. **RL training** (`src/pokerlab/rl/`, `players/rl_agent.py`) —
   **implemented and tested.** Observation encoding, discrete action space +
   legal-action mask, `RLAgentPlayer`, trajectory collection with GAE, a
   PyTorch actor-critic, self-play PPO with an opponent pool, checkpointing,
   evaluation in bb/100, a Gym-shaped `TablePokerEnv`, and the `poker-train`
   CLI. Requires the `rl` extra. See "RL" below.
4. **Live table vision** (`src/pokerlab/vision/`) — **step 1 done, recognition
   not started.** Screen capture (`mss`), a mouse-drag region selector and the
   zones saved to `vision_data/regions.json` exist, with a screen of their own
   ("Collect vision data" on the main menu) that collects labelled card crops;
   nothing reads a card yet. Requires the `vision` extra
   (opencv-python/numpy/mss). See "Vision" below.
5. **GUI for live testing** (`src/pokerlab/gui/`, `players/gui.py`) —
   **implemented.** A Tkinter desktop app (`poker-gui`): a setup screen
   (players/stack/blinds/hands/bot selection, offering trained models the
   same way the CLI does) and a table screen (seats, board, pot, hole cards,
   action buttons, hand-by-hand log). See "GUI" below.

## Key architectural decisions (and why)

- **Chips are always `int`, never `float`.** Pot-splitting with floats risks
  rounding drift that breaks the "total chips in play never changes"
  invariant. That invariant is directly tested (see Testing below) and is
  the single most valuable regression guard in this codebase.
- **The hand evaluator (`evaluator/evaluator.py`) is hand-rolled, not a
  library.** This was an explicit user choice over using `treys`. It brute-
  forces every 5-card combination out of 5-7 cards (`itertools.combinations`)
  rather than using lookup tables, trading a little speed for something
  easy to verify by hand. Treat any future change to this file with extra
  caution and re-run `tests/unit/test_evaluator.py` (18 cases covering the
  wheel straight, kickers, and best-5-of-7 selection) before trusting it.
- **Hand history is JSON Lines**, one hand per line, with a `schema_version`
  field from day one specifically so the schema can evolve later (e.g. once
  an RL data pipeline needs new fields) without breaking old log files.
  Serialization is hand-written in `history.py` (not `dataclasses.asdict`),
  because `asdict` recursively flattens `Card` (a dataclass) into
  `{rank, suit}` dicts instead of the compact `"Ah"` string form — this bit
  once already, see `_to_jsonable`'s docstring.
- **`Player.act(observation, legal_actions) -> Action` is the only
  extension point.** `Table` never branches on what kind of `Player` it's
  talking to. This is *the* reason a `GuiPlayer` or a future `RLAgentPlayer`
  can be added later as a new file in `players/` without touching
  `engine/` at all — the engine only ever calls `act()`.
- **`Observation` (in `players/base.py`) is a plain, JSON-friendly dataclass,
  not a tensor.** Encoding it into a model-ready array is deliberately left
  to the future `rl/env.py`, not baked into the engine.
- **The min-raise / short-all-in-doesn't-reopen-betting rule** is
  implemented via two pieces of `HandState`: `to_act` (who still needs to
  respond to the current bet level) and `raise_barred` (who has already
  responded and therefore may only call/fold, not re-raise, until a genuine
  full raise clears the bar). This is the trickiest rule in the engine; see
  `engine/betting.py::_handle_new_bet_level` and the four dedicated tests in
  `tests/unit/test_betting_legal_actions.py` before changing betting logic.
- **Big blind option bug (fixed)**: `compute_legal_actions` used to gate the
  aggressive action on `current_bet_to_match == 0`, which is only true for
  a fresh street. Preflop, `current_bet_to_match` starts at the big blind,
  so when everyone just calls around to the BB (`to_call == 0` but
  `current_bet_to_match > 0`), the BB was wrongly offered only fold/check/
  all-in, never raise. Fixed by branching on `current_bet_to_match == 0`
  (offer BET, a fresh open) vs. `> 0` (offer RAISE off the BB's own posted
  bet instead). Regression test:
  `test_big_blind_option_can_raise_when_everyone_just_called`. If a similar
  "why can't seat X raise here" report ever comes up again, check this
  exact branch first — it's the one spot where to_call==0 doesn't imply
  "no bet has happened yet".
- **Engine code never imports from `players/`** (one lazy, function-local
  import in `table.py` is the deliberate exception, to avoid a real import
  cycle) — keeps `engine/` reusable by RL/vision/GUI without dragging in
  CLI or terminal-input concerns.

## Heuristic bots, removed

`players/scripted.py` used to hold a hand-coded bot catalog: five named
personalities (Random, CallingStation, Maniac, Rock, Shark) ranked by
difficulty, four sharing one parametrized implementation
(`make_heuristic_bot`, on tightness/aggression/bluff_frequency/size_variance
axes scaled by how many opponents were still live), plus a `custom:...` spec
for one-off parameter combinations. It has been deleted, at the user's
request, along with every dependency on it, now that the RL training
pipeline has its own population of trained checkpoints to draw opponents
from — the hand-coded bots existed to bootstrap self-play and give the CLI
and GUI something to seat before any model existed, and are no longer the
only way to do either:

- **`poker-play`/`poker-gui`**: every non-human seat is now a trained model,
  given as `model:<path>` (see `cli/play.py::build_players`,
  `discover_trained_models`). With no `--bots` given, the CLI/GUI cycle
  through the best-rated models found across every machine's pool instead of
  a catalog; with no trained models on disk yet, seating a non-human seat now
  raises a clear error rather than falling back to a hand-coded bot.
  `--list-bots` prints those discovered models instead of a catalog.
- **RL training** (`rl/train.py`, `rl/rollout.py`): `TrainConfig.bot_keys`,
  `SelfPlayTrainer.evaluate`/`evaluate_duplicate` (bb/100 against one
  catalog bot), and `rollout.scripted_opponent` are gone.
  `OpponentPool`'s only opponent source now is previously trained pool models
  (`extra_opponents`) — the run's own frozen snapshots were a second source
  until they too were removed, see "The run's own past selves are not in the
  training field"; with none available (the very first run ever), it
  simply seats the learner in every chair — `OpponentPool.sample` already
  returned `None` (meaning "seat the learner") whenever its candidate list
  was empty, so pure self-play was always the graceful fallback, not a new
  code path. `poker-train --eval-opponent`/`--eval-bot` are gone; evaluation
  is unconditionally against the ranked pool now (`evaluate_against_pool`),
  which is what already fed the pool's Elo ratings and is the modern
  replacement for a bot-based win-rate check.
- **The pool registry** (`rl/pool_registry.py`): `PoolRegistry.
  ensure_scripted_members`/`scripted()` and the `SCRIPTED` member kind are
  gone; `fill_slots` no longer pads a short pool with catalog bots, only by
  cycling models. A `registry.json` saved before this change may still
  contain `kind: "scripted"` entries; loading one is harmless (the field is
  simply ignored by anything that still reads it), but such an entry can no
  longer be seated — `registry_opponents` skips it via `on_skip`, the same
  way it skips an unreadable checkpoint.
- **Engine-level tests**: `test_full_hand_flow.py` and several other test
  files used `make_random_legal_bot`/`make_always_call_bot` purely as cheap,
  deterministic opponents for fuzzing engine correctness (chip conservation,
  action legality) — nothing to do with bot "difficulty". That need didn't
  go away with the catalog, so a minimal `RandomActionPlayer`/
  `AlwaysCallPlayer` pair now lives in `tests/support.py`, test-only and
  decoupled from anything product-facing.

## RL (`rl/`, `players/rl_agent.py`)

Target algorithm is **self-play PPO** (actor-critic). Requires the `rl` extra
(`pip install -e ".[rl]"`).

- **One policy network, not one per street.** The street arrives as a one-hot
  input feature; there is a single shared trunk, one policy head and one value
  head. Four independent per-street networks were explicitly rejected: preflop
  sees ~100% of decisions and the river 10-15%, so splitting the data four ways
  starves the street where mistakes cost most, and — the decisive argument for
  PPO — the reward is terminal-only, so GAE bootstraps `V(s_{t+1})` *across*
  street boundaries; four critics would make that bootstrap hop between
  approximators with independently drifting value scales. A shared trunk with
  four heads is a strict superset that can be added later without retraining
  the trunk; only do it if diagnostics show per-street entropy or value
  collapse. Note there is deliberately no separate street embedding table —
  with a one-hot input, the trunk's first `Linear` *is* that table.
- **`features.py` and `action_space.py` are pure Python** (`list[float]` /
  `list[bool]`, no numpy, no torch). This is the load-bearing decision: it puts
  the highest-bug-density code inside the normal test suite with zero new
  dependencies, and it is why `players/rl_agent.py` can be imported eagerly by
  `players/__init__.py` without dragging torch into the core install. **torch is
  imported in exactly one module, `rl/policy.py`.** A regression here is easy to
  catch: `import pokerlab.players` must leave `torch` out of `sys.modules`.
- **`OBS_DIM` is 480**, defined as the sum of named per-section constants so it
  cannot silently drift. Cards are **6 binary 4x13 planes** (hole, flop, turn,
  river, whole board, hole∪board): a street not yet dealt is an all-zero plane,
  which solves the 0/3/4/5-board-cards problem for free and keeps the door open
  to a `Conv2d` over `(6, 4, 13)` with no change to `encode_observation`.
  Seats are indexed **relative to me** (slot 0 is always me), so the encoding is
  invariant to absolute seat numbering. Action history is aggregated per street,
  **not sequenced**, which loses the order, who did each action, and the
  individual bet sizes — the planned replacement is a flat positional encoding of
  the last 20 actions behind a versioned encoder, not a GRU; see "Observation v2"
  in the TODO section for the measurements and for why the versioning has to come
  first.
- **`encode_observation` takes `big_blind` and `starting_stack` as keyword
  arguments** because `Observation` deliberately carries no table config. Do
  *not* try to recover the blind from the `POST_BLIND` records: a short blind
  posts less than the big blind. `legal_mask` is a *required* argument (echoing
  it into the input helps the value head) — a default would silently produce
  different features at training and inference time.
- **Field size is encoded twice, on purpose**: as aggregate counts (live,
  still-actionable, and all-in opponents, plus the static table size) and as the
  81 per-seat slots, which say *which* seats folded and where they sit relative
  to me. Both are recomputed from the live `Observation` at every decision, so a
  hand naturally tightens and loosens as players fold.
- **Positional value does not follow the seating order from the button**, and
  getting this wrong is easy: postflop action opens on the small blind and
  closes on the button, so the seat *before* the button is second-best while the
  small blind is worst. An earlier version ranked by distance clockwise from the
  button, which scored the cutoff 0.00 and the big blind 0.60 — inverted for
  every seat except the button itself. The feature now ranks by how many seats
  act after me; regression test:
  `test_position_feature_follows_postflop_action_order`.
- **`committed_by_seat` (in `features.py`) rebuilds per-hand totals from the
  action log**, because `SeatPublicInfo` only exposes the *current street's*
  bet. It works because the engine logs `ps.current_bet` *after* the action, so
  the max per (seat, street) summed over streets is the hand total. There is a
  direct test that these sum back to `Observation.pot_size`.
- **11 discrete action bins**: fold, check/call, min-raise, 25/33/50/75/100/150/
  200% pot, all-in. Two rules matter. First, **BET vs RAISE is read off
  `legal_actions`, never inferred from `to_call == 0`** — that is exactly the
  big-blind-option trap documented above, and there is a mirrored regression
  test for it. Second, **out-of-range bins are masked, never clamped**: clamping
  up duplicates the min-raise and clamping down duplicates the shove, and two
  bins meaning one action split the policy's probability mass over a choice that
  does not exist. Colliding raise levels (common in small pots) are deduplicated
  the same way, lowest bin wins. `FOLD` is masked when `to_call == 0` (folding
  for free is strictly dominated) behind `mask_dominated_folds=True`.
- **Training collects trajectories through `Player.act()`, not through a Gym
  `step()`** (`rl/rollout.py`). PPO needs `(s, a, logp, V, r, done)` tuples, not
  a `step()`; `Table` drives hands synchronously and `RLAgentPlayer.on_decision`
  reports each decision as it happens. Zero threads, zero duplicated betting
  logic. For throughput the answer is N `Table` instances in N worker processes,
  not control-flow trickery.
- **`TablePokerEnv` (`rl/env.py`) is for debugging/eval, not training.** It runs
  `Table` on a daemon thread and hands control back through two queues — the
  same pattern as `GuiPlayer`. `close()` is mandatory between episodes (it
  pushes an abort sentinel that unwinds `_QueuePlayer.act()`, then joins), or
  every `reset()` strands a parked thread; there is a test asserting
  `threading.active_count()` is unchanged after repeated resets. Queues are
  drained only *after* the join, so a dying worker cannot write into them.
- **Reward is terminal-only, in big blinds**, `γ = 1.0`, `λ ≈ 0.95`. No potential-
  based shaping: rewarding "won the pot" teaches nit play and rewarding
  hand-strength EV leaks information the agent must not condition on. Chips stay
  `int` inside the engine — the float conversion happens only in `rl/`.
  **Episode = one hand**, and `SelfPlayCollector` resets stacks between hands
  (`rebuy=True`) because `Observation` has no memory of earlier hands, so a
  multi-hand episode would not be Markov w.r.t. the features.
- **Gotcha, encoded as a test**: mask logits with a large *finite* negative
  (`MASK_FILL = -1e9`), never `-inf`. With `-inf`, `Categorical.entropy()`
  computes `0 * -inf = NaN` whenever only one action is legal — which happens
  routinely in poker.
- **A chopped pot legitimately leaves every stack unchanged**, so "a played hand
  must move chips" is *not* a valid invariant (it cost one wrong test
  assertion). The real invariants are chip conservation and per-hand rewards
  summing to zero.
- **The reward scale is not cosmetic** (`SelfPlayCollector.reward_scale`, wired
  from `TrainConfig.reward_scale`, defaulting to `big_blind / starting_stack`).
  With an unscaled big-blind reward and a 200-chip stack, value targets span
  ±100, the squared value loss sits around **10,000** and — against a policy
  loss of order 0.01 — the critic's gradient is all the network ever sees.
  Measured: `value_loss` 9,300 → 14,000 and *climbing* unscaled, vs. a stable
  ~1.1 scaled. `HandTrajectory.reward` stays in big blinds for reporting;
  `DecisionRecord.reward` carries the scaled training signal. If you ever change
  stack size or blinds and training stalls, check this first.
- **PPO reuses the mask stored at rollout time** (`DecisionRecord.legal_mask`),
  it does not recompute it. Recomputing or omitting it would measure
  `pi_new/pi_old` against a different distribution than the one that acted, and
  the gradient would be silently wrong.
- **The run's own past selves are not in the training field, and the whole
  mechanism is gone.** `SelfPlayTrainer` used to freeze a copy of the learner
  every `TrainConfig.snapshot_every` (10) iterations, keep the last
  `max_snapshots` (5) of them in the `OpponentPool`, and give them a
  `snapshot_share` (`--self-share`, default 0.4, and for a while one of the axes
  `poker-loop` swept per worker over {0.0, 0.2, 0.4}) of the opponent seats.
  **All of it — `snapshot_every`, `max_snapshots`, `snapshot_share`,
  `SelfPlayTrainer.snapshot()`, `OpponentPool.add_snapshot` and the sweep axis —
  was removed at the user's request.** The reasoning: a snapshot is a copy of
  the network being trained, so it drifts with it and anchors nothing, which is
  precisely the job the fixed pool exists to do; every seat it took was a seat
  *not* facing an independently trained model, and at 6-max with
  `opponent_probability` 0.5 that was 40% of the 2.5 opponent seats, leaving
  barely 1.5 of 6 facing a real opponent. Now every opponent seat faces a
  previously trained model from the store. Two consequences worth knowing:
  - **`--self-share` is retired on `poker-train`, not deleted** — parsed and
    discarded — because a supervisor started before this change keeps forwarding
    it on every worker it launches. See "Never delete a `poker-train` CLI flag
    while a supervisor is running". On `poker-loop` the flag is gone outright,
    since nothing but a person passes it there and `run.sh` never did.
  - **A parent published before the change still carries `self_share` in its
    metadata**, which is simply ignored: `perturb_hyperparameters` reads the
    axes of `HP_LADDERS`, so a key that is no longer an axis costs nothing and
    inheritance keeps working across the change in both directions.
- **`OpponentPool` fills the seats the learner is not in**, drawing **uniformly**
  from the previously trained pool models it was given (`extra_opponents`, see
  `train.py::registry_opponents`). With none available (an empty store, the very
  first run ever), `OpponentPool.sample` returns `None` and every seat goes to
  the learner — plain self-play is the graceful starting point, not a special
  case. If a curriculum over opponent strength is ever wanted, it belongs in
  `SelfPlayTrainer`, by choosing which pool models to draw from per phase.
  Seats are swapped between hands
  through `SeatProxy` rather than by rebuilding the `Table`, which would reset
  the button and stacks. Exactly one seat is always the learner, so a hand can
  never yield zero training data.
- **Two kinds of saved model, and the distinction matters.**
  `checkpoints/agent.pt` (`--checkpoint`) is the *live* one: weights plus
  optimizer state, **overwritten every save**, and what `--resume` reads.
  `checkpoints/models/<machine>-[<prefix>-]agent-<runid>.pt` is a *published*
  model in the one shared store: **written once, never modified, and only one per
  run** — the run's **latest** checkpoint. Every `--archive-every` iterations
  `main()` overwrites the run's **scratch** file
  (`--scratch-dir/agent-<runid>.pt`, plus a `.json` sidecar holding the rating and
  iteration) in place; only when the run ends is the scratch copy **published**
  (`global_store.publish_model`: a dotted `.partial` copy renamed into place, so a
  reader on another machine never sees half a checkpoint) and registered in the
  global ranking at the rating the learner measured. Publishing once at the end
  is what makes a label mean one set of weights forever — overwriting a model that
  already has a rating history would blend two different networks into one Elo.
  If a run dies before it can publish, the loop's sweep publishes the scratch
  file from the sidecar (see "Continuous training loop"). The name carries the
  machine (`--machine`) and, from `poker-loop`, the generation and worker
  (`--archive-prefix gen0155-w07`), so two hosts can never collide. **The run id
  is only second-resolution**, so genuinely concurrent runs of `poker-train` on
  one machine need distinct `--archive-prefix` *and* `--scratch-dir` (the parallel
  -sweep recipe below does this), or the later one finds its name taken and its
  model is not published. Both `checkpoints/` subtrees are gitignored.
- **The published checkpoint is the run's *last*, not its best-rated.** This is
  deliberate and counter-intuitive. Archiving used to keep whichever checkpoint
  scored highest on `learner_rating`, which comes from a few hundred hands
  against the run's own drawn pool. Across **27,610 real runs** that rating
  called the mid-run model better than the final one **47% of the time** — a coin
  flip — while a duplicate-deck duel of the two *from the same seed* put the
  later one ahead in **all 6 seeds, by +56.7 bb/100 (t = 7.5)**. Selecting on the
  rating captured only 30.1 of those 56.7 bb/100; taking the last captures all of
  it, and a measurement would have to be right ~95% of the time to beat the blind
  rule. The general principle: **a reliable prior beats an unreliable
  measurement of the right quantity.** Swap it only for a measurement good enough
  to also catch the runs where the later model genuinely is worse — the
  benchmark bb/100 is the candidate, being deterministic and scored against
  models never seated in training, once the frozen set is rebuilt.
  Pinned by `test_a_run_publishes_its_last_checkpoint_not_its_best_rated`.
- **Elo is applied pairwise** (`pairwise_elo_delta`): a session's participants
  are scored against each other from their chip deltas, win/loss/draw, and the
  per-pair changes are divided by the number of opponents faced — otherwise
  table size would silently rescale K and a 6-max session would move ratings
  five times as far as a heads-up one. Every update reads the ratings as they
  stood *before* the session, so the result does not depend on iteration order.
  A draw is not a degenerate case here: a chopped pot genuinely leaves two
  stacks equal. **The ranking criterion is deliberately not bb/100**: a single
  evaluation has a measured spread of ~350-570 bb/100 on one fixed model, so
  ranking on it would rank the luckiest; ratings instead *accumulate* one result
  per rated session until they reflect strength rather than one draw from a
  heavy-tailed distribution.
- **Every training run draws its own opponents** from the shared store
  (`rl/training_pool.py::draw_training_pool`), so workers do not all train against
  the same field — runs are diverse by construction, not by accident. The draw
  mixes two sources for `--pool-models` (20) seats: **top** (`--pool-top-share`,
  50%, uniformly from the `--pool-top-n` = 100 best-rated models) and **random**
  (every remaining seat, uniformly from the whole store, which keeps weak and odd
  opponents — never-rated newcomers included — in the mix and stops the field
  being only the current elite). There used to be a third, **fresh**, source with
  a quarter of the seats reserved for the least-played models so newcomers were
  trained against and seen at all; it was removed at the user's request, and those
  seats went to the random draw, which reaches a newcomer in proportion to how
  many there are rather than by reserving anything. The random source is also what
  makes a short top source harmless: with nothing rated yet the top share has
  nobody to draw and the pool still comes out full; with fewer models than seats it
  is short and `fill_slots` cycles it. Seeded by
  `--seed`, so a given seed reproduces its draw. The ranking it reads is the
  `registry.json` snapshot (one file; see "Population-wide Elo and pruning"),
  because every worker of a generation draws at the same moment.
- **The drawn opponents are frozen reference points, in memory only.** The draw
  becomes an in-memory `PoolRegistry` (`build_training_registry`) whose members are
  *copies* marked `frozen=True`: `evaluate_against_pool` scores the learner
  against them, and `record_session_with_ratings` never moves a frozen member's
  rating, so **a training run never edits anyone else's rating** and nothing it
  does to the registry is ever saved (there is no `registry.save()` in `main()`).
  Only the learner's rating moves (`SelfPlayTrainer.learner_rating`). That rating
  — measured against opponents on the global scale — is what
  `rate_against_benchmark` starts from and therefore feeds the number a published
  model enters the global ranking with. **It used to be stuck at the baseline,
  and this was measured**: at the flat K a session moves a rating by at most 4
  points, so 17 live workers were all reading between 1469 and 1534 with bb/100
  anywhere from −686 to +47. The learner is now rated through
  `DEFAULT_K_SCHEDULE` on its own session count, which is what makes the number
  mean something inside a run — see "K follows a 10-step staircase".
  - **It starts at the parent's rating, not at 1500** (`train.inherited_rating`,
    read off the resumed checkpoint's `publish_rating` metadata, falling back to
    `pool_rating` and then to the baseline). A child is its parent plus some
    training, not an unknown quantity, so starting at 1500 would spend the run's
    first rated sessions re-discovering where its own lineage already sits. The
    *session count* still starts at zero, and the two together are exactly what
    the schedule's gain expects: a starting point worth something, and no claim to
    having earned it. Every model published since the metadata existed carries
    `publish_rating`, so this works on the store as it stands with no migration.
  The live learner is never itself a member: it changes every
  iteration, so persisting it would rate a moving target, and that is exactly why
  its K has to be handed to the registry rather than looked up. With an *empty* store
  (the very first run) there is nobody to evaluate against: the run is plain
  self-play against the current policy, skips evaluation, and still saves and
  publishes.
- **`evaluate_against_pool()` produces the learner's rating.** It plays
  `--eval-sessions` (**10**) rated sessions of `SESSION_HANDS` (**1000**) hands
  each, re-drawing the opponents from the pool and re-seating the learner before
  every one. At `--eval-every 100` over a 1000-iteration run that is **100 rated
  sessions**, which then continue into the round against the frozen anchors on one
  count. Seats are swapped through `SeatProxy` (made public for this) so one
  `Table` lives across the whole round and the button keeps rotating; rebuilding
  it per session would reset the button to the same seat every time and hand
  whoever sits there a systematic positional edge.
  - **`--eval-hands` and `--eval-rotate-every` are gone, deleted rather than
    retired.** They asked for a hand count that had to be a multiple of a block
    size, which said nothing about what a reader wants to know — how many rated
    results a round produces. A round is now a session count, full stop. Deleting
    them breaks every running supervisor at its next generation (see "Never delete
    a `poker-train` CLI flag while a supervisor is running"); it was done at the
    user's explicit decision, with a fleet reset immediately after.
  - **The session length is 1,000 hands everywhere and that is not a coincidence.**
    `train.SESSION_HANDS`, `benchmark.DEFAULT_SESSION_HANDS` and
    `global_arena.DEFAULT_HANDS_PER_GAME` are all 1,000, pinned together by
    `test_a_session_is_a_thousand_hands_everywhere`. The Elo scale is *defined* by
    how often a session of that length picks the stronger model (measured: 69.7%
    for a 136-point gap, against the 68.6% the logistic predicts), so a round of a
    different length would be a different scale silently sharing the same numbers.
  - **What the validation rating is not.** It is measured against *live* models
    whose own ratings carry error, and it rates a moving target: the weights change
    between rounds, so a session played at iteration 100 scored a network that no
    longer exists, and the duplicate-deck duel puts a run's final model **+56.7
    bb/100** ahead of its mid-run self. That is why the validation is deliberately
    the *smaller* half of a run's rated sessions — 100 against 500 — and why the
    published rating is earned against the pinned anchors. Measured: at 100 + 500
    about **21%** of the final rating comes from the validation phase; a 300 + 300
    split would make it 53%, which is why that split was considered and dropped.
  - **The block was 25 hands and the whole evaluation 500, and both were far too
    small to mean anything.** At 25 hands Elo sees the *sign* of a chip delta whose
    spread is ~±580 bb/100 against a real gap of 10-40, so the stronger side
    finished ahead ~52% of the time — which equilibrates ~14 rating points above
    the pool instead of the ~150 that strength deserves, against an accumulated
    noise of ~46. The rating was a quarter signal, three quarters coin flip, and
    the bb/100 carried a 95% interval of ±255. At 1000-hand blocks the same
    strength shows as ~91 points against the same ~46 of noise, and 10,000 hands
    put the bb/100 at ±57. **Session length is the only lever that moves the
    equilibrium** — a lower K shrinks the jitter around it and cannot touch it.
    Exactly the argument on `global_arena.DEFAULT_HANDS_PER_GAME`.
  - **`--eval-every` went 50 → 250 → 100.** The middle value was a deliberate
    trade — four reliable readings a run beat ten unreadable ones — made when a
    round was 10,000 hands and the rating it produced was the *only* thing
    refining the learner before publication. It is back to 100 because the
    arithmetic changed: the round is the same 10,000 hands but the rating is now
    built over 600 sessions end to end, so what the interval buys is *rated
    sessions in the count*, and 100 of them is what the schedule's gain was sized
    against. Ten rounds a run, ~10% of a run's time. **`run.sh` must not override
    it** — it used to pass `--eval-every 250 --eval-hands 10000` explicitly, which
    would have quietly kept the fleet on four rounds a run; see the note on
    `--inherit-fraction` for the day that class of bug cost.
  - **The last iteration always evaluates**, whatever the interval. Without it
    a run whose iteration count is not a multiple of
    `--eval-every` ends with no final reading at all — and that reading is also
    the rating the model is published with whenever the benchmark round cannot
    run.
- **`fill_slots(n)` always returns exactly `n` seat fillers**: the ranked models,
  then the models again by cycling once there are fewer than `n`. A repeat is the
  same `Opponent` object seated twice — it weights that model more heavily in the
  draw and does not create a second copy of the weights (`registry_opponents`
  loads each distinct member once).
- **`pool_registry.py` is pure Python — no torch**, the same load-bearing
  argument as `features.py`/`action_space.py`: the Elo arithmetic and the code that
  decides *who plays and who is removed* belong in the ordinary test suite with no
  extra dependency, and `rl/training_pool.py` and `rl/global_store.py` follow suit.
  The torch glue that turns a member back into an `Opponent` lives in `train.py`
  as `registry_opponents`. Tests: `tests/unit/test_rl_pool_registry.py`,
  `test_rl_training_pool.py` (no torch) plus the pool section of `test_rl_ppo.py`.
- **`FEATURE_VERSION` (in `features.py`) must be bumped whenever the *meaning*
  of a feature changes, even when `OBS_DIM` does not.** The position fix
  reordered what slot 340 means while keeping the vector exactly 480 long, so a
  dimension check alone would have happily loaded a model that reads the wrong
  thing from every slot. Checkpoints record it and `check_compatible` rejects a
  mismatch. Checkpoints also record `hidden`/`num_layers`, so
  `build_model_from_checkpoint` can rebuild an archived agent at whatever shape
  it was trained with.
- Healthy diagnostics from a real run: `value ~1.0-1.3` and flat, `kl ~0.005-0.01`,
  `clip 0.05-0.20`, `entropy` drifting slowly down from ~1.7 (ln 11 ≈ 2.4 is the
  uniform-over-all-bins ceiling). A climbing value loss or `kl` above ~0.05 per
  iteration means the step size is too large.
- Tests: `tests/unit/test_rl_action_space.py`, `test_rl_features.py`,
  `test_rl_rollout.py`, `test_rl_opponent_pool.py` need no extra dependencies;
  `test_rl_policy.py` and `test_rl_ppo.py` open with
  `pytest.importorskip("torch")`. `test_rl_ppo.py` includes the
  gradient-direction check (a positive advantage must raise the taken action's
  probability, a negative one must lower it) — that is the test that would
  catch a sign error in the surrogate. The real gate is
  `tests/integration/test_rl_env_flow.py` — 2-9 players × 5 seeds with every
  seat an `RLAgentPlayer` on a uniform-over-unmasked policy, asserting no
  `IllegalActionError` ever (the action-mapping analogue of chip conservation),
  chip conservation, and zero-sum rewards.

## Continuous training loop (`rl/loop.py`)

`poker-loop` runs training indefinitely: N worker **processes** (not threads --
the engine is pure Python and CPU-bound, so threads would serialise on the GIL
and run at the speed of one), organised into generations.

- **One generation** = sweep the residues of any interrupted run, pick which
  workers inherit weights and from whom and draw each one's hyperparameters,
  launch N `poker-train` subprocesses, wait for them (by polling, not by
  blocking -- see the Elo fill-in phase, where a worker deliberately does not
  exit until this supervisor releases it), sweep again (which also publishes any
  model a crashed worker did not), benchmark, print the leaderboard. Then
  repeat. **Nothing is seeded,
  merged, re-ranked or trimmed by the loop**: each worker draws its own opponents
  from the shared store, trains, publishes its best model back, and plays a
  cross-population rating round, so the ratings live entirely in the global
  registry. (The loop used to copy a central pool into every worker, merge their
  archives back, re-rank them in a local arena and trim to 20 — all of that, plus
  `exchange/` and `retired/`, is gone; see "Population-wide Elo and pruning".)
- **Where things live.** Models in `checkpoints/models/` (shared, write-once);
  ratings in `checkpoints/global/`; and per machine only `--state-dir`
  (`loop_state.json` and the `STOP` file), `--work-dir` (each worker's scratch
  directory `work/gen<N>-w<K>/` and live checkpoint `work/agent-gen<N>-w<K>.pt`)
  and `--log-dir`. `run.sh` points all of it at `checkpoints/machines/<host>/`.
- **Published names are unique by construction**:
  `<machine>-gen<N>-w<K>-agent-<runid>`. `poker-train` derives the run id from a
  start-time timestamp with second resolution, so workers launched together in a
  generation share it — observed directly in the first end-to-end test, where
  three of four workers had the identical `agent-20260914-194509.pt` — and the
  machine/generation/worker prefix is what keeps them apart, including across
  hosts that happen to be on the same generation number.
- **Monitoring is `poker-loop --status`**, or `--status --watch 30` to redraw in
  place every 30 seconds. It reads `loop_state.json` from `--state-dir`,
  written after *every phase* (starting, training, publishing, benchmark) so a
  check mid-generation still reports something true. It shows the per-worker
  table (below) and nothing else. The per-series bb/100 matrix that used to sit
  under it went with the series themselves: the round against the anchors draws
  from the whole frozen set, so there is one reading per worker, in the table.
  **Deliberately no leaderboard, no per-generation history and no benchmark
  trends** -- all three were removed at the user's request, for one reason:
  `--watch` answers what this machine is doing *now*, and they pushed the live
  worker table off the screen. Nothing is lost by it: the history stays in
  `loop_state.json`, and the ranking and both benchmark figures are still
  printed into the supervisor log as each generation ends. `format_leaderboard`
  and `render_trend` remain for those.
  - **Per-worker progress is parsed from the workers' own logs**
    (`worker_progress` → one `WorkerProgress` per worker), not reported by the
    supervisor. The supervisor is blocked in `process.wait()` for the entire
    training phase and learns nothing until a worker exits, so without reading
    the logs a twenty-minute generation would show only "phase: training". The
    parser has to tolerate a half-written final line, since the logs are read
    while the workers are still appending to them; `parse_worker_log` is split
    out from the file handling precisely so every case can be tested on a string.
  - **What each worker's row carries**: iterations done out of the target, the
    stage it is in, the training win rate, the last pool evaluation and rating, the
    **mean rating of the field it drew**, the bb/100 against the frozen anchors
    (`ancore/100`) and the rating it published with (`elo ancore`), entropy, and
    how long ago its log was last written.
    - **The `pool` column is what makes `eval` and `rating` readable at all.**
      Those two say how the learner did; this says *against whom*. Every worker
      draws its own ~50 opponents (`draw_training_pool`), so two rows of the same
      table are not comparable without it -- a rating of 1520 means opposite
      things against a field averaging 1450 and one averaging 1600. `train.py`
      prints `pool rating: media <n> min <n> max <n>` right after the draw and
      `monitor.py` parses it; "-" for a log written before the line existed, or a
      run against an empty store.
    - **`train/100` is a mean over the last `REWARD_WINDOW_HANDS` (100,000)
      hands**, in **bb per 100 hands**. One iteration swings far too much to read
      on its own, and a cumulative mean would bury the present under the first
      iterations of the run. Note what it measures: the learner against the
      whole training field, the copies of itself in the seats `opponent_probability`
      left unfilled included, which is *not* the same
      opposition as `eval` (the drawn pool) or `serie` (the frozen set) —
      the three columns are not comparable with each other.
      - **It was the last 10 *iterations*, and a fixed hand budget replaced it
        at the user's request** once `--hands` became a swept axis. A window of
        10 iterations covers 3,200 hands for a worker that drew 320 and 8,000
        for one that drew 800, so two rows of the same table carried noise
        differing by **1.58x** with nothing saying which was which — an
        asymmetry that simply did not exist while every worker ran 512.
      - **How noisy it was, measured on 1,488 real runs**: the reward of one
        iteration has a standard deviation of **0.417 bb/hand** at 512 hands,
        which is 9.4 bb on a single hand. The old 10-iteration window therefore
        carried **±13.2 bb/100** of noise, against a column that typically reads
        between −3 and +3 — mostly noise at the level anyone reads it. 100,000
        hands take that to **±3.0 bb/100**, identical for every worker whatever
        it drew, which is exactly the 4.4x the hand counts predict. Treat the
        column as "is this run alive and not collapsed", not as a measurement:
        the measurements are `ancore/100` (500,000 hands, ~±8) and, distantly,
        `eval` (10,000 hands, ~±57).
        - **Measure this from first differences, not from the raw spread.** A
          run improves over its life, and a plain standard deviation over its
          iterations charges that trend as noise. Done the naive way on a
          narrower sample it came out 0.718 bb/hand, 1.7x too high, and every
          figure derived from it was wrong by the same factor until it was
          redone as `stdev(diffs)/sqrt(2)` over a 2,500-log random sample.
      - **The window is partial early in a run, and that is shown rather than
        blanked.** At 512 hands an iteration it fills at iteration 196, about a
        fifth into a 1000-iteration run (a twelfth at 800 hands, a third at
        320). Blanking it would remove the "is it alive" signal for exactly the
        stretch where it is read minute to minute, so the partial mean is shown
        and `WorkerProgress.train_hands` says how many hands are behind it — the
        dashboard puts that in the cell's tooltip and marks a filling window
        with a `*`.
      - **The window is sized from the header line** every worker prints as it
        starts (`device cpu | 6 seats | 512 hands/iteration`), not from the
        `iperparametri:` line, because every log ever written has the header,
        including the years of logs that predate the sweep. With no header at
        all the parser averages every iteration recorded and reports
        `train_hands` 0, claiming no precision it cannot back.
      - **The column was `rew/mano`, in bb per hand, and the unit changed at the
        user's request.** Every other win rate in this project is bb/100, so a
        single column in bb/hand was the one number a reader had to rescale in
        their head, and at two decimals it showed `+0.01` where the others show
        `+1.0`. **The conversion happens at the parse boundary**
        (`monitor.HANDS_PER_RATE`), not in `train.py`: the `iter` line on disk
        still carries bb per hand, deliberately, because a worker launched before
        the change keeps writing the old line and a changed unit would make the
        two indistinguishable and wrong by a factor of 100. The field is
        `WorkerProgress.train_bb100` / `WorkerHistory.train_bb100`, named for its
        unit for the same reason. Pinned by
        `test_the_training_rate_is_converted_to_bb_per_100_hands`.
    - **An age column, and a warning above the table.** A killed worker leaves
      its log behind, so without the age its bar sits at 40/100 forever and
      reads as a slow worker rather than a dead one. Past
      `STALE_LOG_SECONDS` (10 minutes) in a stage that should still be writing,
      the row is flagged with `!` and named in an "ATTENZIONE" line. Generous
      on purpose: a population round plays thousands of hands between two lines.
  - **The stage comes from `phase: <name>` lines the worker prints itself**
    (`rl/phases.py`, `train.py::announce`), because several stages of a run
    print *nothing* for minutes — the population round plays ~13,000 hands in
    silence — and the last ordinary line would otherwise make a worker deep in
    the Elo round look like it were still training. `rl/phases.py` is the whole
    shared vocabulary, so the name a worker prints and the name the watcher
    parses cannot drift apart; the population round announces its own three
    stages through the `on_phase` callback threaded down from
    `run_population_round`. The watcher adds three states of its own: `avvio`
    (no iteration yet), `training`, and `ERRORE` — set by a traceback in the log
    and deliberately **sticky**, since nothing printed after a crash undoes it.
    At the target iteration count `stage_label` renames the last stretch
    ("fine training"), because "training" at 100/100 really
    means the wrap-up — the saving, the publishing and the round against the
    anchors. A log written *before* the markers existed ends at the
    "global round:" summary line, which is read as `concluso` — without that,
    every archived log would show its finished worker as stuck in training.
  - **The two long stages also report how far into themselves they are**
    (`phases.py::progress_marker`/`parse_progress`, `train.py::PhaseProgress`).
    A marker says *what* a worker is doing; these say *how much is left*. They
    exist because the benchmark round (~2h45 at 500 sessions) and
    the population round (~25 min since `--global-games-per-model` went to 12,
    ~90 min before) print, between them, four lines — long enough that the
    10-minute stale-log warning fires on a perfectly healthy worker, and long
    enough that "elo: gioco" for hours answers nothing. Note it is the
    *benchmark* round that dominates now, not the population round. The line is
    `avanzamento <stage>: <done>/<total> (<pct>%), <detail>, ~<n>m rimasti`,
    written and parsed in `phases.py` for the same reason the stage names are:
    the worker that prints and the watcher that reads cannot drift apart.
    - **The population round counts owed games, not sessions.** A session seats
      `num_players` models and decrements each one's debt, so
      `len(labels) * games_per_model` is exact up front while the session count
      is not (it depends on how often the random seats land on a model that
      owes nothing — 55 models x 50 games is ~459 by division and 485 in
      practice). A bar that can overshoot its own denominator is worse than no
      bar. The series round counts sessions, which it does know, and takes a
      skipped series back out of the total — with one closing report, because a
      series skipped *after* the last session played would otherwise leave the
      count stuck at 66%.
    - **The round reports every session; `train.py` decides the cadence**
      (`PROGRESS_EVERY_SECONDS`, 30), always printing the first line and the
      last. ~100 and ~180 lines per stage: `--status` is never more than half a
      minute stale, and the watcher, which re-reads the whole log on every poll,
      stays cheap.
    - **Progress is cleared when a new stage is announced**, so a percentage
      shown next to a stage always belongs to that stage. `stage_label` still
      answers "what is it doing" (the dashboard and the phase counts use it);
      `stage_cell` adds "and how far", which only a table cell wants.
    - **`--status` also prints a `fine :` line and the dashboard a `fine ~38m`
      pill**, both the *longest* ETA rather than the mean: a generation ends
      when its slowest worker does, and that is the number a supervisor waiting
      to stop the machine is actually asking for.
    - Only the in-process path reports (`workers <= 1`), which is the one
      production uses. A sharded round (`benchmark_arena`, run by hand) spreads
      its playing over subprocesses with their own scratch logs.
  - **The per-worker live benchmark is gone, removed at the user's request**
    ("non mi interessa più e occupa tempo"). A worker used to benchmark its own
    still-training model every `--benchmark-every` iterations
    (`--worker-benchmark-every`/`--worker-benchmark-hands` on `poker-loop`,
    250 iterations / 3000 hands), print
    `        benchmark: <bb/100> bb/100 vs <n> avversari fissi`, and have it
    shown as a `bench/100` column in `--status`, `--watch` and the dashboard,
    with its own curve in the per-worker charts. All of that is deleted: the
    column, the chart series, the `benchmark` field on `WorkerProgress` and
    `WorkerHistory`, the `_BENCHMARK` regex, the `BENCHMARK` phase in
    `rl/phases.py`, and the `benchmark finale` stage label.
    - **`poker-loop`'s own once-per-generation benchmark stays**
      (`run_generation_benchmark`, `--benchmark-hands`/`--benchmark-seed`,
      `rl/benchmark.py::run_benchmark`), and so do the frozen anchors and
      `rate_against_benchmark`. Only the per-worker reading was removed, so the
      cross-generation figure in the supervisor log is unaffected.
    - **The three `poker-train` flags that fed it are retired, not deleted**:
      `--benchmark-every`, `--benchmark-hands` and `--benchmark-seed` are still
      parsed and their values discarded, because a supervisor started before this
      change keeps forwarding them on every worker it launches — see "Never
      delete a `poker-train` CLI flag while a supervisor is running", which is the
      outage this rule exists to prevent. `--benchmark-dir` is *not* retired: it
      still names the frozen set `rate_against_benchmark` uses. Verified by running
      `poker-train` with all three flags exactly as an old supervisor passes them:
      the run parses, trains, publishes, and prints no benchmark line.
    - **An old worker log still parses.** A worker already running when the
      change landed keeps writing `phase: benchmark` and its benchmark line; an
      unknown phase marker is ignored (`announced in STAGE_LABELS`) and the
      result line matches nothing, so the row simply stays in `training`.
      Pinned by `test_a_worker_no_longer_reports_a_live_benchmark`.
- **Every published model is rated against the frozen anchors before the
  population round** (`benchmark.py::rate_against_benchmark`,
  `--benchmark-sessions` **500** sessions of 1,000 hands — **500,000 hands**,
  forwarded by `poker-loop`). A model is otherwise published with the rating its
  own training run measured and only earns real games when a population round
  happens to draw it — about a 0.5% chance per round out of ~9,600 models, so it
  can sit in the ranking on an unearned number for many generations. This plays it
  against the frozen anchors right after training instead, so it enters the global
  ranking with 600 rated sessions behind it.
  - **There is no notion of a *series* here any more.** The round draws its
    opponents at random from the whole frozen set, and the per-series breakdown —
    the `SeriesResult` list, the bb/100 matrix under `--status`, `series_bb100` —
    is gone. It said *which* era of the population a model beat, which is
    information nobody acted on, bought at the price of every session being played
    against a field chosen for its label rather than for what it tells us. What is
    left is one number, `benchmark_bb100`, over the same 500,000 hands and at the
    same ±8 precision, so the hyperparameter sweep's response variable is unchanged
    in everything but its name.
  - **Why 500 sessions.** A 1000-hand session measures a rating with a standard
    deviation of 238 points, so precision improves only as `1/sqrt(sessions)`: a
    run's 100 validation sessions leave a 95% band of about **±42**, 500 more here
    take it to **±20**, and halving that again to ±10 would cost 2,000 sessions —
    11 hours a worker, four times the hands. 500 is the point where the round costs
    what the old 11-series arrangement cost (2.8 hours against 3.1), so the fleet's
    model production rate is unchanged while the published rating improves from
    about ±25 to ±20. Measured against the information floor, the arrangement sits
    15-26% above it at every budget: **it is not mistuned, it is under-fed.**
  - **The K comes from the schedule, continuing the learner's own count.** It used
    to be a flat 8, which settles at a jitter of **±35 points whatever the session
    count** — a fixed K is an exponential average with a fixed effective window, so
    more sessions buy literally nothing. The falling K is what turns 500 sessions
    into 500 sessions' worth of evidence. Measured at 550 sessions: flat 8 gives
    ±35, the exact hyperbolic gain ±20, and the information floor is ±20.
  - **The sessions are deliberately NOT queued for the global merge**, which is the
    opposite of what this round used to do. They are already in the rating the
    model publishes with, and `_apply_session` would re-apply the identical
    evidence a second time from the member's own games count — measured as
    **10-20% worse** than applying it once (a band of 11.2 against 10.0 at 2,233
    sessions), because re-using the same outcomes cannot add information and does
    add movement. Instead `publish_model` takes both the rating **and** the session
    count (`games_after`), which is what makes the local computation authoritative
    rather than a report. Publishing at `games=0`, as it did while the merge owned
    the update, would have a later population round shove a 600-session model
    around at the schedule's top tier on evidence it already has.
  - **The anchors are the right opponents**: their ratings are pinned, so the
    result is measured against a fixed scale rather than a moving field, and
    they are never seated in training, so it is not memorisation. Opponents near
    the model's own rating are the informative ones — a session's information goes
    as `p(1-p)` — so `benchmark_1` (334 points below a 1500 model) costs 77% more
    sessions than `benchmark_6` for the same precision. The draw is left **uniform
    anyway**: biasing it toward near-rating anchors saves 15% of sessions but makes
    every model play a different field, and `benchmark_bb100` has to stay
    comparable across models to serve as the sweep's response variable.
  - **Anchors are loaded in rotating slices, not all at once**
    (`--resident` 10 anchors, re-drawn every `rotate_every` 50 sessions). The
    frozen set is 55 models and only grows; holding all of them resident would cost
    ~440 MB per worker, and every worker of a generation reaches this phase at about
    the same moment, so 25 of them would want ~11 GB together — the profile of the
    unexplained fleet incident. A slice costs ~80 MB and the round still faces the
    whole set. A checkpoint that fails to load is **struck off** the candidate list
    rather than merely skipped, so a bad file is reported once instead of once per
    slice.
  - **A frozen set too small to seat a table returns `None`, and that is not an
    error**: a fresh installation has no anchors yet, and the run still publishes,
    at the rating its validation measured.
  - **`play_sharded` had never actually worked.** It launches
    `python -m pokerlab.rl.global_arena`, and that module had **no
    `if __name__ == "__main__"` block**: every shard imported it, did nothing,
    wrote no output file, and was reported through `on_skip` as a missing file, so
    a sharded round returned *zero* sessions. It stayed dormant because nothing in
    production passes `workers > 1` (`train.py`'s `run_population_round` call
    leaves it at the default 1), so the only code path exercising it was one
    nobody had run. Fixed, and pinned by a test that runs the module as a
    subprocess and requires the argument parser to fire. The same change made the
    games split **exactly** between shards (`shard_games`): it was
    `ceil(games / workers)` per shard, which silently played 30 games where 20
    were asked for whenever the division was uneven.
  - **The history, for reading old logs and old numbers.** The round was 11 series
    x 50 sessions x 1,000 hands, each series scored separately, the rating carried
    across them at a flat K, and the sessions merged into the global registry
    afterwards. Before that it was 10 x 100 = 1,000 hands per series, which could
    not support any conclusion — a 95% interval of about ±180 bb/100, against a real
    set of readings spanning −92 to +50. A worker log written before the change
    still parses: its per-series lines match nothing, so `--status` leaves the two
    cells empty, and the `phase: series` marker is unchanged so the stage still
    reads (pinned by
    `test_a_log_from_before_the_series_were_removed_reads_as_not_yet_rated`).
- **Stopping, the graceful way**: Ctrl-C (or SIGTERM) finishes the generation in
  flight and then exits, rather than killing workers mid-run and losing their
  archives. Creating a file named `STOP` in `--state-dir`
  (`checkpoints/machines/<host>/STOP`, which is what `./run.sh stop` creates)
  does the same, which is how an unattended loop is stopped from another shell.
  Note that "finishes the generation" can mean twenty minutes of waiting.
- **Stopping, the immediate way**: `poker-kill` (`./run.sh kill`,
  `rl/killswitch.py`) kills this machine's supervisor, its workers, their
  children and any global-round shard at once, whatever stage each is in.
  Measured: 0.3 s for a supervisor plus four workers.
  - **SIGSTOP to everything first, SIGKILL only after the table stops growing.**
    A plain `pkill` loop races the supervisor, which can launch the next
    generation's workers between one scan of the process table and the next. A
    stopped process cannot be caught out, cannot fork and cannot clean up, so
    freezing first — supervisor before workers — closes that window; the table
    is then rescanned until no newcomer appears, and only then does SIGKILL go
    out. Every ancestor of the killing process is protected, or a kill run from
    a shell under the work directory could take down the shell reading the report.
  - **Selection is by entry point *and* directory**, never by a substring of the
    command line: a process qualifies only if what it runs is the supervisor,
    `poker-train` or a global-arena shard (argv[0] or the module after `-m`)
    *and* it names this machine's `--state-dir`/`--work-dir`, resolved against
    its own cwd. This is the opposite choice from `sweep_stale_work`, which
    matches loosely on purpose because there an over-match merely keeps a
    directory; here an over-match kills a process, so `tail -f` on a worker's
    log, an editor, or another machine's worker seen through the shared mount
    are all left alone. `--dry-run` (`./run.sh kill-dry`) lists the selection
    without signalling anything.
  - **It cleans up nothing, on purpose.** A killed worker leaves its scratch
    directory and possibly a per-model lock; the next `./run.sh start` sweeps
    it, publishing any unpublished archive from its sidecar first (verified end
    to end: four workers killed mid-run, three archives salvaged and published
    on the next start, one already published by the worker itself), and locks
    expire on their own. It does mark `loop_state.json` as `phase: killed`, so
    `--status` stops reporting a generation with nothing behind it.
  - `rl/killswitch.py` deliberately **imports neither torch nor `loop.py`** — it
    has to start in a fraction of a second on a box whose cores are all busy. The
    one constant it duplicates, `STATE_FILENAME`, is pinned to the loop's own by
    a test.
- **Residues of an interrupted run are swept at the start of every generation, and
  once more after the workers exit** (`loop.py::sweep_stale_work`, skipped with
  `--keep-work`). A loop killed mid-generation never cleans up after its workers,
  and the next run continues from a *new* generation number, so
  `work/gen<N>-w<K>/`, `work/agent-gen<N>-w<K>.pt` and any `work/arena-gen<N>/`
  from an older version were never removed — measured, ~2.2 GB per machine on
  three machines, plus 85 trained models that no ranking could see. The sweep
  first **publishes each abandoned worker's own `agent-*.pt`** into the shared
  store (named as the worker would have named it, registered at the rating its
  sidecar recorded; skipped if that name is already published), then removes the
  directory. The after-training pass is also how a worker that *crashed* before
  publishing still gets its model in. It additionally removes this machine's own
  stale `.<machine>-*.partial` in the models directory (a publish that died
  mid-copy) and stale `global-arena-shard-*` scratch in the temp dir, both only
  once untouched for an hour. **Safety rules that must survive any change**:
  (1) anything a running process still names on its command line is left alone —
  a worker can outlive a supervisor that died, and deleting its directory
  corrupts it; liveness comes from `/proc`, which only sees this host, which is
  correct because `work/` is this machine's own and never shared; where `/proc`
  is missing nothing is deleted; matching is by substring, so anything that merely
  *mentions* a worker's name (a `tail -f` on its log, say) keeps its directory
  around — the safe direction; (2) a directory whose archive could not be
  published is kept; (3) only the exact name shapes the loop itself creates are
  touched. **Never run the sweep on one machine against another machine's
  `work/` directory from a shell** — this host's `/proc` cannot see that machine's
  workers, so a live one would look abandoned.
- **Every worker inherits weights; none starts from scratch any more**
  (`--inherit-fraction`, default **1.0**, via `inheritance_plan`). It was 0.5,
  and the fresh half was removed at the user's decision after being measured
  over 2,810 real runs: a from-scratch worker trained **worse by 25-46 bb/100 in
  every group**, 59 of 95 exceeded the `kl` 0.05 "step too large" threshold at
  iteration 1, and 8 of 95 collapsed below −100 bb/100. The argument the fresh
  half existed for — that a dead end, a policy collapsed onto folding say, could
  otherwise trap every lineage at once — is now answered by the hyperparameter
  sweep below instead: the lineages differ by their *settings*, which is
  diversity that costs nothing, rather than by throwing away a trained network.
  Without *any* inheritance the loop only ever produces models exactly
  `--iterations` deep — it generates variety, not strength; measured early on,
  the best model was still generation 1's and ten of the top fifteen were
  pre-loop models with 200 iterations behind them.
  Inheritors take **distinct parents drawn at random from the `--pool-top-n`
  best-rated models on disk** (`training_pool.pick_parents`: rated, non-frozen,
  cycling only if there are fewer than needed), a different set every
  generation, so the population is competing lineages rather than copies. A
  model that has never been rated is not a parent.
  - `--resume` reads `--checkpoint`, so the loop copies the chosen parent from the
    store into the worker's checkpoint path first (the store's file is only read).
    **Only weights carry over**: published models store no optimizer state, so
    Adam restarts cold. That costs a short transient and is far cheaper than
    discarding the weights too (saving Adam is on the TODO list).
  - **`run.sh` must not pass `--inherit-fraction`, and this cost a day of fleet
    time.** The default moved to 1.0 here, but `run.sh start` passed `0.5`
    explicitly and an explicit flag beats a default, so half of every
    generation kept starting from a random network after from-scratch runs were
    supposed to be gone. Found by reading live logs, not by reasoning:
    zebele-slaves-2 generation 425 had 12 of 25 workers with `resumed from` and
    13 at `lr=0.001`, a perfect 1:1 match with `--fresh-lr` — exactly the 0.5
    that line asked for. `--hands` and `--self-share` were being passed the same
    way and were equally inert once the sweep decided them, so all three left
    the launcher (`--self-share` has since gone altogether — see "The run's own
    past selves are not in the training field"). **The general rule: when a default moves in `loop.py`, check
    whether `run.sh` is overriding it**, because a default only governs the
    paths that do not. Pinned by
    `test_run_sh_does_not_override_the_axes_the_sweep_decides`, which reads the
    launch command out of `run.sh` and requires every swept axis to be absent.
  - **`--fresh-lr` (1e-3) is dormant, not deleted.** It is now reached only when
    the store cannot supply a parent at all — an empty store, the very first run
    ever — where a random network genuinely does have further to travel. It is
    deliberately left *outside* the sweep: a rate drawn for a warm start means
    nothing on a cold one. Deleting the flag is what takes the fleet down (see
    "Never delete a `poker-train` CLI flag while a supervisor is running"), so it
    stays whatever happens. Pinned by
    `test_no_worker_starts_from_a_random_network_any_more`.
  - `--status` marks inheriting workers with `^` and fresh ones with `.`. With
    `--inherit-fraction` at 1.0 every row should read `^`; a `.` now means the
    store had no parent to give, which is worth noticing rather than hiding.

- **Every worker of a generation trains with its own hyperparameters**
  (`loop.hyperparameter_plan`, recorded in the model it publishes by
  `train.run_metadata`). Before this, all N workers of every generation on every
  machine ran the *identical* settings, so the fleet produced thousands of runs
  of one configuration and no evidence whatsoever about any other. A generation
  is now a sweep.
  - **Each axis is a ladder of allowed values, and the *sampled* arm draws on it**
    (`HP_LADDERS`; the inherit arm multiplies instead and is not bound by the
    rungs — see below): `--lr` (1.9/2.4/3.0/3.8/4.7e-4), `--hands` (320/400/512/
    640/800), `--opponent-probability` (0.32/0.40/0.50/0.62/0.78),
    `--ppo-epochs` (2/3/4/5/6), `--clip-epsilon` (0.13/0.16/0.20/0.25/0.31) and
    the two that decide how strong a field the worker draws, `--pool-top-share`
    (0.32/0.40/0.50/0.62/0.78) and `--pool-top-n`
    (51/64/80/100/125/156/195). There used to be a `--self-share` axis, gone
    with the mechanism it sized — see "The run's own past selves are not in the
    training field". **`--pool-models`, `--global-sample` and
    `--global-games-per-model` are held fixed**, at the user's decision: the
    first is the field's *size*, which is also what a run's cost scales with, and
    the last two are how the whole fleet's ratings are earned, so varying them
    would make runs incomparable rather than comparable along one axis.
  - **The centre rung of every ladder is what the fleet ran before**, so the
    sampled arm is a draw *around* the known-good point rather than a jump away
    from it. It says nothing about where the inherit arm sits: that one walks
    from its parent and can be anywhere by now.
  - **The two arms move by different mechanisms, deliberately.** The sampled arm
    draws a rung. The inherit arm multiplies its parent's value by one of
    `HP_MULTIPLIERS` — **x{0.8, 1.0, 1.2}, unbounded**, never snapped back onto
    the ladder — so a lineage holds whatever its ancestors' draws multiplied out
    to and can walk clean past the ends of the ladder. Changed at the user's
    request, from an earlier ±1 rung step clamped to the ends.
    - **Why unbounded suits that arm.** The ends of these ladders were never
      measured, and the clamp made them load-bearing: a parent already on the
      top rung stayed there **two thirds** of the time (both the +1, clamped,
      and the 0), so the ladder's author decided in advance the furthest the
      fleet could ever go. The real bound is selection, and that one *is*
      measured — a parent is drawn from the `--pool-top-n` best-rated models, so
      a lineage that walks its lr up to something that breaks training rates
      badly and stops being a parent. An arithmetic clamp guesses where the edge
      is; the ranking finds out.
    - **The cost, accepted: a lineage really can drift far.** The step is a
      random walk in log space, so the spread after N generations is about
      `ln(1.2)*sqrt(2N/3)` — roughly 27x either way over 500 generations, which
      would put `hands` anywhere between ~12 and ~8600. Nothing stops that but
      the ranking. Simulated over 400 free lineages with **no selection at all**,
      60 generations from lr 3.0e-4 spanned 2.9e-6 to 1.2e-2.
    - **The multipliers are not symmetric in log space, and the drift is
      measurable.** 1.2 is not 1/0.8, so the set's geometric mean is
      `(0.96)^(1/3) = 0.9865`: every axis of an inherited lineage shrinks ~1.35%
      per generation with nothing opposing it but selection. In that same
      simulation the median lr landed at 1.27e-4 — exactly the `0.9865^60 = 0.44`
      the geometric mean predicts. **This is why canonical PBT uses x1.25**: 1.25
      is 1/0.8, so (0.8, 1.0, 1.25) has a geometric mean of exactly 1. Keeping
      1.2 is the user's choice; the unbiased counterparts, if the drift ever
      proves unwanted, are (0.8, 1.0, 1.25) or (1/1.2, 1.0, 1.2). One tuple in
      `loop.py`.
    - **Only two limits survive, and neither is a rung.** A probability axis is
      capped at 1.0 (`_HP_PROBABILITY_AXES`: `opponent_probability`,
      `pool_top_share`), because a probability above 1 is not one and both reach
      code that would quietly do something odd with it. And a **count axis moves
      by at least 1** whenever the multiplier is not 1.0, floored at 1
      (`_moved_count`).
    - **The count axes needed that rule, and finding out cost a measurement.**
      Plain rounding looks sufficient and is not: `round(2 x 1.2) = round(2.4)
      = 2` and `round(2 x 0.8) = 2`, so **2 was absorbing in both directions** —
      and 2 is the bottom rung of the `ppo_epochs` ladder, one step under 3 and
      two under the fleet's own 4. It was a one-way ratchet, not an edge case:
      measured on the code before the fix, **62% of lineages were stuck at
      `ppo_epochs` 2 after 20 generations and 96% after 200**, against a median
      of 4-5 afterwards. So the rule is: round as usual, and only where that
      would leave the value where it started, step exactly one in the
      multiplier's direction — at the user's request ("se varia, metti la
      variazione minima di 1"). *Always* rounding away from the current value
      would fix the ratchet too and was rejected: it inflates the step wherever
      rounding was working, turning 6 into 8 or 4 instead of the 7 or 5 that
      x1.2 and x0.8 actually ask for.
    - **The floor is 1, not 0 and no longer 2.** `ppo_epochs` 0 is not a smaller
      setting, it is the absence of training, and 1 is the integer analogue of
      the fact that multiplying a positive value by 0.8 never reaches 0. 1 is
      reflecting rather than absorbing — the up draw from 1 gives 2 — and note it
      is *reachable*, which the ladder's bottom rung of 2 never allowed. On
      `hands` and `pool_top_n` the floor is unreachable in practice, ~24
      consecutive downward draws away.
    - **Multiplication now works on every axis, which it did not before.** One of
      the two original arguments for a ladder was `self_share`, whose bottom rung
      was 0.0 and which no multiplier can ever leave; that axis went with the
      snapshots and no ladder has a 0 rung any more. The other argument — an
      integer multiplied drifts off any grid an analysis could group by — still
      stands and is simply accepted: inherited points are a continuum now. It
      costs nothing that was not already lost, because an inherited point never
      could be read as a response curve anyway (it correlates with its parent's
      quality by construction). The sampled arm is still on the grid, and it is
      still the readable one.
    - `HP_LADDERS` still says *which* axes exist, so adding an axis there is all
      it takes for both arms to move on it; only its *values* have stopped
      constraining the inherit arm. Pinned by
      `test_an_inheriting_worker_multiplies_every_axis_by_one_of_the_multipliers`,
      `test_a_lineage_can_walk_clean_past_the_ends_of_the_ladder`,
      `test_a_probability_axis_is_capped_at_one` and
      `test_a_count_axis_is_rounded_so_it_can_never_reach_zero`.
  - **Every worker inherits its parent's settings and perturbs them; there is no
    sampled arm** (`hyperparameter_plan`). It used to be a 50/50 split
    (`--hp-inherit-share`, default 0.5 — the user's "un mix"); the sampled half
    and the flag were both removed at the user's request, so a generation is now
    a pure search. The only workers that do not inherit are the ones that
    *cannot* — see the fallback below.
    - **What the split was for, and what removing it costs.** Sampling alone was
      a controlled experiment that never compounds: every generation re-drew from
      the same box and the best combination found was never built on. Inheriting
      alone is a search that cannot be *read*: after a couple of generations the
      surviving settings are whatever the top-100 happened to carry, so there is
      no comparison left to make and no way to tell a good setting from a lucky
      lineage. The fleet can now find a good configuration and can no longer say
      *why* it is good. That is the accepted trade.
    - **The second cost, which is easier to miss: the anchor is gone.** The
      sampled arm was re-drawn from the ladder every generation, so half the
      fleet sat at the known-good centre by construction and no lineage could
      drift far from it. Nothing anchors the search now except the ranking that
      picks parents — and `HP_MULTIPLIERS` is not symmetric in log space, so
      every axis of every lineage shrinks ~1.35% per generation on average with
      no cohort at the centre pulling against it. If the fleet's settings are
      seen sliding downward over many generations, this is the mechanism, and
      the fix is the symmetric multiplier set (see above).
  - **The arm is still recorded in every published model** (`hp_arm`:
    `inherited`, `sampled-fallback`, and `sampled` only from a hand-driven
    `poker-loop` with no plan). Still not bookkeeping: an inherited point
    correlates with its parent's quality *by construction*, because the parent
    was drawn from the top 100, so it cannot be read as a response curve — what
    has changed is that there is no longer a cohort of independent draws to
    compare it against. `poker-train` records the field and never acts on it —
    which arm a worker is in is a `poker-loop` decision.
  - **The fallback is the whole of the non-inherited population now, and it is
    not a choice.** A worker can only inherit from a parent whose checkpoint
    actually carries the metadata; with no parent, or one published before the
    metadata existed, there is literally nothing to perturb, so that worker draws
    a rung of `HP_LADDERS` and is labelled `sampled-fallback`. **Measured on the
    live fleet the day the sampled arm was removed: of 72 workers that attempted
    to inherit, 17 fell back — 24%.** So "only inherited" means ~three quarters
    of each generation in practice, falling as more top-100 models carry the
    metadata. The generation header prints the count, which matters more than it
    did: it is now the only signal that inheritance is working at all, and a
    metadata regression would quietly turn every worker into a ladder draw.
  - **Hyperparameters are inherited only from the weights parent.** Perturbing
    the settings of a model this worker is not resuming from would attribute a
    configuration to a run that never had it.
  - The plan is reproducible from `--seed-base`. (It used to also be *shuffled*,
    so the arm was not tied to a worker index — with one arm there is nothing left
    to shuffle, and which workers fall back is decided by which parents carry
    metadata, not by position.)
  - **The response variable is `benchmark_bb100`**, recorded in the same
    metadata: the round against the anchors plays the published model over 500
    sessions of 1,000 hands, so **500,000 hands** and a 95% interval of about
    ±8 bb/100. It replaced the per-worker live benchmark (removed at the user's
    request, 3,000 hands and ±104) and is twelve times more precise at no extra
    cost, because those hands are played anyway. It was called `series_bb100`
    while the round was broken down per series; the quantity and its precision are
    the same, which is why the breakdown could go without touching the sweep.
  - **Run lengths are now deliberately ragged**, since `--hands` is swept: a
    worker drawing 320 finishes in well under half the time of one drawing 800.
    That is accepted, and the Elo fill-in phase below is what the difference is
    spent on.
  - **Each worker prints what it was configured with, into its own log**
    (`phases.hyperparameters_marker`, `iperparametri: hp_arm=... lr=... hands=...`).
    The values already existed in three places -- the worker's command line, the
    supervisor's log and the metadata inside the published model -- and *none of
    the three is what a watcher reads*: `--status` and the dashboard parse the
    worker's own log and nothing else. It also makes an archived log
    self-describing, which matters because a log saying a run reached -30 bb/100
    is worth much less once the supervisor log that held its settings has gone.
    - Deliberately `key=value` pairs, not a fixed-shape line:
      `parse_hyperparameters` hardcodes no key, so adding an axis needs no change
      in the watcher. The line is built from `run_metadata`, so it cannot
      disagree with what rides inside the checkpoint.
      `test_every_axis_of_the_sweep_is_one_the_worker_reports` fails if an axis
      is added to `HP_LADDERS` and not to `train.REPORTED_AXES`.
    - **`--fresh-lr` is substituted in exactly one place**
      (`loop.effective_hyperparameters`), and this was a real bug caught on a
      live run: with an empty store no worker had a parent, so every one of them
      trained at `--fresh-lr` while the generation header printed the *drawn*
      range -- `lr 0.00019-0.00047` in the header against `lr=0.001` in all four
      worker logs. The worker was right, since its metadata records the rate it
      was actually passed. The substitution is idempotent, so the supervisor
      resolving before launching and `launch_worker` resolving again cannot
      compound.
    - **Where to read it**: `--status` gains one `sweep :` line (how the
      generation splits between the arms, and the range of `lr` and `hands`) --
      one line, because the per-worker table has no room for six more columns.
      The dashboard carries the full per-worker set in the **expanded panel**
      (click a row), above the training curves, plus a per-machine pill with the
      arm split. It rides in the `/api/status` payload the page already polls,
      so opening a panel costs no extra request. A log written before the sweep
      existed records nothing, and every reader treats that as normal rather
      than as an error.
    - **A worker launched by a supervisor that predates the sweep names itself.**
      Such a worker runs perfectly well -- no flag was removed, which is the
      whole point of the retired-flags rule -- but it is given no `--hp-arm`, so
      it prints `hp_arm=` and the parser drops the empty value. Both `--status`
      and the dashboard label that case **"senza sweep: supervisor da
      riavviare"** and *count* it rather than skipping it, because a supervisor
      is a process that lives for weeks and only picks up new behaviour when it
      is restarted: this is how the fleet says which hosts still need it. The
      rest of that worker's settings are recorded and shown normally.

- **A worker that finishes early fills the wait with Elo rounds instead of
  idling** (`train.run_elo_fill_in`, `--elo-fill-in`; on by default in
  `poker-loop`, off by default in `poker-train`). Run lengths are now
  deliberately ragged — `--hands` is swept — so several workers a generation
  finish well before the slowest, and on a 25-worker box that was hours of cores
  doing nothing every generation.
  - **Why Elo and not more training.** Training more would publish more models,
    and the store's problem is not that it holds too few (~9,600) but that
    almost none of them have a rating worth anything: a round seats ~50, so a
    given model comes up about 0.5% of the time and can sit for many generations
    on the number its own training run published. And a longer run is not
    comparable with the others in its generation, which is the whole point of
    the sweep — the fast workers are fast *because* they drew fewer hands, so
    extra iterations would erase the very axis being measured. Rating rounds
    touch no published model's weights, only what is known about them. The fast
    workers become the fleet's rating engine, which is where the work belongs.
  - **When it stops: a floor in sessions, a cap in minutes, and a flag.**
    `--fill-min-sessions` (**50**) is played before the stop flag can end the
    phase, so a generation whose workers all happen to finish together still
    gets a rating phase — the case the user explicitly asked to protect.
    `--fill-deadline-minutes` (**150**) is the *only* cap, and it is expressed in
    minutes on purpose, at the user's decision: a maximum number of rounds would
    cap the work, while what actually has to be bounded is how long a worker can
    hold its core when nobody is coming to release it (a supervisor killed
    mid-generation, a state directory that moved). Pinned by
    `test_the_cap_is_only_ever_expressed_in_minutes`.
  - **The handshake, and the deadlock it exists to break.** A filling worker
    does not exit — it is waiting for its supervisor — and the supervisor used
    to block in `process.wait()` waiting for exactly that worker. Each would
    have held the other until the worker's deadline expired hours later. So the
    worker creates `draining` in its own scratch directory, meaning "I am done
    with my own work and only killing time"; the supervisor polls every
    `FILL_POLL_SECONDS` (10) and, when **every worker still alive** says that,
    creates `FILL_STOP` in its state directory, which the workers check between
    rounds. Both names live in `rl/phases.py`, with the stage names and for the
    same reason: a name that drifted between the two modules would not fail, it
    would leave every worker filling until its deadline while the supervisor
    waited for workers it thought were still training.
    - Both files are on the machine's **own** disk, never the shared volume: a
      shared stop flag would have the first machine to finish a generation
      release every other machine's workers too.
    - The stop flag is **removed at the start of every generation**, before
      anything is launched — a flag left by the previous one would release this
      generation's workers the instant they reached the phase.
    - The `draining` marker is removed in a `finally`, the error path included,
      or the next generation's supervisor would read a stale directory as a
      worker already draining.
    - `wait_for_workers` replaced `process.wait()` and still reports every
      non-zero exit, which is what the generation record and the fleet-outage
      diagnosis both read.
  - **A fill-in round is small and biased: `--global-sample` 50 models,
    `--fill-games-per-model` 1, so ~10 sessions and ~3 minutes.** The end-of-run
    round owes 12 games each and takes ~25 minutes; a fill-in round is a unit of
    *waiting*, so it has to be short enough that the worker notices the stop flag
    soon after it goes up and short enough to re-draw often — a long wait is then
    many independent draws rather than one stale one, and the ranking is re-read
    every round because the previous one just moved it. What it does not avoid is
    loading the drawn models, ~203 MB and a few seconds a round, which is why it
    is not made shorter still.
  - **It draws through `tiered_draw`, and so does the end-of-run round**
    (changed at the user's request, replacing `top_biased_draw`, which spent
    half the seats on the 1,000 best). A quarter of the seats goes to each of
    ranks 1-10, 11-100, 101-1,000 and everyone else (`DRAW_BANDS`, disjoint,
    unrated models in the last band). The top ten cannot fill their 12-13 seats,
    so they are all seated in every round and the shortfall spills uniformly
    over the models not yet drawn. The top is the only part of the ranking
    anything reads — `pick_parents` draws from the best 100 and the ordering
    there was measured wrong. **The fill-in rounds still lift `trigger_size` to
    `NO_PRUNE_TRIGGER` and can never prune**; the end-of-run round does *not*,
    see the next section for what that costs. Pinned by
    `test_a_fill_in_round_can_never_prune`.
  - **Torch-free at import** (it reaches torch only through `global_arena`'s
    function-local imports), so its tests run in the ordinary suite and it starts
    instantly. Pinned by a test.
- **Trigger for a round: the end of every single `poker-train` run**, not once
  per `poker-loop` generation — `--global-round` (on by default;
  `--no-global-round` to opt out) is checked as the very last thing `main()`
  does, in both standalone `poker-train` and every worker `poker-loop` spawns.
  Many workers finishing together is fine: they merge concurrently.
- **Sampling is by rating band, in every round that draws from the population**
  (`global_arena.tiered_draw`, passed through the `draw` hook): `run_population_round`
  draws `--global-sample` (~50) models from `discover_population(root)` — every
  model in `checkpoints/models/` — plus `--global-benchmark-sample` (~5)
  uniformly from `discover_benchmark_population`. The population draw gives a
  quarter of the seats to each of ranks 1-10, 11-100, 101-1,000 and the rest
  (the end-of-run round and the fill-in rounds alike, at the user's request; it
  used to be uniform here and half-top-1,000 in the fill-in).
  - **Cost, accepted and not yet measured: the end-of-run round now prunes under
    a biased draw.** Eligibility for deletion is a percentile of `games`, so the
    top bands (seated every round or every other round) become eligible sooner
    and the tail (a quarter of the seats over ~8,600 models) stays below the
    percentile and immune; `eliminate_lowest_rated` then removes the lowest
    rated *among the eligible*. Watch what the first passes delete — the
    population is within a few hundred of `--global-trigger-size` (10,000). The
    fix, if it shows, is `trigger_size=NO_PRUNE_TRIGGER` on that call in
    `train.py::main`, as the fill-in does.
  - A model never before seen is bootstrapped at the default rating when it
    first plays; a benchmark draw is bootstrapped `frozen=True`.
- **K follows a 10-step staircase approximating the optimal gain, and one
  schedule rates everything.** A flat K made a model with a thousand games move
  as much per session as a newcomer, so a well-determined rating still jumped
  around. `PoolRegistry(k_schedule=...)` gives each registered member its own K
  from its own `games` (`k_for_games`, tiers in `DEFAULT_K_SCHEDULE`): **20 tiers
  from 16 down to 0.01**, each a factor 1.4745 below the one above — 16 below 20
  games, then 11, 7.4, 5.0, 3.4, 2.3, 1.6, 1.05, 0.72, 0.49, 0.33, 0.22, 0.15,
  0.10, 0.07, 0.047, 0.032, 0.022, 0.015 and 0.01 from 60,000.
  - **Where those numbers come from, because it is derived rather than tuned.** A
    1000-hand rated session measures a rating with a standard deviation of
    **238 points** — Elo reads only the *sign* of each pair's chip delta, and the
    zero-sum structure correlates the learner's five pairwise comparisons at
    exactly 0.5, so they are worth 2.18 independent Bernoullis. For a quantity
    that does not move, the gain that extracts all of that evidence and no more is
    Kalman's, and in Elo's parametrisation it is exactly hyperbolic:
    `K_t = 1/(slope·(t + V/P0))` with `slope = 0.001421` and `V = 238² = 56,864`.
    The staircase is geometric in `(t + 35)`, and the two ends fix the ratio:
    `16 / 1.4745^19 = 0.01`, with 0.01 sitting at `704/0.01 − 35` ≈ 70,000 rated
    games, which is where the large thresholds at the bottom come from.
    - **It was 10 tiers halving from 16 to 0.10, and going to 20 improved the part
      that is actually used.** A staircase is necessarily low at the start of a
      tier and high at the end; at ratio 1.76 that was a **25%** swing inside each
      tier, costing 3% of the final precision of a published rating (a 95% band of
      19.9 points against the curve's 19.3, over 15,000 simulated runs). At 1.4745
      the worst tier is **4.5%** off the curve, and **12 of the 20 tiers fall below
      3,000 games** — the range the population can actually reach — against 8 of
      the 10 before. So the extension is not only headroom at the bottom: the live
      part got finer too.
    - Pinned by `test_the_staircase_tracks_the_hyperbolic_gain_it_approximates`,
      which checks each tier against the curve at the middle of its own range
      rather than against a literal, holding the open-ended bottom tier to a looser
      bound because it cannot track a falling curve.
  - **`V/P0` is the one number here that is an estimate, and it is where "we
    already know roughly where this model sits" is encoded.** A run inherits its
    parent's rating rather than starting at 1500, so `P0` is how far a child's
    strength differs from its parent's — taken at sd 40, giving an offset of 35.5
    and a first-session K of 20 instead of the 105 an uninformative prior would
    ask for (capped at the 16 of the first tier). Set it too low and the inherited
    rating is thrown away; too high and the inherited error persists, which matters
    because `pick_parents` draws from the top 100 and that is a positive feedback
    loop. **What keeps it safe is measured**: at this offset the inherited value
    carries only **~6% of the weight** of the final rating after 600 rated
    sessions, the other 94% coming from the sessions actually played. Worth
    measuring properly one day, with a duplicate-deck duel of a model against its
    own parent.
  - **The bottom tiers are headroom, not live tiers.** Measured over the 8,331
    **non-frozen** members — the ones the schedule can actually move — the
    distribution is median 911 games, 90th percentile ~1,780 and **max 2,727**, so
    everything from 3,750 games down (K ≤ 0.15) is reached by nobody yet. That will
    change faster than it used to: a model is now published with the ~600 rated
    sessions it earned rather than with zero. The bottom tier, 0.01 at 60,000 games,
    is inside the range the *frozen anchors* occupy (up to 115,372), and their
    deltas are discarded anyway — so read anything below ~0.2 as a promise the curve
    keeps rather than as a live tier. The frozen-only distinction matters: counting
    the anchors in makes those tiers look live when they are not.
  - **The whole of a run is one continuous session count, from its first
    validation session to publication.** `SelfPlayTrainer` starts at
    `learner_games = 0` and counts one per rated session; `evaluate_against_pool`
    reads its K from the schedule through the `k_factors` override on
    `record_session_with_ratings` (the learner is deliberately **not** a registry
    member — it changes every iteration, so persisting it would rate a moving
    target, which is exactly why its K has to be handed over rather than looked
    up); and `rate_against_benchmark` takes that count as `games` and keeps
    incrementing it. So a run's ~100 validation sessions take it from 16 to **5.0**
    by the time it reaches the anchors, and its 500 sessions there end at **1.05**,
    which is the tier it enters the global ranking in.
    **Nothing in the rating path uses a flat K any more.** What is left of
    `DEFAULT_K_FACTOR` (8) is the fallback for a participant nobody names, and
    `benchmark_arena`'s `--k` (16), which is a separate dial.
  - **The history, because the direction of travel is measured and the destination
    was not.** The tiers went 24/16/12/8/6 → 12/8/6/4/3 → 4.0/1.0/0.5/0.1 →
    3.0/1.0/0.3/0.1 → those four with a burn-in (24 below 10 games, 8 from 10)
    prepended → ten steps → today's twenty. Each earlier step was chosen by hand;
    the staircase is the first that follows from a derivation. The burn-in it replaced existed
    because a rating starting at 1500 could not travel at all: measured on the
    fleet, **17 live workers all rated between 1469 and 1534 with bb/100 from −686
    to +47**, and in simulation a learner *exactly* as strong as its pool finishing
    a 40-session run at 1537, 63 points below the field it was level with. The
    staircase does that job too — its first tiers are 16 and 9 — and does it
    without the flat 8 that used to rate the round against the anchors, which
    settles at a jitter of **±35 points whatever the session count**, because a
    fixed K is an exponential average with a fixed effective window and more
    sessions buy nothing.
  A session is rated at the experience the model had *when it sat down* (K read
  before `games` is incremented), an explicit `k_factors` entry wins over the
  schedule, and an unregistered participant nobody names keeps the flat K. Two
  consequences: a session is no longer exactly zero-sum in rating (each side
  moves by its *own* K). That does not matter here: frozen anchors never move, so
  exact conservation never held, and a ranking only needs the *order* to be
  right. A veteran converges more slowly to a genuinely changed strength, which
  is the intended trade. A registry built without a schedule (`k_schedule=None`,
  the default) uses the flat K; the in-memory registry of a training run does,
  because all its members are frozen — which is exactly why the learner's own K
  has to be passed in rather than looked up. `pairwise_elo_delta` takes an
  optional `k_factors` mapping for this.
- **Frozen Elo anchors**: `PoolMember.frozen` pins a member's rating forever (the
  benchmark anchors, and the opponents in a training run's in-memory registry) —
  `record_session_with_ratings` still scores it normally against everyone else
  and still counts its `games`, it just never applies its own delta. A frozen
  member is a fixed reference point the rest of the scale is measured against.
  - **`rl/benchmark_arena.py` is the single deliberate exception**, and the only
    code in the project that writes an anchor's rating. Nothing imports it; it
    is run by hand (`python -m pokerlab.rl.benchmark_arena --rounds 5`), rarely.
    It exists because an anchor is pinned at whatever rating it held the instant
    it was promoted — one number from the ordinary rounds, earned against a
    field that no longer exists — and the anchors have never been rated against
    *each other*, so nothing has ever checked that `benchmark_1`'s models sit
    correctly relative to `benchmark_14`'s. It plays them among themselves and
    settles that, printing every model of every series with its rating after
    each round. It bypasses `record_session_with_ratings` (which would refuse)
    and writes the member files itself, one at a time under its own lock,
    re-reading each so only `rating` is overwritten and `games`/`ref`/`frozen`
    survive.
    - **Every round prints the full per-anchor table and a per-series line**
      (`format_series_line`: each series' mean rating, its span, and how far its
      mean has moved since the run started). `--report-every` defaults to **1** —
      30 lines at 60 anchors, which is affordable because a round is minutes long
      and watching the anchors sort themselves out is the whole point of the run;
      turn it down for a long unattended sweep, or to `0` for only the ends. After
      the regroup the series *are* the strength bands, so the summary line reads as
      a ladder.
    - **It writes the ratings after *every round*, not once at the end.** The
      write is one small file per anchor, each under its own lock and re-read so
      only `rating` is overwritten -- milliseconds against an hour of play. Writing
      once at the end meant a run killed on its second day lost everything, which
      for a job needing `--min-rounds` x ~an-hour is the likeliest way for it to
      end; and because a flat K redistributes rating without moving the mean,
      every round leaves the anchors in a *consistent* state rather than a partial
      one. Verified by SIGKILLing a run after its second round and finding all
      eight ratings moved off 1500 on disk with their mean still exactly 1500.
      - **The shared `registry.json` snapshot is refreshed in the same breath**,
        forced, every round. The member files are the truth but nothing reads
        them one at a time: `--status`, the dashboard and the GUI's default table
        all read the snapshot, and `write_snapshot`'s own staleness rule only
        rewrites it when some other writer happens along. A convergence run holds
        the anchors for *days*, so refreshing only at the end left the whole
        fleet reporting the ratings the run started from. The cost is ~2 s
        (9,300 member files re-read into one 3 MB file) against a round measured
        in minutes, and a failure is caught and reported rather than allowed to
        end a multi-day run. `--dry-run` refreshes nothing, like every other
        write. Pinned by `test_every_round_refreshes_the_shared_snapshot` and
        `test_a_dry_run_refreshes_nothing`.
    - **When it stops: two independent floors and a drift test.** The loop breaks
      when `round >= --min-rounds` **and** `drift <= --tolerance`, where `drift`
      is the **net** movement — where a rating sits now against where it sat
      `--window` rounds ago, not the distance travelled in between, so a rating
      jittering around a settled value scores ~0 however far it wandered. It is
      the **worst** anchor, not the mean: one anchor still moving means the order
      is not settled however quiet the others are. Before the window is full
      `drift` returns `inf`, so convergence can first fire at round `--window`
      (the history starts with the pre-round-1 state, so after round *w* it holds
      *w+1* entries), and `--min-rounds` is a second, independent floor on top.
      Defaults: tolerance 1.0, window 10, min-rounds 20, max-rounds 500.
      - **`--max-rounds` is a safety cap, not a failure.** Hitting it prints
        "NON CONVERGE" and then does everything a converged run does — the
        ratings were already written every round, the snapshot is refreshed, and
        the regroup still runs. It only means the ratings were still moving.
      - **`--tolerance` is not scale-free: it has to be read against `--k`.** An
        anchor moves at most `k x 0.5` per session (`pairwise_elo_delta` divides
        each pair's change by the opponents faced), so at most
        `k x 0.5 x --games-per-model` per round. At `--k 0.1` with 20 sessions
        that is **≤1.0 point a round**, so a tolerance of 1-2 is a real test; at
        the default `--k 16` it is ≤160 points a round — a single round can shove
        an anchor further than the whole ladder spans, and a tolerance of 1.0
        will essentially never be met, so the run always burns `--max-rounds`.
        Lower the K *and* the tolerance together, or neither.
      - **Measured on the real anchors at `--k 0.1`**: over ~15 rounds the ten
        anchors moved 2.10 points each on average, but the worst moved 6.3 — and
        it is that 6.3 the stopping rule reads. Their mean stayed put (1537.74 →
        1537.82), as a flat K guarantees.
    - **`--workers N` shards the round across N local processes**, reusing
      `global_arena.play_sharded` -- playing parallelises perfectly, rating does
      not, so the sessions come back and are rated here in one place. It is the
      only thing that makes a convergence run finish in hours: a round is
      `anchors x --games-per-model / players` sessions, and at today's 50 anchors,
      `--games-per-model 20` and `--hands 1000` that is ~167 sessions and
      **167,000 hands, ~35 minutes on one core**. `--min-rounds 50` therefore
      means ~29 hours single-threaded, and `--max-rounds 500` well over a week.
      (These scale linearly with the anchor count, and the anchor count has
      grown: the same figures read 116 sessions and ~24 minutes when the set
      held 35.)
      A projection of exactly this is printed at startup, so a two-day round
      cannot be launched by accident.
      - **The projection has to include the shard startup, and once did not.**
        Every round spawns `workers` fresh processes and each imports torch and
        loads *every* anchor before playing a hand — `workers x anchors`
        checkpoints pulled over one NFS mount — so the cost goes with their
        product, not with one shard's startup. **Measured on the live run: 235 s
        a round** at 35 anchors and `--workers 20`, against the 73 s the hands
        themselves account for. The projection said ~1 minute, 4x optimistic,
        which defeats the point of printing one; it now adds
        `workers x anchors x 0.0039` minutes, calibrated on that single point.
        At the 50 anchors the set holds today that formula gives **~6 min a
        round at `--workers 20` -- so `--min-rounds 50` is ~5 h and the default
        `--max-rounds 500` is ~2 days.** Note how the startup term dominates:
        3.9 of those 6 minutes are 20 shards each loading 50 checkpoints, and
        only ~2 are hands, so *more* workers stops helping well before the cores
        run out. It is also why the number grows faster than the anchor count --
        both terms scale with it.
      - **Never raise `OMP_NUM_THREADS` instead.** `OMP_NUM_THREADS` sizes
        torch's *intra-op* pool -- how many threads split one tensor operation --
        and here that operation is a batch-of-one forward pass through a small
        MLP. Measured on an **idle** 32-core box, 6-max hands per second:

        | configuration | hands/s | cores used |
        |---|---|---|
        | 1 process, 1 thread | 72.1 | 1 |
        | 1 process, 15 threads | 80.1 | 15 |
        | **15 processes, 1 thread each** | **1,102.9** | 15 |

        Fifteen cores' worth of threads buys **1.11x**; fifteen processes buy
        **15.3x**. Threads are a ~14x worse use of the machine, and the reason is
        that torch is only ~32% of a hand's 12.4 ms (11.9 actions x ~334 us); the
        other two thirds is `encode_observation`, the legal-action masks and the
        betting state -- Python bytecode no torch thread can touch, and which the
        GIL would serialise even if they were Python threads. Shards are therefore
        launched with `OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`, since torch may
        route an op through MKL rather than OpenMP and each reads its own variable).
        - **Under load it is far worse than 1.11x, and that is the normal state.**
          A 15-thread forward measured **3,346 us** against 374 us single-threaded
          while another 15-thread process held the same cores -- a 9x collapse from
          pure contention. On a fleet machine running 10-20 workers you are always
          in that regime, so the safe setting is one thread per process, always.
          (An earlier note here quoted that 3,346 us as an idle-machine property.
          It is not: it was contention, and the throughput table above is the
          real argument.)
    - **There is no regroup any more.** The anchors used to be re-sorted into
      `benchmark_<N>/` strength bands when a run ended; the frozen set is now one
      flat directory (see "How a model becomes a benchmark anchor"), so
      `--regroup`/`--no-regroup` and `regroup_benchmark` are gone.
    - **A flat K is what keeps the anchors' mean exactly fixed**, and that is
      the load-bearing property: `pairwise_elo_delta` is zero-sum for a flat K
      (a pair's two deltas are equal, opposite and divided by the same count),
      so a closed population playing itself redistributes rating without moving
      its centre. Every other model in the store is rated against these, so a
      drift in their mean would silently shift ~9,400 ratings. The per-experience
      K schedule is *not* used here, since it breaks that cancellation. Verified
      on real anchors: the mean read 1450.5 before and after every round.
    - Every rating is written to a timestamped JSON backup under
      `global/benchmark_arena_backups/` before anything is touched, and
      `--restore <file>` puts them back (verified end to end, `games` and
      `frozen` preserved). `--dry-run` plays and reports without writing.
- **Pruning trigger: the real on-disk population reaches
  `--global-trigger-size` (10,000).** That is the whole rule: whenever
  `len(discover_population(...))` — the number of files in `checkpoints/models/`
  (one per network) — is at or above the trigger at the end of a merge, a
  pruning pass runs; after it removes ~a quarter of the eligible models the count
  drops below the trigger, and it fires again when the backlog has grown back to
  it. It is *not* keyed to how many models the ledger has rated (the ledger only
  grows by the handful each round samples, the real population is the whole
  backlog). Pruning is serialised by an exclusive `__prune__` lock (30-minute
  TTL) that stops two machines pruning — or numbering the same new benchmark
  series — at once; it does **not** stop ordinary merges, and the population is
  recounted *after* taking it so a pass that another machine just finished is not
  repeated on a stale count.
- **Elimination physically deletes the checkpoint file. This is deliberately
  destructive, not bookkeeping.** `_eliminate` removes `--global-eliminate-
  fraction` (~25%) of the *eligible* members: the non-frozen ones whose `games`
  is at or above the `--global-protect-percentile` (**25th**, lowered from the
  50th) of games played among
  the models rated so far — never below it, since a rating built on a handful of
  games is not evidence. Measured on the real population, that bar is still met
  with room to spare: the 25th percentile of games among the 9,588 rated
  non-frozen models on disk is **627 rated sessions** of 1,000 hands, against
  962 at the 50th. It was lowered because eligibility is a *percentile of
  games*, so any draw that seats some models more often than others — and
  `tiered_draw` deliberately does — makes the well-played eligible sooner
  and leaves the rarely-drawn tail permanently immune, which would have the
  pass eat the middle of the population instead of its bottom. The eligible set
  goes from 50% to 75% of the population, and the accepted cost is that a pass
  removes ~1,800 models instead of ~1,200, so prunes fire less often and bite
  harder. **There is no absolute minimum-games floor** (an earlier
  `--global-min-rated-games`, default 30, was removed on purpose); with nothing
  rated yet the percentile is undefined and no pruning happens. Each doomed model
  is then locked individually and re-read fresh, so one being rated at that
  moment is skipped rather than deleted from under a merge. Every confirmed model
  has its `.pt` removed by `delete_checkpoints` (resolved through a fresh
  `discover_all_copies()`, never through `ref`) and its member file removed.
  **Nothing is promoted to the benchmark any more** — see the next section.
- **How a model becomes a benchmark anchor** (`global_arena.py::
  add_benchmark_candidates`, called by `train.py` right after the run's
  population round). Pruning promotes nothing: the old rule — five doomed models
  drawn at random into a fresh `benchmark_<N>/` series — is gone, along with
  `promote_to_benchmark`, `next_benchmark_dir`, `DEFAULT_BENCHMARK_REFRESH`,
  `poker-loop --no-arena-after-prune` and `benchmark_arena`'s regroup. The rule now:
  - Population = non-frozen members whose checkpoint is in `models/`. A model
    qualifies when its `games` are **above the 90th percentile**
    (`BENCHMARK_GAMES_PERCENTILE`) of that population's games and its rating is
    **more than 10 points** (`BENCHMARK_MARGIN`) above the best anchor.
  - Candidates are taken from the lowest rating upwards and each must also clear
    the previously added one by more than the margin, so anchors stay at least 10
    apart instead of a cluster joining at once.
  - An added model is *moved* into the flat `checkpoints/benchmark/`, every other
    copy deleted, `frozen=True` at the rating it holds, `ref` rewritten. No
    re-settling by hand. Held under the
    `__prune__` lock so a prune cannot delete it mid-move.
  - **Every addition requests a `benchmark_arena` run**
    (`global_store.request_benchmark_arena`, claimed by one supervisor between
    two generations, `loop.py::run_requested_benchmark_arena`): `--tolerance 1`,
    `--max-rounds 500` (`--min-rounds 50`), logged to `arena-gen<N>.log`. A failed
    run is not retried. **The arena always uses the hyperbolic K staircase**
    (`--k` is gone): each anchor's K comes from its own `games`, so a session is
    no longer zero-sum and the anchors' mean is no longer exactly fixed — veterans
    at the bottom tier barely move, a new anchor moves most.
  - **Do not use `delete_checkpoints` on an anchor**: `discover_all_copies` only
    looks in `models/`, so it silently deletes nothing from `benchmark/`. A hand
    thinning of the anchors once removed the member files and left the `.pt`
    behind, and the live fleet re-registered two of those orphans as frozen
    anchors at 1500 with ~40 games. Delete the file *and* the member together,
    and never one without the other.
- **The benchmark was thinned and filled by hand on 2026-10-01.** The 50 anchors
  were moved out of `benchmark_1..10/` into the flat directory; 22 were removed so
  adjacent ratings are ≥10 apart (kept the lowest of each cluster); 14 models
  from outside the top 100 and with the most games were added to bring gaps to
  10-20, including two above the old top anchor. Three gaps (88, 85 and 26.9
  points) remain, where no eligible model sits. Benchmark readings from before
  this are not comparable with later ones.
- **`ref` is a real, maintained path, not a hint.** A member's `ref` is where its
  checkpoint is *now*, relative to the same root the rounds run from
  (`checkpoints/...`). It is kept current three ways: `promote_to_benchmark`
  rewrites it when it moves a file into a `benchmark_<N>`; every round that
  plays a model refreshes its `ref` from the disk discovery it already did
  (`_apply_session`); and `repair_member_refs(global_dir, root, machine=...)`
  sweeps all members at once for the models that have not played since their
  file moved (a migration of the store, say). Before this, promotion left
  every promoted model's `ref` pointing at its old path and ~2,000 refs had gone
  stale, which made `discover_global_top_models` silently drop two of the six best
  models from the GUI's default table. That function now also falls back to
  finding the checkpoint by label when a `ref` is stale, so it degrades to "a
  bit slower", never to "missing".
- **Hard rule: `registry.json` under `checkpoints/global/` is a snapshot, never a
  source of truth.** The truth is the member files. Write it only through
  `write_snapshot`, and never read it to *decide* anything that changes state
  (merging and pruning read `load_global_registry`); it may be minutes stale, and
  anything hand-edited there is overwritten by the next snapshot. The corollary
  is that **whatever writes member files has to refresh the snapshot itself**, or
  its work is invisible to every reader: a training merge does it as its last
  step, and `benchmark_arena` and `poker-elo` force one after every round (see
  their sections above), because both can run for hours or days without any
  other writer coming along to do it for them.
- **Ghosts**: `prune_ghost_members` drops the member files of models whose
  checkpoint exists nowhere any more, compared against the on-disk label set and
  not against `ref`. An *empty* label set (an unreadable volume) drops nothing,
  so a transient NFS problem cannot wipe the ledger.
  - **The label set is a snapshot, and a snapshot goes stale — this deleted live
    models' ratings.** `apply_pending_population_rounds` listed the disk once at
    the top and then merged pending sessions, which on a busy fleet runs for tens
    of minutes (27 measured on a 13,014-session round), while five machines kept
    publishing new models and registering them. Each newcomer appeared in
    `list_member_labels` but not in the snapshot and was swept as a ghost: a live
    model, freshly rated over 500 sessions against the frozen series, silently
    reset to rating 1500 with zero games. The signature in a `poker-elo` log is
    bursts of "fantasmi rimossi" landing **exactly on the longest rounds** and
    nothing on the short ones — 20 ghosts on a 1,235 s round, 0 on a 595 s one.
  - **How to tell a false positive from a real ghost, and it is easy**: nothing
    in this project deletes a checkpoint except a prune. With pruning off
    (`poker-elo` without `--prune`) and the population under
    `--global-trigger-size`, **any ghost at all is a false positive.**
  - **The fix is to ask about one label at the moment it matters.** The sweep
    now re-lists the disk immediately before running (instead of reusing the
    pre-merge snapshot), and `checkpoint_on_disk(root, label)` re-checks each
    candidate **after its lock is taken and immediately before removal**, which
    is the only instant the answer is authoritative; the snapshot is then just a
    shortlist. It looks in `models/<label>.pt` and in `benchmark/` recursively,
    because a promoted anchor lives in a series directory — a check that looked
    only in the store would delete every anchor's rating, which is the fixed
    point the whole scale rests on. Three tests cover it, including that a
    genuinely missing checkpoint is still dropped: the re-check must narrow the
    sweep, not switch it off.
- **Why pruning exists**: nothing else ever removes a model, so without it the
  store grows forever. The volume already held roughly 23,000 raw checkpoint files
  before the store was deduplicated into one file per network (~9,800 distinct
  networks), and models are added every generation. Real physical deletion, gated
  behind an Elo signal reliable enough to trust (the percentile-of-games
  protection) and a real disk-population trigger, is what bounds it.
- **Never delete a `poker-train` CLI flag while a supervisor is running. It is
  not a retirement, it is a fleet outage.** A `poker-loop` supervisor is a
  process that lives for weeks, and every generation it launches its workers
  with the flag list *its own* loaded `loop.py` knew about at startup. The
  moment `poker-train` stops recognising one of those flags, argparse exits 2
  and **every worker of every running supervisor dies at that supervisor's next
  generation** — and because a generation that launches 20 workers and has them
  all exit immediately takes ~90 seconds, the loop then spins through empty
  generations forever. Nothing reports this as an error: `loop_state.json` keeps
  being written every few seconds, `--status` says "training", the dashboard
  looks alive. The only visible symptom is the generation counter climbing
  absurdly fast.
  - **It happened on 2026-09-28.** `--global-benchmark-refresh` was removed from
    `train.py` at 20:47 (`--pool-random-share` and `--pool-fresh-n` had been
    retired earlier and were in the same forwarded list). The five training
    machines finished their last real generation between 23:08 and 00:29 —
    whenever each happened to reach a generation boundary — and then burned
    roughly 250 empty generations each overnight. Diagnosis, in order: the
    generation numbers had advanced by hundreds in six hours (impossible for a
    real ~3 h generation); `supervisor.log` showed `worker N uscito con codice 2`
    for all 20; the worker log held nothing but an argparse usage block ending
    `error: unrecognized arguments`.
  - **The fix has to be deployable without touching the machines**, because the
    NFS server has no SSH access to them. Making `train.py` *accept and ignore*
    the flags again is enough: workers load the current code at every launch, so
    the fleet recovered by itself within one generation (~90 s), verified by
    watching `iter` lines reappear in the live worker logs. Restarting the
    supervisors is the real fix, but it is the slow one and it is manual.
  - **The rule, encoded in `train.py` and pinned by
    `test_a_flag_removed_from_train_stays_parseable_for_running_supervisors`**:
    a flag removed from `poker-train` stays parseable, with its value discarded,
    until every supervisor in the fleet has been restarted. The behaviour is the
    new behaviour; only the parsing is backwards compatible. The list is
    `--pool-random-share`, `--pool-fresh-n`, `--global-benchmark-refresh`,
    `--benchmark-every`, `--benchmark-hands`, `--benchmark-seed` and
    `--self-share`. Note that the e2e
    contract test (every flag the loop passes must be one train accepts) cannot
    catch this — it compares the *current* loop against the *current* train,
    while the whole problem is a supervisor running last week's loop.
  - **The rule was broken once on purpose, on 2026-10-01.** `--eval-hands`,
    `--eval-rotate-every`, `--series-sessions` and `--series-hands` were **deleted**
    rather than retired, at the user's explicit decision, with a fleet reset
    scheduled immediately afterwards. That is the only safe way to do it: the
    moment the code lands, every running supervisor's next generation dies at
    argparse, and it stays dead until the supervisors are restarted. If you are
    reading this because the generation counter is climbing absurdly fast and
    `--status` still says "training", check `supervisor.log` for
    `uscito con codice 2` first — and then check whether someone landed a flag
    removal without the reset.
- **Operational note when changing this code**: processes already running an
  older version keep using the code they loaded. After the switch from a single
  `registry.json` to member files, an old-code merge still writes only
  `registry.json` (now merely the snapshot), so whatever it merges is lost;
  nothing is corrupted. Workers pick up new code at their next `poker-train`
  launch, so it clears itself within a generation.

## The ranking is wrong at the top (measured)

The global Elo orders the population correctly over large gaps and **not at all**
near the top, which matters because everything the loop does rests on that order:
`pick_parents` draws from the top 100, and each run keeps whichever checkpoint
rated best. Both choices are therefore close to random among good models.

Measured with **duplicate decks** — 3 copies of one model against 3 of another,
then the same shuffles replayed with the teams swapped, so card luck cancels.
`Table` consumes its `random.Random` only to shuffle, once per hand, so a fixed
seed deals the identical sequence no matter how the betting goes; `(a - b) / 2`
over the two arrangements is the skill effect alone. A null control of the #1
against itself returned **+2.0 bb/100 (t = 0.43)** over 48,000 hands, so the
estimator invents nothing.

| Pair | Elo gap | Real result | |
|---|---|---|---|
| #1 vs last (5844) | 313 | +490 bb/100 | as ranked |
| #1 vs #1000 | 96 | +26 bb/100 | weakly as ranked |
| #1 vs #20 | 49 | +37 bb/100 | as ranked |
| #1 vs #100 | 65 | +6 bb/100 | **indistinguishable** |
| **#20 vs #100** | **16** | **−37 bb/100 (t = −6.1)** | **inverted** |

The order is not even monotone: #100 holds up better than #20 does. Consistent
with this, the population's best models have not improved — the top 1% sits flat
at ~1578 across every age cohort, while inheritance itself demonstrably works
(+49 global rating for inheriting workers over fresh ones, over 1,121 runs).

**Re-measured on 2026-09-29, after the 1,000-hand sessions and the lower K.**
The same pair, #20 against #100 of the live ranking (`duel_power`, 16 mirrored
streams, 320,000 hands):

| | then | now |
|---|---|---|
| Elo gap | 16 | **54.9** |
| real result for the higher-ranked model | **−37 bb/100 (t = −6.1)** | **−4.4 bb/100 (t = 5.9)** |
| per-hand noise | — | sigma = 5.2 bb |

**The sign is still wrong and the magnitude has collapsed by a factor of eight.**
Both facts matter, and the second is the one that changes what to do about it.

- **It is not an Elo bug any more, it is a power limit.** The two models are
  4.4 bb/100 apart, and at that gap a session of `DEFAULT_HANDS_PER_GAME` = 1,000
  picks the stronger one **58.1% of the time** (measured; 60.5% predicted). Ninety
  per cent would need 23,016 hands a session and ninety-five per cent **37,915**.
  No K schedule fixes that: the evidence per session simply is not there, which is
  the same conclusion "How long must a session be?" reached for a wider pair,
  now at the spacing that actually occurs at the top.
- **What it means for `pick_parents`, which is why the question was asked.** It
  draws **uniformly from the top 100**, so what matters is whether that *band* is
  assembled correctly, not whether #20 outranks #100 within it. An error of
  4.4 bb/100 between two models a random draw treats identically is close to
  irrelevant; the old 37 bb/100 inversion meant the band itself was wrong. On
  this evidence the parent selection is no longer the bottleneck it was.
- **Caveats, both real.** This is **one pair**, exactly as the original finding
  was; characterising the whole top needs several. And the per-hand noise came
  out sigma = 5.2 bb here against 11.4 bb for the old #1-vs-#1000 duel — two
  similar strong models produce a much quieter game, which is part of why the
  duplicate-deck estimator resolves 4.4 bb/100 at t = 5.9 from 320,000 hands.
- **Consequence for the "Observation v2" gate.** That work was explicitly gated
  behind verifying this ranking. The verification is done and the answer is
  *partly*: the coarse ordering is sound and the top band is assembled about
  right, while fine ordering inside the top is beyond what 1,000-hand sessions
  can resolve and will stay so. That is good enough for a selection rule that
  draws uniformly from a band, so the gate should no longer block the encoder
  work.

**What was tried and rejected**, each with numbers in the TODO section or here:
weighting the Elo update by chip margin (two variants, both turned out to be a
lower effective K in disguise); training against more real opponents instead of
copies of the learner (`--opponent-probability` 1.0 with `--self-share` 0.1
scored **−29.4 bb/100** against the frozen set versus **+11.6** for the default,
8 seeds each — the parameter is now exposed, and the default is the right value).
**Note what that measurement now sits against**: `--self-share` has since been
removed entirely at the user's decision, so the field is permanently at the
equivalent of 0.0, and the one experiment on that axis moved it *and*
`--opponent-probability` together, so it cannot say which of the two caused the
−29.4. Worth re-measuring if the fleet's `ancore/100` drops after the change: the
clean comparison is the current build against one with `--opponent-probability`
left alone, which is a code change now rather than a flag.
What was kept: **lowering `DEFAULT_K_SCHEDULE`**, the only change that improved
ordering in simulation, and it has since been lowered much further. The schedule
went 24/16/12/8/6 → 12/8/6/4/3 (the simulated halving, 0.889 → 0.935 against a
population whose true skill is known) → 4.0/1.0/0.5/0.1 → 3.0/1.0/0.3/0.1, whose
thresholds also moved out (100/1,000/10,000) as the population's games-played
distribution grew, and then to today's **six tiers, which prepend a burn-in
(24 below 10 games, 8 from 10) to those four**. The burn-in is a separate
question from the ordering measured here — it governs how fast a *new* rating
reaches its neighbourhood, not how stably the settled ones are ordered — and it
was added because a rating starting at 1500 could not travel at all inside a run
(see "K falls with experience, and now rises first"). The simulated grid only reached 6/4/3/2/2 (0.9604), and lower K improved
the ordering monotonically at every point on it, so the direction is measured;
the specific numbers below that grid were chosen by hand and have **not** been
simulated. The cost is the one the simulation also showed: new models take
correspondingly longer to climb to where they belong, and ~1,000 of them arrive
every generation.

**If this is picked up again**, the promising direction is to stop ranking the
top by Elo at all: run a duplicate-deck round-robin among the top 100 — the
technique above, which resolved a 16-point gap the ordinary rating had inverted —
rarely and out of band like `benchmark_arena`, and let `pick_parents` read that
order instead. It is the expensive option, which is why lowering K came first.

## How long must a session be? (`rl/duel_power.py`)

Everything the ranking does assumes that the model finishing a session ahead is
the better one. That is not a fact, it is a probability, and it depends entirely
on the session length. `poker-train`, the population round and every rated
result rest on it, so it is worth having measured rather than assumed. Run by
hand, nothing imports it:

    OMP_NUM_THREADS=1 python -m pokerlab.rl.duel_power --model-a '#1' --model-b '#100'

- **A long stream *is* many sessions.** Stacks are reset after every hand in
  every evaluation path here, so hands are i.i.d. and a stream of 10,000 hands
  is 1,000 sessions of 10 or 10 of 1,000 depending only on where it is chopped.
  Every session length in the grid is read off the same played hands. This is
  what makes the whole question answerable in minutes instead of days.
- **Duplicate decks give the ground truth.** "How often does the stronger model
  win" presupposes knowing which one is stronger, and measuring *that* the
  ordinary way is exactly as noisy as the thing being studied. Each stream is
  played twice from the same seed with the two models' seats swapped. Verified
  directly on 30 hands with two different models: the dealt card sequences are
  identical in both arrangements despite completely different betting, because
  `Table` touches its `random.Random` only to shuffle, once per hand. Note what
  mirroring does and does not do -- the pooled mean is exactly the mean of the
  paired differences, so it does not move the estimate, it sharpens it, by
  roughly an order of magnitude.
- **The normal approximation extrapolates.** With a per-hand edge `mu` and noise
  `sigma`, a session of `H` hands is won with probability `Phi(mu*sqrt(H)/sigma)`,
  and both are measured from the *total* hands played. So the report shows a
  measured frequency wherever there are enough blocks and a prediction
  everywhere, including at lengths nobody played. The two columns side by side
  are also the check on the approximation: they must disagree at ten hands (the
  per-hand delta is heavy-tailed) and agree from a few hundred on. Measured:
  52.1 vs 52.1 at 10 hands, 56.2 vs 56.5 at 100, 70.3 vs 69.7 at 1,000, 85.9 vs
  87.5 at 5,000.
- **The null control works**: the same model against itself returns an edge of
  exactly +0.00 bb/100 and every session an exact 50/50.

**Measured on this fleet**, #1 against #1000 of the global ranking (a 136-point
Elo gap), 3 seats each at a 6-max table, 320,000 hands mirrored:

| | |
|---|---|
| real edge | **+18.6 bb/100 per seat** (+/- 3.5, t = 10.5) |
| per-hand noise | sigma = **11.4 bb** |
| a 1,000-hand session picks the stronger one | **69.7%** of the time |
| hands needed for 95% | **10,219** |
| hands needed for 99% | **20,440** |

The middle row is the one that matters: `DEFAULT_HANDS_PER_GAME` is 1,000, so
**almost a third of every rated session moves the ratings the wrong way** even
for a pair this far apart. For two adjacent models the edge is a fraction of
18.6 bb/100 and the hands needed scale with its *square*, which is the
quantitative form of "The ranking is wrong at the top" above: halving the edge
quadruples the hands. It also says the right direction for any fixed hand budget
is fewer, longer sessions -- the same conclusion already reached on
`global_arena.DEFAULT_HANDS_PER_GAME`, now with a number attached.

- **The distribution, not just the frequency.** `format_distributions` draws
  what 100 sessions of each length actually looked like. Same two models
  throughout: at 100 hands the results span -228 to +265 bb/100 and the stronger
  model takes 52 of them; at 1,000 hands the spread is 37.7 and it takes 73; at
  10,000 the spread is 13.0 and it takes 29 of 32. The median barely moves (+8,
  +23, +17) because it is the real difference between the two networks -- only
  the scatter around it shrinks, as `1/sqrt(hands)`. Sessions are drawn
  round-robin across streams, not as consecutive blocks of one, since part of
  what makes two sessions differ is which seats each model drew.
- **`--mode field` measures what a population round actually measures**: one
  seat each and the rest filled from the frozen anchors, instead of half the
  table each. On a short probe the per-hand noise came out ~20 bb against ~11
  for the 3v3 duel, which would mean roughly **four times** the hands for the
  same confidence -- plausible (a 3v3 duel averages over three seats) but
  measured on one short stream, so re-measure before resting anything on it.
- **The statistics half is pure Python and the play half imports torch lazily**,
  the same argument as `features.py` and `pool_registry.py`: the arithmetic that
  decides what the experiment *concludes* belongs in the ordinary test suite
  with no extra dependency. `tests/unit/test_rl_duel_power.py` (21 tests, no
  torch) covers the blocking, the histogram binning, the round-robin sampling,
  the Wilson interval, and the round trip between `win_probability` and
  `hands_for_probability`.
- `--save` writes the per-hand chip deltas and `--load` re-analyses them without
  replaying: the hands are the expensive part by orders of magnitude, so trying
  a different grid of session lengths should cost nothing.

## GUI (`gui/app.py`, `players/gui.py`)

Tkinter was chosen over a local web app specifically to avoid adding a new
dependency (Flask, etc.) — it ships with Python. Architecture:

- **The bot builder seats trained models, and only trained models** (there is
  no hand-coded bot catalog any more — see "Heuristic bots, removed"):
  `AddBotDialog` offers a "Modello addestrato" combobox listing the 20
  best-rated checkpoints found across every machine's pool, newest ratings
  first, and nothing else. The spec it produces is
  `{"key": "model", "path": ...}`, which `_bot_spec_to_key_string` renders as
  `model:<path>` — the same string the CLI understands, so nothing in
  `build_players` needs a GUI-specific branch. "Aggiungi" is disabled when no
  checkpoint is found, rather than offering a choice that would fail on
  confirm.
- **Setup screen bot builder**: `SetupFrame` no longer has a free-text bot
  field or a "number of players" field. Instead `self.bot_specs: list[dict]`
  (each `{"key": "model", "path": ...}`) drives a row of rectangles (one per
  configured bot, each with a "-" to remove) plus a trailing "+" that opens
  `AddBotDialog` to pick a trained model. `num_players` is *derived*
  (`human_seats + len(bot_specs)`, capped by `GameConfig`'s own 2-9
  validation) rather than typed separately — keeps the visual builder as the
  single source of truth. `_bot_spec_to_key_string` turns a spec back into
  the exact `model:<path>` string `build_players` and `validate_bot_key`
  already understood, so neither needed to change.
  - **`bot_specs` is pre-populated, not empty, when the setup screen opens**:
    one instance each of the top 6 models by rating in the *global* Elo
    registry (`cli/play.py::discover_global_top_models`, reading
    `checkpoints/global/registry.json`, the snapshot of the per-model files), so a first-time player sees a
    sensible table immediately instead of an empty one needing six manual
    "+" clicks before anything can start. `AddBotDialog`'s
    `discover_trained_models` reads the same global ranking (its top 20), so the
    default table and the picker agree on what "best" means. Still just a starting
    point — the player can remove any of these or add others through the
    same dialog as always, and if the global registry is missing, empty, or
    has fewer than 6 members with a file still on disk, the list is
    correspondingly shorter (never an error).
  - **No `checkpoints/` at all (a PC that only cloned the repo, like the
    Windows box the GUI runs on): the models come from `top_models/`**
    (`cli/play.py::discover_top_models_folder`, the folder
    `rl/push_top_models.py` commits: `<label>.pt` + `ratings.json`). It is a
    fallback inside `discover_global_top_models`/`discover_trained_models`, so
    the default table, the bot picker and the spot screen's advisors all get it;
    it applies only when the registry yields nothing, never mixed in.
    `push_top_models` passes `fallback_dir=None` -- it must not "find" the very
    copies it is replacing -- and so must any test asserting an empty result,
    because the tests run from the repo root, where `top_models/` exists.
- **Busted players disappear from the table**: `TableFrame._hide_seat`
  calls `grid_remove()` (not just blanking the labels) on a seat's box once
  its stack hits 0, called from both `_render_observation` (seat missing
  from `Observation.seats`) and `_render_final_stacks` (`final_stacks[seat]
  == 0`). `grid_remove()` hides the widget but does **not** clear its
  Canvas contents — a hidden seat's card canvases still have stale drawn
  items sitting in them, which bit a throwaway verification script that
  checked `canvas.find_all()` without first filtering out
  `seat in self._busted_seats`. Keep that in mind before trusting a canvas
  item count on a seat that might be hidden.

- **`GuiPlayer`** (`players/gui.py`) is the only new integration point, and
  it's tiny: `act()` puts a `GuiEvent("your_turn", (observation, legal_actions))`
  onto a shared `queue.Queue`, then blocks on its own private `decisions`
  queue until something pushes an `Action` back. Same shape as `ManualPlayer`
  swapping `input()` for a blocking `queue.get()` — `Table`/`engine` are
  completely unaware a GUI exists.
- **Threading model**: Tkinter's mainloop must own the main thread, so
  `Table.play_session`-equivalent play (`_run_session` in `gui/app.py`) runs
  on a background daemon thread instead. The background thread is only ever
  blocked waiting on the human's own decision (bot decisions are
  instantaneous); the GUI thread polls the shared event queue every 100ms
  via `self.after(100, self._poll_events)` and updates widgets from there.
  Never touch Tkinter widgets from the background thread directly.
- **No mid-hand "stop" mechanism** — the daemon thread just dies when the
  window closes. Deliberately not built (see `_run_session`'s docstring);
  add one later only if it's actually needed.
- **Actions are reported from the engine, not from a `Player` wrapper**
  (`Table.on_action_applied` -> `gui/app.py::ActionReporter`). This was a
  `SteppingPlayer` wrapping every non-human seat and publishing an
  "action_taken" event from inside `act()`; that wrapper has been deleted,
  because it is structurally incapable of showing what an action did. The
  engine applies the action *after* `act()` returns, so a wrapper only ever
  sees the state from before it: the table showed every move as not yet
  made and was corrected only by the *next* action -- which meant the last
  action of a street was **never** drawn at all, the next event being the
  new community card. Measured, three players calling preflop: the log said
  "P0: call 2" while the table still read `pot 3, P0 bet 0`.
  `on_action_applied` carries `{hand_id, seat, player_id, name, action,
  record, observation}` with the Observation from *after* the action, and
  `ActionReporter` publishes it and only then paces -- so what the pause
  holds on screen is the finished action, chips in and pot updated. The
  Observation is built only when a hook is installed, so training pays
  nothing for it.
  - **The human is reported the same way**, which is why `GuiPlayer.act()`
    no longer publishes anything itself: one path, one place, post-action
    state for everyone. `ActionReporter` skips the pause for the human's
    own seat -- they already "stepped" by clicking.
  - **Pacing lives in the GUI, not the engine**: `ActionReporter` sleeps
    `BOT_ACTION_DELAY_SECONDS` (1.0) after a bot action, or blocks on the
    shared `step_gate` while step mode is on, on the session thread. Without
    the delay a whole hand's events land between two of the GUI's 100ms
    polls and render as one jump; that is what makes an unattended bot-only
    session watchable at all, not just step mode. `delay_seconds` is
    overridable so tests do not sleep for real.
  - **A new community card needs its own beat too**
    (`NEW_STREET_DELAY_SECONDS`, 1.0, in `on_street_dealt`). The engine
    deals the street and the next player acts microseconds later, so
    without it the card and that first action arrive in the same poll and
    read as one event -- the card looks like it came *instead of* the
    action. Measured before and after: flop and first flop action both at
    3.05s, versus the flop alone at 3.09s and the action at 4.01s.
  - Toggling step mode off pushes one release onto `step_gate` (in case a
    bot is mid-block right then) and toggling it back on drains any
    leftover release first -- without that pairing, either a bot could stay
    stuck until one extra "Avanti" click, or a stale release could silently
    skip the next real pause. Step mode can also be preset from
    `SetupFrame` before a session starts (`start_session(..., step_mode=...)`),
    since toggling it only after `start_session` already returned can lose
    the race against instant bot-only hands finishing before the checkbox
    click even lands (learned the hard way while testing this).
- **"Who am I" is a property of the GUI, never of the Observation being
  rendered.** `_render_observation` takes the human's seat from
  `self._human_seat`, not from `observation.my_seat`. Most Observations it
  draws belong to whichever player just acted, so reading `my_seat` labelled
  *that bot* "(tu)" and passed `is_me=True` to `_draw_seat_cards`, which
  drew its hole cards as an empty slot -- the seat looked like it had
  folded, every time it acted, and in spectator mode one bot always wore the
  marker. The same fix applies to the full-size "Le tue carte" panel: with
  no human seated it now draws nothing, instead of flipping between
  opponents' hands.
- **The dealer's whole seat box is tinted**, not just tagged
  (`_set_dealer_seat`, `DEALER_BACKGROUND`). ttk widgets take their
  background from their *style*, never from a `bg` option, so this needs
  two parallel families of styles -- `Seat.TLabelframe`/`.TLabel`/`.TFrame`
  and `Dealer.*` -- set up once in `_configure_seat_styles`; restyling only
  the `LabelFrame` would leave the name/stack/bet labels and the inner card
  frame sitting on default-grey patches inside a yellow box. The card
  Canvases are classic `tk` widgets and are recoloured directly. The
  highlight is set from `hand_started`'s `button_seat` and refreshed from
  every `Observation.button_seat`, so it cannot drift out of step with the
  "D" label below.
- **The dealer button is its own bold label above the seat box**
  (`seat_widgets[seat]["dealer"]`/`"dealer_label"`), not folded into the
  name text the way "(tu)" still is -- a plain "(D)" suffix on a 16-char-wide
  name label was easy to miss. `_render_observation` sets it from
  `seat_info.is_button` every render; `_hide_seat` grid-removes it alongside
  the seat's frame (a busted seat must not leave a stray "D" behind), and
  `_reset_seats_for_new_hand` clears it up front for the same stale-data
  reason as the other per-seat fields (see the note on that function).
- **Opponent-card "spy" toggle**: `Table` accepts an optional
  `on_hand_started` callback (fired once per hand, right after blinds are
  posted, with `{hand_id, button_seat, sb_seat, bb_seat, small_blind,
  big_blind, hole_cards}`) purely as a spectator/debug hook -- nothing in
  `Player`/`Observation` was touched to add this, since leaking opponents'
  hole cards into the normal per-player interface would be a real design
  flaw for the eventual RL section. The GUI wires this hook to cache
  `hole_cards` and a checkbox flips whether `TableFrame._draw_seat_cards`
  draws opponents' actual cards or a face-down back.
- **The all-in runout used to be invisible, and that needed an engine
  hook.** When everyone still in the hand is all-in, `Table` calls no
  `Player` at all for the remaining streets -- `_run_betting_round` returns
  immediately -- so a spectator that only watches `Player.act()` saw
  *nothing* between the last bet and the final stacks: the board never
  appeared and the hand looked like it skipped its own ending. `Table` now
  takes a second optional spectator hook, `on_street_dealt`, fired right
  after each street's community cards are dealt with `{hand_id, street,
  community_cards, betting_closed}`; `betting_closed` is read *before*
  `start_new_street_betting` (`len(actionable_seats()) <= 1`). Two
  deliberate choices: the flag lives in the engine because only the engine
  knows it, but the *pacing* (`ALL_IN_STREET_DELAY_SECONDS`, on the session
  thread in `start_session`'s callback) lives in the GUI, exactly like
  `ActionReporter`'s own delay -- `Table` stays free of display concerns.
  The GUI also turns over everyone still contesting the pot at that point,
  as a real poker room does once there is nothing left to protect.
- **Who has mucked is read off the last Observation**
  (`TableFrame._folded_seats()`), which is the state from *after* its
  action, so a fold is visible in it immediately and the GUI keeps no
  fold-tracking of its own. There is no Observation at all when a runout
  begins with no action having been taken -- every seat all-in on its own
  blind -- and then nobody has folded, which is the one case the method
  guards.
- **Every participant's cards are shown for `SHOWDOWN_REVEAL_SECONDS`
  (1.0) when a hand ends**, then the table is cleared by a `self.after`
  timer (`_reveal_hand_end` / `_clear_after_showdown`). Two things that
  look like details and are not: a seat that just *busted* is deliberately
  **not** hidden during the reveal -- a player who shoved and lost is
  precisely the one whose cards are worth seeing, and `_hide_seat` would
  delete them the instant they became showable -- so the hiding moved into
  the clear; and the timer is cancelled on the next `hand_started`
  (`_cancel_pending_clear`) and guarded with `winfo_exists`, or it fires
  into a frame the "Torna al menu" button has destroyed.
- **The raise panel is scrollable and has pot-fraction shortcuts.** A mouse
  wheel notch over the slider moves the bet by `WHEEL_STEP_BIG_BLINDS` big
  blinds, and a row of buttons (`POT_FRACTION_PRESETS`: 30/50/66/100%) sets
  it to `_pot_fraction_raise_to`, which is the standard sizing -- match the
  outstanding bet first, *then* bet the fraction of the pot that matching
  produced, giving a raise-*to* level (what `Action.amount` means for
  BET/RAISE, see `Action`'s docstring). Every route into the amount goes
  through one `apply` that clamps to `[min_amount, max_amount]`, so a
  pot-sized raise the stack cannot cover simply offers the legal maximum;
  the buttons only *propose* a size, confirming is still a separate click.
  - **Gotcha, and it silently inverted the scroll direction once**: X11
    reports the wheel as `<Button-4>`/`<Button-5>` and Windows/macOS as
    `<MouseWheel>` with a signed `delta`, so the reflex is one handler
    reading `event.num`. That field is not dependable -- measured on this
    build, a synthesised `<Button-4>` arrives with `num=8`. Each sequence
    is bound to its own direction instead, which is also what makes it
    testable with `event_generate`.
- **Action-by-action + showdown log**: `_log_hand_start` (from
  `on_hand_started`) prints the blind postings, `_log_action` (from every
  "action_taken" event, human included) prints one line per decision via
  `_describe_action` -- which computes CALL/ALL_IN amounts from the
  `Observation` rather than `Action.amount`, since the engine deliberately
  leaves that field meaningless for those two action types (see
  `Action`'s docstring) -- and `_format_hand_summary` (at "hand_complete")
  prints a showdown section with revealed hole cards whenever 2+ seats
  didn't fold, plus "Pot vinto da: ..." and final stacks. `_describe_action`
  reads the chips off the engine's own `ActionRecord`
  (`stack_before - stack_after` for CALL/ALL_IN, whose `Action.amount` the
  engine deliberately leaves meaningless; `record.amount`, the street total,
  for BET/RAISE) rather than re-deriving them from an Observation.
  **The table view used to lag the log by one action; it no longer does** --
  see the `ActionReporter` bullet above for why that needed an engine hook.
  - **The summary names the winning five-card combination**, in Italian,
    next to every showdown contestant and again on the "Pot vinto da:"
    line (`_describe_combination`). Two judgement calls: the Italian
    category names live in `gui/app.py`, not in the evaluator -- they are
    presentation, and the evaluator is the one file in this project
    deliberately left alone -- and a pot taken down by everyone folding
    prints "(senza showdown)" with no combination at all, because nothing
    was shown and the board may not even be complete, so naming a winner's
    hand there would be inventing one. `_order_for_display` re-sorts
    `HandRank.best_five` (which is in combination order, so a pair of tens
    prints as "Ah 7h Tc Th Kd") into pair-first, kickers-descending.
- **Footer/menu-button layout**: the "Torna al menu" button's footer is
  packed with `side="bottom"` specifically so it always reserves its space
  regardless of window height -- it was disappearing (silently clipped)
  when packed as a normal top-to-bottom widget below the log's
  `expand=True` `Text` widget on a window shorter than the summed content
  height. Any future widget added below existing content should go through
  the same `side="bottom"` treatment rather than plain `.pack()`.
- **Reuses `cli/play.py`'s `build_players`** (extended with an optional
  `human_player_factory` param, defaulting to `ManualPlayer`) and
  `validate_bot_key`, so the GUI's bot-selection field accepts the exact
  same `model:<path>` spec as `poker-play --bots`, with zero duplicated
  parsing logic.
- **Known environment quirk, worked around in code**: this project's own
  Python install has `TCL_LIBRARY`/`TK_LIBRARY` pointing at the wrong
  folder (`<prefix>/lib/tcl8.6` instead of the real `<prefix>/tcl/tcl8.6`),
  which makes a bare `tkinter.Tk()` fail with "Can't find a usable
  init.tcl". `gui/app.py` sets those env vars itself (only if unset and the
  real folder is found) before importing tkinter — see
  `_fix_tcl_tk_library_paths`. If GUI tests ever fail with that exact
  error on a fresh machine, this is almost certainly why.
- `GuiPlayer` is unit-tested (thread-safety of the block/unblock handoff)
  in `tests/unit/test_gui_player.py` without needing a real Tkinter window,
  and `ActionReporter`'s pacing and step-gating in `test_gui_app.py`. Some `TableFrame`/`SetupFrame` *logic* (seat
  hiding, spy-mode card visibility, bot-spec formatting) turned out to be
  perfectly testable headlessly too and is covered in
  `tests/unit/test_gui_app.py` — turns out `tk.Tk()` + `.withdraw()` works
  fine without a real display on this machine, so "no meaningful way to
  test Tkinter" was an overstatement; what's still not covered is full
  click-driven end-to-end flows (those stay manually smoke-tested by
  driving `PokerGuiApp` headlessly and invoking rendered buttons
  programmatically, or counting a card Canvas's drawn items via
  `canvas.find_all()` — but see the busted-seat caveat above about hidden
  canvases first). **Gotcha**: creating/destroying multiple separate
  `tk.Tk()` instances in rapid succession *within one process* was flaky
  on this machine's Tcl/Tk install (an intermittent "couldn't read file
  ...button.tcl" error despite the file existing) — `test_gui_app.py`
  works around it with **one session-scoped `app` fixture in
  `tests/conftest.py`** that every GUI test file shares, destroying only its
  own frames (not the Tk root) between tests. It used to be one
  module-scoped root per file; with four GUI files plus the selector's own
  `tk.Tk()`, a run failed at setup about one time in three with
  `invalid command name "tcl_findLibrary"`. Never add another `Tk()` or
  `PokerGuiApp()` fixture: request `app`.
  **Second gotcha, and it fails an unrelated test file**: that fixture must
  `gc.collect()` on the main thread before tearing the root down. Left to
  chance, the frames' `StringVar`s are collected later by whatever thread
  happens to trigger a GC -- a worker in `test_rl_global_arena.py`, say --
  and `Variable.__del__` then raises "main thread is not in main loop" as
  an unraisable exception, which pytest reports against *that* test.
- **Spot advisor mode** (`gui/spot.py` logic, `gui/spot_table.py` seating,
  `gui/spot_view.py` Tk frame; button "Chiedi ai modelli (spot)" on the setup
  screen). The screen is an **oval table with nine chairs** (`CHAIRS`), yours at
  the bottom and always seated; "+" on an empty chair seats an opponent, with a
  name and a stack, and every player has a "D" button for the dealer. Everything
  is buttons: the board is five card buttons in the middle, your two cards sit
  under your chair, and each card is chosen in a popup (`CardPicker`) by rank and
  then suit, with cards already used disabled. When it is someone's turn their
  action buttons appear on their chair; on yours, with both cards set, the top 5
  global-Elo models answer by themselves.
  - **Your box hangs from under the felt and the canvas grows to fit it**
    (`_build_chair` anchors chair 0 "n"; `refresh` ends in `_fit_canvas`, which
    enlarges the canvas to its contents and the window with it, within the
    screen, never shrinking). It used to be centred on its chair like the
    others, and being the tallest -- cards plus actions, 314 px at 125% scaling
    -- it ran 32 px past the 650 px canvas and cut the last action off.
  - **Chairs may overlap; nothing is ever cut off, and a press brings a box to
    the front.** A top chair grows upwards when its actions appear (~100 px to
    ~245) and used to leave the canvas: `_fit_canvas` now slides the whole table
    right/down when anything sticks out of the top or left, measured from each
    chair's *requested* size (`_content_bbox` -- `canvas.bbox` lags right after
    the buttons are rebuilt). Overlap between neighbours is accepted, at the
    user's choice: a press anywhere on a box, buttons included, `lift()`s it
    (`_make_raisable`: a per-chair bindtag placed *first* on every descendant,
    so the clicked button still works; re-applied after every refresh because
    the action buttons are rebuilt), and whoever is to act is lifted
    automatically. The tag is a process-unique counter, never `winfo_id()`:
    Windows reuses handles, so a new chair inherited a destroyed one's binding
    whose `lift()` failed and aborted the script. Tk delivers no pointer events
    to a withdrawn root, so the tests for this `deiconify()` for their duration.
  - **Every amount on the spot screen is in big blinds** (at the user's
    request): pot, bets on the chairs, action buttons, the raise field and its
    wheel, the log, the models' advice and the status line, written as the
    client writes them ("18,5 BB", comma decimals). The engine still counts
    chips at fixed blinds of 1/2 (`ENGINE_SMALL_BLIND`/`ENGINE_BIG_BLIND`, the
    blinds the models trained at; one chip = 0,5 BB) and converts only at the
    edges: `spot.format_bb`/`bb_number` to show, `spot.parse_bb` to read input
    (comma or dot, rounded to the nearest chip). The small/big blind entries
    are gone -- "Blind 0,5 / 1 BB" is a fixed label -- and stacks are typed in
    BB (default 100 = 200 chips). `describe_action` takes an optional
    `big_blind`; without it it still speaks chips. Gotcha: never `.capitalize()`
    a label with "BB" in it -- it lowercases the rest.
  - **The dealer button is the order of play.** `TableLayout.order()` lists the
    occupied chairs clockwise (as seen on screen) from the dealer, and those are
    engine seats 0, 1, 2, ... — which is why no seat number is ever typed. Chairs
    are fixed, so a player can be inserted between two others only where a chair
    is free between them.
  - **Changing the seating clears the actions** (add, remove, move the button):
    the script is a flat list in the engine's own turn order, so it means
    something only for one seating. Stacks, blinds and cards do not clear it —
    the replay truncates whatever stops being legal and says so.
  - **Typed values apply on Return only, never on FocusOut**: a FocusOut binding
    rebuilds the action buttons when one entry is clicked after another, and the
    click on the second is lost.
  - The board is filled in order (a slot is disabled until the one before it is
    set); the engine deals random cards for any street the board does not cover,
    and the screen warns.
  - **The models are never shown a hand-built `Observation`.** The spot is a
    *script* replayed through a real `Table` (`replay`), from scratch after every
    edit (microseconds); cards are placed by `ArrangingRandom`, whose `shuffle`
    arranges the deck at the positions `deal_positions` names (pinned by a test).
    The engine decides whose turn it is and what is legal, so a script cannot be
    illegal by construction; one made stale by editing the table (fewer players,
    smaller stack) is cut at the first refused/orphaned action and reported via
    `SpotState.invalid_from`. The button is always seat 0, so choosing the seat
    chooses the position (`position_names`).
  - `advise` bypasses `RLAgentPlayer` (it samples and drops the distribution) and
    returns the full softmax; torch is imported lazily and the models are loaded on
    the first "Chiedi", not when the screen opens.
  - Sanity-checked on the live top 5: 72o folds, AA/AKs call -- but at ~100%
    everywhere, which is the collapsed low-entropy policies, not a pipeline bug.
- **Cards are drawn on a `tk.Canvas`, not image files** (`gui/cards_canvas.py`):
  a plain rectangle plus rank/suit text (Unicode ♠♥♦♣, red for hearts/
  diamonds, black for spades/clubs) for a face-up card, a solid-fill
  rectangle for a face-down back, and a dashed outline for an undealt slot.
  No Pillow/image-asset dependency needed for this. The human's own hole
  cards are drawn full-size above the seat grid; still-live opponents show
  small face-down backs next to their seat box (folded/busted/self show an
  empty slot there instead).

## Vision (`vision/`)

Goal: fill the spot screen (dealer, your cards, the board, the opponents' actions)
from the poker client on screen. Started with the cards, and **only as far as
getting an image of them**: how to recognise the cards is decided once there are
real crops to look at.

- **`regions.py`** (pure Python): `Region(left, top, width, height)` in pixels of
  the virtual desktop — the numbers `mss` uses, so they can be negative on a
  multi-monitor setup — and `RegionConfig`, saved atomically to
  `vision_data/regions.json` (`HOLE_CARDS`, `BOARD`; outside `checkpoints/` on purpose, next to the crops). A missing or corrupt
  file, or one bad entry in it, is "no region", never an error.
- **`capture.py`**: `list_monitors`, `grab_monitor`, `grab_region` (a BGR `numpy`
  frame, what OpenCV expects), `frame_to_png_bytes`/`save_png` (through
  `mss.tools`, **no Pillow**). `mss`/`numpy` are imported inside the functions and
  a missing extra raises `VisionUnavailable`, so the GUI opens without it. Works
  with both `mss.MSS` (new) and `mss.mss` (old, deprecated in 10.x).
- **`selector.py`**: `select_region(parent)` hides pokerlab's own window,
  photographs the monitor and shows it dimmed full screen; drag a rectangle,
  Return confirms, Esc cancels. **Tk coordinates and screenshot pixels are not
  assumed equal** (on a scaled Windows display they differ by the scale factor):
  `scale_to_pixels` converts by comparing the canvas with the picture. The
  picture goes to Tk as a PNG through `PhotoImage(data=...)`. A click without a
  drag is not a selection and leaves the overlay open.
  **It must always be closable**, which it once was not: the overlay is
  override-redirect, some Linux window managers give such a window no keyboard
  focus, and `focus_force` before the window is mapped does nothing — so Return and
  Esc were ignored on a full-screen window holding the mouse. Now `present()` waits
  for the window to be mapped (with a time limit; **never `wait_visibility`**, which
  blocks forever if it never maps), then grabs and focuses it; there are also
  on-screen Conferma/Annulla buttons, double-click to confirm, right-click to
  cancel, and a 180 s self-cancel.
- **The vision screen is compact and scrolls.** No instructions on it any more,
  neither at the top nor in the sections (removed at the user's request, for
  space). The sections sit in a canvas (`VisionFrame.body`) under a fixed title
  bar and scroll with the wheel anywhere over them: the wheel is bound with
  `bind_all` because the event goes to the widget under the pointer (a button,
  a label), `_on_wheel` ignores events outside this screen or from a popup,
  and `destroy` unbinds so no other screen inherits it. Both wheel families
  are bound, each to its own direction (`<MouseWheel>` delta, X11
  `<Button-4/5>`).
- **In the GUI: a screen of its own** (`gui/vision_view.py::VisionFrame`, the
  "Collect vision data" button on the main menu, next to "Chiedi ai modelli").
  It used to be a block inside the spot screen and was moved out at the user's
  request, so collecting examples does not crowd the table; `vision_view`
  reuses `CardPicker`/`place_near_pointer` from `spot_view`, never the other way
  round. Tests in `tests/unit/test_gui_vision.py`. Its buttons: "Zona mie carte", "Zona board",
  "Anteprima" (shows what is captured now) and "Salva carte" / "Salva board"
  (one zone each, so a crop of the cards does not need a board zone set; a
  "Salva tutte" was removed at the user's request) — the examples the
  recognition is built from.
- **Dealer button: zones and examples, no recogniser yet.** The "Dealer (6
  giocatori)" section sets one zone per seat (`regions.dealer_region_name`,
  `dealer_6_<seat>` in `regions.json`; seat 0 is you at the bottom, then
  clockwise on screen from your left, the spot screen's chair order; only 6-max
  so far, the positions differ at 9). "Salva dealer" captures every zone set and
  opens `DealerLabeler`: all crops side by side, each "Presente"/"Non presente",
  at most one present (none is fine, between hands). On "Conferma" *every* crop
  is written -- the absent ones are examples too -- to **`vision_data/dealer/`**
  with a `{"zone", "dealer": bool}` label (`labels.save_dealer_label`), a folder
  and a label shape of its own so neither dataset is mistaken for the other.
  Card actions ("Salva carte/board", the test section) capture only the card
  zones, never the dealer ones that share the regions file.
- **Player seats: zones and examples, no recogniser yet.** The "Giocatori (6
  giocatori)" section sets one zone per seat around the player box (avatar,
  name, stack; `regions.player_region_name`, `player_6_<seat>`, same numbering
  as the dealer, seat 0 = you included since you can fold too). "Salva
  giocatori" opens a `SeatLabeler` with one state per seat
  (`labels.SEAT_STATES`): `in_gioco`, `fuori` (folded, or just joined and
  waiting), `sit_out` (seated but sitting out; added at the user's request) and
  `libero` (empty seat); every crop is written on Conferma to
  **`vision_data/players/`** with `{"zone", "state"}`
  (`labels.save_player_label`). Each seat starts from the state it was given
  last time (`VisionFrame.last_seat_states`), since between captures only a
  seat or two changes; the very first capture starts blank and Conferma waits
  until every seat has a state.
  - **Recognition: `vision/seats.py`, colour rules plus one template**
    (`python -m pokerlab.vision.seats` lists every crop where rule and label
    disagree -- the place to find a mislabelled example). Measured on 72 crops:
    an opponent in the hand shows pink-magenta card backs (share up to 0.59, 0
    for every other state; threshold 0.10); seat 0 in the hand is read from the
    white of *your* face-up cards (0.08-0.09 live, 0.00-0.02 folded and dimmed;
    threshold 0.05 -- not their colours, which a hand of two black spades
    lacks); an empty seat shows a green chair outline (0.026, 0 otherwise);
    sit-out is the grey "SIT OUT" pill, matched by shape against the lettering
    of labelled sit-out crops (`sit_out_templates`), so that one state needs an
    example; anything else is "fuori". Result: 70/72, and both misses were
    labels carried over wrongly by the remembered default, not rule errors.
    Caveat: the two sit-out examples are the same seat, near-identical, so the
    template's reach to other seats is unverified (others score <= 0.26
    against a 0.6 threshold).
  - **Wired into the spot screen's 0.5 s scan** (`ScreenReader._read_seats`,
    `ScreenReading.seats`). Seated = every seat but `libero`/`sit_out`
    (`screen_reader.seated_seats`): a folded player is still at the table, and
    one picture cannot tell "out, folded" from "out, waiting to join", so a
    newcomer is seated until the next hand. **Seating is applied only at a new
    hand** (button moved, new hole cards) or on the first reading -- mid-hand
    it would wipe the actions being typed every time someone stood up -- and
    only on the chairs mapped to client seats. **Folds show at once**:
    `_mark_seat_states` tags a seated player read `fuori`/`sit_out` ("fold",
    grey border), but never inserts a FOLD into the actions, because *when* in
    the sequence they folded is unknown. The button is put back on its chair
    if seating moved it.
  - **One screen grab per reading** (`capture.grab_regions`: the bounding
    rectangle of every zone, captured once and cut up). Fourteen
    `grab_region`s -- 2 card zones, 6 dealer, 6 player -- took 233 ms on the Tk
    thread every 2 s, a visible stutter; one grab takes ~60 ms.
  - **Dealer and player zones are separate and read separately**: the button
    only in `dealer_*` zones, the seat state only in `player_*` zones. They
    once *looked* merged because the dealer zone of seat 5 had been redrawn
    round that player's whole box (149x111 against the usual ~38x37) -- the
    two sections' buttons were both labelled "Posto 5". Now they read
    "Gettone N" / "Giocatore N", the selector overlay names the zone being set
    (`select_region(label=...)`), and a dealer zone with a side over
    `regions.DEALER_ZONE_MAX_SIDE` (80) is flagged in the vision screen and
    **skipped by `ScreenReader`**, since a player box contains a gold stack
    chip that the dealer rule would take for the button.
  - `SeatLabeler` is the one labelling window for per-seat states;
    `DealerLabeler` is it with "Presente"/"Non presente" and an
    at-most-one-present check.
- **Bets and pot: zones and examples, no recogniser yet.** The "Puntate e
  piatto (6 giocatori)" section sets a `pot` zone and one `bet_6_<seat>` per
  seat (`regions.POT`, `regions.bet_region_name`), round where the number is
  written. "Salva puntate" opens `AmountLabeler`: the crops side by side, pot
  first, each with a text field for the amount **exactly as written on screen**
  -- separators and K/M suffixes kept, "" for nothing shown -- because how the
  client writes numbers is what a reader has to learn (`labels.AMOUNT_PATTERN`
  allows digits, `.`/`,` and an optional K/M). Written on Conferma to
  **`vision_data/amounts/`** as `{"zone", "text"}` (`labels.save_amount_label`).
- **Reading them: `vision/amounts.py`** (`python -m pokerlab.vision.amounts`,
  leave-one-out). The client writes the amount in big blinds, white on a dark
  pill, **always followed by "BB"**, decimals with a **comma** ("18,5 BB");
  labels hold the number only. White blobs left to right; the last two are the
  "BB" (a reading not ending in two B-height blobs is not trusted); a blob under
  `COMMA_MAX_HEIGHT` of the digit height is the comma; each other blob is a
  digit, 1-NN by normalised correlation (`recognize.normalise_glyph`). A zone
  with no blob as tall as a digit (`DIGIT_MIN_HEIGHT`) is "no amount" -- chips
  lying there have white stripes, 3-5 px against ~18.
  - **Two-stage digit match, measured.** The card ranks are the same typeface:
    on their own they read 14/14 bet digits. But pooled with the bets' digits
    they lose (a bet "5" out-matched a card "6", 0.73 vs 0.70), so: the bets'
    own digits first, and only when their best match is below `OWN_SURE` 0.8
    (right answers scored >= 0.865, wrong <= 0.73) the card ranks alone. That
    covers digits 2-9 never seen in a bet; 0 and 1 have no card source (a ten
    is one merged glyph).
  - **Loops decide 3 vs 8** (`glyph_holes`, applied in `_Matcher`): candidates
    are restricted to examples with the same number of closed loops -- 8 has
    two; 0, 4, 6, 9 one; 1, 2, 3, 5, 7 none, measured identical for every
    example in the bets *and* the card ranks. Correlation alone was a coin toss
    between them (same right half): even with six 3s labelled, a 3 one pixel
    taller than the others scored 0.825 on the 3s and 0.830 on the 8s. A ±1 px
    shift-tolerant match tied them exactly and grey-level glyphs halved the
    margin on every other digit, so both were rejected. If no example shares
    the loop count (a stray mark broke a loop) all are considered again.
  - **Measured: 152/154 leave-one-out**, and both misses are *labels* left
    empty on crops that plainly show "7 BB" and "1 BB" (same capture,
    `…025541`), not reading errors.
  - The vision screen's amounts "Anteprima" captions each zone with its reading.
- **Stacks: zones, examples, a reader awaiting its first crops.** The "Stack (6
  giocatori)" section sets `stack_6_<seat>` zones (`regions.stack_region_name`)
  and labels crops like the amounts (`AmountLabeler`, `labels.STACKS_DIR` =
  `vision_data/stacks/`, same `{"zone", "text"}` shape). **The stacks are
  written in yellow** (the user's word), so `amounts.text_mask` picks the
  lettering's colour by zone -- `yellow_mask` for `stack_*`, white otherwise --
  and then everything is shared: digits are compared as black-and-white
  shapes, so white bet digits and yellow stack digits lend each other examples
  (`build_reader` is fed both folders; `python -m pokerlab.vision.amounts`
  checks both). Both assumptions made before any stack was seen held on the
  first 24 crops: the yellow band (`YELLOW_HUE` 15-40, S > 80, V > 140)
  isolates the lettering cleanly, and a stack is written "103,5 BB" like a bet
  (`STACK_SUFFIX_BLOBS` = 2). **Measured: 177/178 bets+stacks leave-one-out**,
  the miss being a label ("0" on an empty seat). Evaluate the two folders
  *together*: run on `vision_data/stacks` alone, the only stack "0" had no
  example left once excluded and read as 6 -- an artefact of the check, not
  of the app, which pools both. Live, a full reading with stacks takes ~76 ms.
  - The spot screen sets each seated player's stack **at a new hand only**,
    with the seating (`SpotFrame._apply_stacks`): the stack shown plus the
    chips already in front (the blinds, preflop). Never mid-hand: the chair's
    field (now labelled "inizio BB") is the hand's *starting* stack, and the
    engine takes every bet off it itself, so rewriting it after a bet would
    take that bet off twice. Stacks are still *read* every 0.5 s, for a
    check: each chair shows "resta X BB" -- what the engine leaves it -- and
    "≠ schermo Y BB" when the stack read now differs (`_screen_stacks`,
    redrawn whenever it moves), which is how a missed or misread action shows.
    (The user once took the missing live stack for a reading that "did not
    update after the bets"; it was being read, just not shown.)
- **The spot screen rebuilds the actions from the screen** (at the user's
  choice over a show-and-compare mode): `gui/action_sync.py`, pure Python over
  `gui/spot.py`, run on every 0.5 s reading after cards/seats/dealer are applied
  (`SpotFrame._sync_actions`). `ScreenReader` now also reads every bet zone and
  the pot, in big blinds (`ScreenReading.bets`/`.pot`); the spot converts them
  to chips with its own big blind.
  - **States, not events.** Several players can act between two readings, so
    nothing tries to *see* an action: while the engine's player to act can be
    proven to have acted already, that action is appended and the engine asked
    again; the first seat with no proof stops it until the next reading. Proof:
    out of the hand -> FOLD (after first matching chips already put in);
    chips in front above the engine's -> CALL / BET / RAISE to that amount /
    ALL_IN; the board further on than the engine's street -> the street closed,
    so CALL facing a bet, CHECK otherwise; someone due *after* this seat has
    acted -> with chips unchanged it can only have been a CHECK. An action the
    engine refuses, or an amount it cannot make, stops it without inventing.
  - **Your countdown bar proves the checks.** A check leaves nothing on the
    table, so a round of checks used to be discovered only when the next card
    came. The vision screen's "Mio turno (barra del tempo)" section collects a
    `turn_timer` zone with "Presente"/"Non presente" crops
    (`vision_data/turn/`, `labels.save_turn_label`); `vision/turn.py` reads it
    with a **fixed colour rule** (it started as a provisional 1-NN on
    thumbnails, replaced once the crops were seen): the bar is bright green on
    a dark track, measured 28-39 % saturated-and-bright pixels with it and
    exactly 0 % without, so `PRESENT_SHARE` is 1 % -- low enough for a bar
    nearly spent -- and *any* hue counts, since countdown bars commonly turn
    yellow then red. Needs no examples (`python -m pokerlab.vision.turn` checks
    it on the labelled ones; 4/4). `ScreenReading.my_turn` feeds
    `TableView.my_turn`: while it is up and the engine still waits on a seat
    before yours, that seat acted -- a CHECK with chips unchanged, a FOLD if
    out; facing a bet with nothing changed it is still not guessed.
  - **No "Chiedi ai modelli" button inside the spot screen** (removed at the
    user's request): the models answer by themselves whenever it is your turn
    with both cards known. The main menu's "Chiedi ai modelli (spot)" button,
    which opens the screen, stays.
  - **The client's pot excludes the current bets** (measured: empty preflop
    with the blinds out, 8.5 BB with 8 and 2 BB still in front). So the check
    `_pot_check` compares pot + bets in front with the engine's pot and prints
    "ATTENZIONE piatto" on a mismatch -- which is how a reconstruction gone
    wrong shows.
  - **Limits, by construction**: a raise swept into the pot before any reading
    saw it is invisible (the street then reads as calls; the pot check flags
    it); stacks are not read, so the spot's stack setting decides what an
    all-in is; the hero's own action is taken the same way, from the screen.
    Manual edits are overridden by the screen on the next reading, since the
    screen is the source of truth. Tests: `tests/unit/test_gui_action_sync.py`
    and the spot-frame test in `test_gui_screen_reader.py`.
- **Dealer recognition: `vision/dealer.py`, a fixed colour rule.** The button
  is a gold disc with a "D" on green felt: share of gold pixels (HSV H 15-35,
  S >= 100, V >= 120) >= `PRESENT_FRACTION` 0.20. Measured on 24 labelled crops:
  present 58-66 %, absent exactly 0 % (felt H 67-70); 24/24, and read right
  live on a seat with no "present" example at all -- the rule needs none.
  `find_dealer` takes the most gold zone and flags `ambiguous` if several pass.
  `python -m pokerlab.vision.dealer` checks it on `vision_data/dealer/`. Shown
  in the vision screen's dealer "Anteprima" (gold share per seat).
- **The spot screen reads the dealer in the same 0.5 s scan** (`ScreenReader`
  `._read_dealer`, `ScreenReading.dealer` = client seat). Client 6-max seats map
  to the spot's 9 chairs by nearest angle (`spot_table.CLIENT_SEAT_CHAIRS[6]` =
  0,2,3,4,6,7; seat 3, straight across, falls between chairs 4 and 5 and 4 was
  picked). A move puts the button on that chair, **seating a player there if it
  was empty** (the button is always in front of someone), and clears the
  actions (a new hand). Compared with the *last seat it was seen on*
  (`_last_dealer`), so the button vanishing between hands and reappearing on
  the same seat is not a move, and a manual dealer correction sticks. An
  ambiguous reading is not applied. The dealer is read even when no card crop
  exists yet, since its rule needs no examples.
- **A crop is written only together with its label, on "Conferma"**
  (`vision_view.py::CropLabeler`, one window per captured zone, in turn). The
  capture is held in memory; the window shows it and the visible cards are
  chosen with the same `CardPicker` buttons, **left to right as on screen**, the
  slots filling in order so a label has no gaps. Conferma stays disabled until
  the count is valid: hole cards 0 or 2, board 0/3/4/5 (`VALID_COUNTS`). The
  "Nessuna carta visibile" tick is a real example (folded, street not dealt) and
  is confirmed the same way; "Annulla" writes nothing. Output goes to
  **`vision_data/crops/`** (`labels.CROPS_DIR`), deliberately *outside*
  `checkpoints/`: `<zone>-<timestamp>.png` plus a same-named JSON
  (`vision/labels.py`, pure Python: `{"zone", "cards": ["Th", "2c", ...]}`).
- **The client runs on the user's Windows PC**, where pokerlab's GUI also has to
  run. Keep pokerlab's own window off the captured zones. Not verified on Windows
  yet; capture and the preview/save path were checked on Linux/X11.
- **Card recognition: `vision/recognize.py`, pattern matching, no network.**
  `python -m pokerlab.vision.recognize` scores it on `vision_data/crops/` by
  leave-one-image-out (each crop read by a recogniser built from all the others);
  `--image <png>` reads one crop. The client uses a **four-colour deck**
  (spades dark grey, hearts red, diamonds blue, clubs green) on a fixed layout:
  - **Slices**: board 5 equal fifths; hand **not** halved but cut at 0.47
    (`SLOT_EDGES`). The right hole card overlaps the left and its rank starts at
    ~83/170 px, so an exact half left a 2-px sliver in the left slice that was
    read as its rank: 11 of 31 hands wrong, all on the left card.
  - **Empty slot**: white-pixel share below 0.06 (empty measured up to 0.027,
    cards from 0.10). The board stops at its first gap; a hand is 2 cards or none.
  - **Rank**: topmost white blob *starting in the left 40% of the slice*, plus
    blobs touching it on the same row (a ten's two digits), padded square to
    32x32 and matched by normalised correlation, 1-NN over every labelled glyph
    (left, right and board pooled -- the slight rotations are covered by
    examples, not modelled). The left-40% and touching rules come from a client
    animation (a player's reaction) drawn white over a 9h, higher than the rank,
    which the looser "whole row" rule merged into a ten.
  - **Suit**: the card's background colour by a **fixed rule** (`SUIT_RULES`,
    at the user's request, replacing a nearest-labelled-colour match): spades
    black, diamonds blue, clubs bright green, hearts red; the symbols on top are
    white. Read as the median non-white colour around the rank glyph. Measured
    bands (HSV): spades S<=38 V 57-64, hearts H 3-4, clubs H 59-60 S ~200,
    diamonds H 106-108; the felt sits at H 66-90 S 45-165, so clubs are told
    from it by hue *and* saturation. The rule's bands are much wider than
    measured and still do not overlap. A colour matching none reads as suit "?".
    Rank and suit are matched **separately**, so a card never collected whole
    (Qh, at the time) is still read, and a suit needs no examples at all.
  - **Measured on 65 real crops**: 189/189 cards; the right rank correlates
    >= 0.91 and beats the best wrong rank by >= 0.14 (closest: 5 vs 6, 3 vs 8);
    with the colour rule, 205/205 on 73 crops.
  - **No test section in the vision screen any more.** A "Test vision model"
    section that read every zone on demand was removed at the user's request:
    readings are checked in the spot screen, which reads every zone every 0.5 s.
    What stays is an **"Anteprima" per section** (`_preview` card zones only,
    `_preview_dealer`, `_preview_players`, all through `_show_preview`), which
    shows that section's zones side by side as captured now, captioned with the
    quick reading (gold share for the button, the seat state for a player).
  - **The spot screen reads the screen by itself every 0.5 s** (`SCREEN_POLL_MS`, 2 s until
    the user asked for it faster;
    no button, at the user's request): `gui/screen_reader.py` (no Tk) captures
    both zones and recognises them; `SpotFrame.apply_reading` puts them into the
    spot. Measured live: 31 ms a reading (350 ms the first, which builds the
    recogniser), so it runs on the Tk thread with no worker. Rules, each tested:
    - **Applied only when the *screen* changed** (`diff_reading` compares with
      the previous reading, not with the spot), so a manual correction sticks
      until the client shows something else.
    - **Different hole cards = new hand: the actions are cleared** (they belonged
      to the last hand); seats, stacks and dealer stay. Recognised across a fold
      (cards, none, other cards), and the same cards returning after a flicker
      are not a new hand (`last_hand`).
    - **Doubtful readings are not applied**: a suit "?", a card read in both hand
      and board, a board of 1-2 cards (the flop mid-animation). The line under
      the situation shows the reading, the time and any such problem.
    - The recogniser is rebuilt when the number of crops changes, and the zones
      are re-read every tick, so both can be updated from the vision screen
      without reopening this one. A missing vision extra stops the polling.
    - `SpotFrame(read_screen=...)` defaults to **off**; only `app.show_spot`
      turns it on. Tests must leave it off (`show_spot(read_screen=False)`), or
      they photograph whatever screen they happen to run on.

## Setup and running things

Classic venv (not uv — deliberate user preference):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Linux (AlmaLinux 9 training box)

Same classic venv/pip workflow, with one wrinkle: `requires-python = ">=3.13"`
and AlmaLinux 9 ships only 3.9 and 3.12. `uv` is used *solely* to fetch a 3.13
interpreter — the environment itself is still a plain venv, and `uv pip` /
`uv run` are not used:

```bash
uv python install 3.13
"$(uv python find 3.13)" -m venv .venv
.venv/bin/pip install -e ".[dev,rl]"
```

The code byte-compiles cleanly under 3.12 and uses no 3.13-only API, so
relaxing `requires-python` to `>=3.12` and using `/usr/bin/python3.12` is a
viable fallback if `uv` is ever unavailable. Don't do it silently — it changes
project metadata.

**Always install the CUDA build of torch (plain `pip install torch` from
PyPI), never the CPU-only wheel from `download.pytorch.org/whl/cpu`**, even on
a box with no GPU. The CUDA build runs fine on CPU (`torch.cuda.is_available()`
is just `False`, and `poker-train` already defaults `--device` to `cpu` in that
case), so the same venv keeps working when training moves to a GPU machine.
The cost is only disk: the venv lands at ~5.4 GB, versus a couple hundred MB
CPU-only.

**Set `OMP_NUM_THREADS=1` for training.** The policy is a small MLP and the
real bottleneck is the pure-Python engine, so torch's default intra-op
parallelism just oversubscribes: measured on a 32-core box, 3 iterations x 128
hands took 8.5 s wall / **2m16s CPU** at the default 32 threads, versus 7.9 s
wall / 7.4 s CPU pinned to one. Identical results (same seed), ~18x less CPU —
which is what makes it practical to run many independent training runs in
parallel across the cores instead of one run hogging all of them.

Throughput on that box, single-threaded: roughly **50 hands/s** 6-max, so a
200-iteration x 512-hand run plus its every-10 evals takes ~45 minutes.

```bash
mkdir -p checkpoints/logs
OMP_NUM_THREADS=1 nohup .venv/bin/poker-train \
    --iterations 200 --hands 512 --players 6 --seed 42 \
    --eval-every 100 --archive-every 25 \
    > checkpoints/logs/run-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Logs go under `checkpoints/logs/`, which is already covered by the
`checkpoints/` gitignore entry.

#### Parallel sweeps (many seeds at once)

"Use all the threads" means **N single-threaded runs, one per core** — not one
run with N torch threads, which is measurably slower (see above). One run pins
one core, so the box takes about as many concurrent runs as it has cores.

Memory, not cores, is what you have to measure first, and the naive number is
wrong: a `poker-train` process shows ~793 MB RSS, but most of that is torch's
shared pages, so the *incremental* cost of one more run is only **~454 MB**
(measured by launching six and differencing `free -m`). On the 32-core / 32 GB
box that means cores bind before memory does. Always measure the increment with
a small probe batch before saturating — do not divide total RAM by RSS.

Each run in a sweep needs **its own `--checkpoint`, its own `--scratch-dir` and
its own `--archive-prefix`**. The checkpoint is obvious (they would overwrite each
other). The scratch dir and prefix matter because the run id is only
second-resolution: runs launched in the same second would share a scratch file
and would try to publish under the same name (the later one finds it taken and
its model is not published). Note that, unlike the old per-run pools, every run
in a sweep draws its opponents from the *same* shared store and publishes into
it, so runs do see each other's published models; if a sweep must stay isolated
for a controlled comparison, give each its own `--models-dir` and `--global-dir`
(and `--no-global-round`).

```bash
for s in $(seq 1 29); do
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 setsid nohup poker-train \
      --iterations 200 --hands 512 --players 6 --seed $s \
      --eval-every 100 --archive-every 50 \
      --checkpoint "checkpoints/seeds/agent-seed$s.pt" \
      --scratch-dir "checkpoints/seeds/scratch-seed$s" \
      --archive-prefix "seed$s" \
      > "checkpoints/logs/seed$s.log" 2>&1 < /dev/null &
done
```

Progress across a whole sweep: `for s in $(seq 1 29); do echo "$s
$(grep -c '^iter' checkpoints/logs/seed$s.log)"; done`. Note `setsid` makes bash
print `[1]+ Done` for each job as it detaches — that is job-control noise, not a
failed run; count the live processes with
`pgrep -f '\.venv/bin/python \.venv/bin/poker-train' | wc -l` instead. **Pass `PYTHONUNBUFFERED=1`** (or `python -u`)
whenever the run is redirected to a file: `poker-train` reports progress with
plain `print`, and Python block-buffers stdout when it is not a tty, so without
it the log stays at 0 bytes for minutes at a time and the run looks hung when
it is perfectly healthy. `setsid` keeps the run alive independently of the
shell that launched it.

```powershell
pytest -v                                    # full suite (712 tests)
pytest --cov=pokerlab                        # with coverage
ruff check .                                 # lint

poker-gui                                    # Tkinter desktop app

poker-train --iterations 200 --hands 512     # self-play PPO (needs the `rl` extra)
poker-train --resume --device cuda
# every run draws its own opponents from checkpoints/models/ and publishes its
# best model there when it ends
poker-train --pool-models 8 --archive-every 20
# a validation round is --eval-sessions rated sessions of 1000 hands against the
# drawn opponents (their ratings are fixed reference points; only the learner's
# rating moves), and the count continues into the round against the frozen anchors
poker-train --pool-models 20 --eval-every 100 --eval-sessions 10
poker-train --benchmark-sessions 500                # the round that sets the published rating

poker-loop --workers 8 --iterations 100      # continuous training until stopped
poker-loop --status                          # per-worker table
poker-loop --status --watch 30               # live, redraws in place
poker-dashboard                              # browser view of every machine, http://127.0.0.1:8770
poker-dashboard --host 0.0.0.0 --iterations 1000   # reachable from the other machines
# --status/--watch need --iterations to match the running loop's, or every
# progress bar is scaled against the wrong target (run.sh passes 100 to both)
# the held-out models in checkpoints/benchmark/ (and its benchmark_<N>/ series)
# are never seated in training
./run.sh stop                                # stop after the current generation (can be ~20 min)
./run.sh kill                                # stop everything NOW, whatever stage it is in
./run.sh kill-dry                            # ... list what that would kill, and kill nothing
# every worker also tries a cross-machine population Elo round at the end of
# its run (on by default -- see "Population-wide Elo and pruning" above)
poker-loop --workers 8 --iterations 100 --no-global-round   # opt out fleet-wide

poker-play --list-bots                       # show the best trained models found
poker-play --players 6 --stack 200 --sb 1 --bb 2 --hands 10 --human-seats 1
poker-play --players 9 --hands 500 --human-seats 0 --seed 42 --bots model:checkpoints/models/<a-published-model>.pt
```

Re-run `pip install -e ".[dev]"` after pulling changes that add a new
`[project.scripts]` entry point (like `poker-gui` was) — an editable
install doesn't pick up a newly added console script until reinstalled.

Hand histories are written to `hand_histories/session_<id>.jsonl` (gitignored).
Trained models go to `checkpoints/` (also gitignored): `agent.pt` is the live
one, overwritten each save; `checkpoints/models/` is the shared store of every
published model (write-once), which later training runs draw their opponents from
and which the ratings in `checkpoints/global/` refer to. The one-time move from
the old per-machine layout (`machines/<host>/pool/`, `exchange/`, `retired/`)
has been done and verified — 9,777 models in one store, no ghost members, no
stale `ref` — so `rl/migrate_store.py` and its test have been deleted. Nothing
reads that layout any longer; a `registry.json` from before the per-model files
is still migrated, but by `global_store.migrate_legacy_registry`, which is a
different thing and stays.

## TODO — decided to postpone, not forgotten

Each item below was discussed and consciously left for later ("per ora teniamo
così"). None is a bug; the current behaviour is coherent.

- **Observation v2: encode the action *sequence*, not per-street aggregates.**
  Today `_history_aggregates` compresses the whole betting history into 40
  numbers — 4 streets x 10 counters — and what it throws away is exactly what
  poker is played on:
  - **The order inside a street.** "check, bet, raise" and "bet, raise, check"
    produce the identical vector: one bet, one raise, one check. Initiative, who
    opened, who reacted to whom — all gone.
  - **Who did what.** Only the *last* aggressor of each street is identified
    (by relative seat) plus the agent's own participation. Two different
    opponents raising the flop reads as "2 raises" with only the second one
    named.
  - **Individual bet sizes.** Only the street's *maximum*, relative to the pot.
  - Per-opponent *amounts* do survive, in the 81 seat slots
    (`current_bet / pot` and `committed / pot`, the latter cumulative over the
    hand). It is the action *types* and their order that are lost.

  Why it matters beyond tidiness: bet-sizing patterns are the signature of the
  thing the agent most obviously cannot do. Small on the flop then large on the
  turn is value building; large on the flop then giving up on the turn is an
  abandoned bluff. To the current encoder those are nearly the same vector. And
  a bluff *is* a sequence — an agent that cannot read the story an opponent is
  telling is unlikely to learn to tell one. Measured on the best-rated model
  (rating 1687, 800 hands): fold 49.8% of all decisions and **70.6% when legal**,
  check/call 31.9%, everything aggressive 18.3%, with 8-9 actions typically
  legal. In one fold spot the logits were fold **21.40** against ~11 for
  everything else — 100.0% vs 0.0%, entropy 0.002. The information is all there
  already: `Observation.action_history` is a list of `ActionRecord` carrying
  `street`, `seat`, `player_id`, `action_type`, `amount`, `stack_before/after`,
  `pot_before`. **No engine change is needed** — only `features.py`.

  **The prerequisite, and it is the whole reason this has not been done.**
  `check_compatible` (`rl/ppo.py`) rejects a checkpoint whose `obs_dim` *or*
  `feature_version` differs, so any encoder change today makes all **9,117
  stored models, the 60 frozen anchors and the entire Elo scale built on them
  unloadable at once**. That is not a migration, it is a population reset — and
  it is why `FEATURE_VERSION` has been 1 since day one: the encoder is currently
  unamendable. So the work is in two parts, in this order:
  - **Part 1, a versioned encoder.** `encode_observation` takes a `version`; the
    current code becomes the `version=1` branch, untouched. `RLAgentPlayer` reads
    the version off the checkpoint it wraps and encodes at that version.
    `check_compatible` accepts any version this build can serve rather than only
    the newest. Then a v1 model and a v2 model sit at the same table, each fed
    its own vector: the pool stays full, the anchors stay valid, the ratings keep
    meaning something, and a v2 model earns its rating against the existing
    population instead of starting in an empty world. Small, no behaviour change,
    directly testable (seat a v1 and a v2 at one table and assert each gets its
    own vector). Worth doing on its own merits whatever happens to Part 2.
  - **Part 2, the v2 features.** Keep the 40 aggregates (a cheap summary the MLP
    already knows how to use) and append a flat positional encoding of the **last
    20 actions**, 24 features each: action type one-hot (7), actor's seat
    relative to me (9), street one-hot (4), `amount / pot_before` (1),
    `log(amount / big_blind)` (1), was-it-me (1), slot-occupied/padding (1).
    480 new features, so `OBS_DIM` 480 → **960** and the first `Linear` grows
    from 480x512 to 960x512 — ~250k extra parameters, negligible.
    **Twenty is measured, not guessed**: over 600 hands of the best model, actions
    per hand (blinds included) run median 12, mean 11.9, max 25; a window of 16
    covers a whole hand 91.2% of the time and 20 covers **99.3%**. When it does
    truncate it keeps the most recent actions, which are the informative ones.
  - **Fix how each new feature is scaled, in the same change as the version.**
    Every feature in `features.py` is a bounded ratio (`_clip01`, with
    chip amounts divided by the big blind and passed through `log1p` over a fixed
    scale), and the network has no input normalisation of its own beyond the
    `LayerNorm` after each `Linear`. The 24 per-action features must follow the
    same convention: `amount / pot_before` and `log(amount / big_blind)` clipped
    to [0, 1] with a stated scale, the one-hots left as they are. A feature left
    unbounded (a raw `amount / pot_before` can be many times the pot on an
    overbet) would dominate the first `Linear` and change what `FEATURE_VERSION`
    means without anyone noticing, so the scale is part of the version's
    definition, not a detail.
  - **Why not a GRU over `ActionRecord`s**, which is the textbook answer and is
    mentioned elsewhere in this file as the natural v2: it also changes
    `PokerActorCritic`'s architecture rather than just the features, multiplying
    the work and the risk, for information the flat encoding already exposes — an
    MLP can learn "two actions ago the player on my left bet". A recurrent
    encoder only pays off once hands routinely exceed the window, and 99.3% of
    them do not. Revisit if the table size or the stack depth ever grows enough
    to change that.

  **Gate this behind verifying the ranking first.** `pick_parents` draws from the
  100 best-rated models, and that order was measured as wrong near the top
  (#20 vs #100 inverted, t = −6.1). If it is still wrong, a better encoder
  produces better models that a near-random selection then throws away. The
  1000-hand rated sessions were introduced to fix exactly that and **nothing has
  verified it yet**; the check costs a few hours of CPU and no code change
  (duplicate-deck duels between a few ranked pairs, the technique whose null
  control read +2.0 bb/100 at t = 0.43 over 48,000 hands).
- **Train on 2 to 9 players, not only 6.** Everything runs at 6 seats, 200-chip
  stack, blinds 1/2 (`--players/--stack/--sb/--bb` in `poker-train`, `poker-loop`,
  `global_arena.py`, `run.sh`). The *observation is already ready*:
  it carries the live/actionable/all-in opponent counts and the table size
  (normalised by `MAX_SEATS = 9`) plus 9 relative seat slots, and
  `test_rl_env_flow.py` already fuzzes 2-9 players. What is missing is the
  infrastructure, which assumes one `GameConfig`:
  - `SelfPlayCollector` builds one `Table` with a fixed `num_players`; it would
    need to pick the table size per hand (one cached `Table` per size, each with
    its own rotating button) and draw that many opponents from the pool.
  - `evaluate_against_pool`, the global arena
    (`play_global_round`) and `benchmark.py` all seat `game.num_players`. The
    benchmark must stay exactly reproducible, so its table size has to be a
    function of the block index, like its seats and opponent draws.
  - Open decision 1 — **how to sample the size.** A uniform draw over 2-9 per
    hand gives big tables many more decisions per hand, so heads-up would be
    under-represented in the data; balance by decisions, or weight the sizes that
    matter most.
  - Open decision 2 — **what to do with the ratings.** Every existing rating was
    measured at 6-handed, and Elo at another size means something different. Either
    keep the current scale and mix sizes into it, or keep a separate rating per
    size. No `FEATURE_VERSION` bump is needed: the meaning of the features does
    not change.
  - Open decision 3 — **the scale of the reward and of the value target.**
    `reward_scale` is a fixed constant (`big_blind / starting_stack`) and the
    critic's `MSE` target is not normalised adaptively, so it is calibrated for one
    table. The spread of a hand's result changes with the table size (more
    opponents, more chips in play per hand), so a constant tuned at 6-max would
    leave heads-up and 9-max with value targets on different scales and the
    `value_loss` out of balance with the policy loss — the failure `reward_scale`
    was introduced to prevent (9,300-14,000 unscaled against ~1.1). Decide whether
    to scale per table size or to normalise the return adaptively (running
    mean/std, or PopArt); the second also covers "Vary the starting stack". Either
    one means saving its statistics in the checkpoint, or `--resume` and
    inheritance start on a different scale from the one the model was trained at.
- **Save Adam's state in the archived models — measured, and dropped.** Only the
  live checkpoint (`agent.pt`, what `--resume` reads) carries the optimizer; the
  archives that become pool models hold weights alone, so an inheriting worker
  restarts Adam cold. This sat here as a promising improvement until the transient
  it was meant to remove was actually looked for, across 153 runs of 1000
  iterations. **It is one iteration long and small.** Inheriting runs show
  `kl` 0.0140 at iteration 1 against 0.0067 at iteration 2, then flat at ~0.007;
  fresh runs sit *higher* throughout (0.0375 at iteration 1, still 0.0125 at 50),
  and the same holds for `clip` (0.094 vs 0.338 at iteration 1) and entropy (an
  inheritor starts at 0.47 and stays, a fresh run at 1.64 and drifts down). A cold
  optimizer damaging a warm model would look like the opposite: oversized,
  unstable updates early. Even that single elevated iteration stays inside the
  healthy 0.005-0.01 band. Not worth roughly tripling the size of every archived
  model, multiplied by ~10,000 models on the volume. **Do not revive this without
  new evidence**; the transient it targets is ~0.1% of a run.
  - Measured at the same time, and relevant to `--inherit-fraction`: the benefit
    of inheriting **halves** in the long-run regime. At 100 iterations inheriting
    workers finished +6.2 bb/100 against their pool and fresh ones −33.0 (a
    40-point gap); at 1000 iterations it is −1.0 vs −16.0, a 15-point gap. Ten
    times the iterations lets a randomly initialised network recover most of the
    distance, which weakens the case for raising the inherited share.
- **Weighting the Elo update by the chip margin — measured, and deliberately
  not done.** Today `pairwise_elo_delta` uses only the *sign* of each pair's
  chip difference, so beating an opponent by one chip counts exactly as much as
  beating them by 1,247 bb. Two fixes were proposed and both were simulated
  against a synthetic population whose true skill is *known* (impossible to do
  on real data), scoring each rule by how well the final ranking orders true
  skill: (a) a continuous score, `0.5 + 0.5 * clip(margin / S, -1, 1)`, and
  (b) multiplying the existing delta by `|chip delta| / starting_stack`.
  - Under **heavy-tailed noise — the realistic case**, one session in ten being
    dominated by a monster pot — (a) gains nothing (0.9207 vs 0.9203 for the
    sign rule) and a loose scale actively hurts (0.9018 at S = 4 stacks).
  - (b) as literally proposed is **worse than today** (0.80 vs 0.89) on two
    counts: a chip delta can be negative, so the multiplier flips the sign and a
    losing model *gains* rating; and it is unbounded, so a lucky session
    multiplies the update by 5 or 10. With `abs()` and a cap it beats today —
    but a control run shows the gain is **entirely** the lower effective K:
    capping at 0.5 scores 0.9426, and simply multiplying K by the same mean
    weight (0.420) scores 0.9439. Identical at every cap tested.
  - The reason is that `|chip delta|` measures how *eventful* a session was, not
    how *well* the model played — a model that lost three stacks is weighted
    exactly like one that won three. The direction is already in the sign, so an
    unsigned magnitude adds variance, not signal.
  - What the simulations really showed is that **K dominates**: with the sign
    rule unchanged, scaling the whole `DEFAULT_K_SCHEDULE` down moves the
    ordering from 0.8894 (24/16/12/8/6, the schedule at the time) to 0.9351
    (12/8/6/4/3) to 0.9604
    (6/4/3/2/2) — some thirty times the effect of either scoring change. That is
    a five-number edit with no new formula, no cap to tune and no migration
    problem, since the scale itself does not change. **The schedule has since
    been taken well past the bottom of that grid**, to 4.0/1.0/0.5/0.1 and then
    to 3.0/1.0/0.3/0.1, which are today's four *settled* tiers under a prepended
    burn-in of 24/8 for a model's first 30 rated games; the measurement below is
    still the one worth making before moving any of them again.
  - **Left as it is at the user's decision.** Before touching the schedule the
    thing to measure is the real ratio between skill spread and per-session
    noise across the population (the simulation assumed 30 bb/100 and 160 bb
    from rough real orders of magnitude): that ratio, not the simulation, is
    what sets the right K. Note also the trade the schedule exists for — a lower
    K means a new model takes longer to climb from 1500 to where it belongs, and
    ~1,000 new models arrive every generation.
- **Make the Elo update take the bb difference into account.** Today
  `pairwise_elo_delta` reads only the *sign* of each pair's chip delta, so
  winning by one chip counts as much as winning by 100 bb. Decided to do, not yet
  done. It supersedes "Weighting the Elo update by the chip margin" above, which
  recorded why two earlier variants were *not* adopted — read it first, since the
  design has to avoid what it found:
  - A continuous score `0.5 + 0.5 * clip(margin / S, -1, 1)` gained nothing under
    heavy-tailed session noise (0.9207 vs 0.9203) and a loose `S` hurt it.
  - Multiplying the delta by `|chip delta| / starting_stack` was a lower effective
    K in disguise (identical to scaling K by the mean weight), and flips sign
    unless `abs()` is taken and capped.
  - `|chip delta|` measures how eventful a session was, not how well the model
    played; the new rule has to be judged by whether it orders a population of
    *known* skill better than the sign rule, not by looking sensible.
  - Whatever is chosen touches the scale everything rests on (the Elo scale is
    defined by how often a 1,000-hand session picks the stronger model, see
    `SESSION_HANDS`), so the K staircase and the anchors' ratings would need
    re-deriving, and readings taken before it are not comparable with later ones.
- **Study how many bb/100 a model should earn for a given Elo gap.** Nobody has
  measured the exchange rate between rating points and chip winnings, and the bb
  update above needs it. The data points already in this file do not line up,
  which is the reason to do it properly:

  | Elo gap | measured edge (duplicate decks) |
  |---|---|
  | 313 | +490 bb/100 |
  | 136 | +18.6 bb/100 |
  | 96 | +26 bb/100 |
  | 49 | +37 bb/100 |
  | 54.9 | −4.4 bb/100 (sign wrong) |

  Different pairs, different dates and different session lengths, so the table
  is not a curve. The study should play pairs of models spanning a range of
  Elo gaps with `rl/duel_power.py` (duplicate decks, which is what resolves a few
  bb/100), enough pairs per gap to fit a relation, and report the edge with its
  interval per gap. Things it has to settle: whether the relation is roughly
  linear or saturates; how much it depends on the field (`--mode field` measured
  ~20 bb of per-hand noise against ~11 for the 3v3 duel); and whether it holds at
  the top, where the ranking was measured wrong. The result is what a
  bb-weighted Elo update would be calibrated against.
  **The tool exists: `rl/elo_bb_grid.py`** (`python -m pokerlab.rl.elo_bb_grid
  --hands 10000 --jobs 20`, results in `checkpoints/studies/elo_bb_grid.json`,
  resumable). Every benchmark model against every other and against itself, 3v3
  at a 6-seat table, duplicate decks, one cell per unordered pair (the grid is
  antisymmetric); the diagonal is the noise floor; it prints a through-the-origin
  fit of bb/100 on the Elo gap. **A cell is the row model's own bb/100 per seat,
  which is half the head-to-head margin `duel_power` reports.** Cost at the
  40 anchors there are now: 820 matches, ~16M hands at 10,000 — a few hours on
  20 processes, so `--max-models` exists for a first look.
- **Handle all-in differently, or remove it as a choice.** The 11th action bin
  (`ALL_IN_BIN`) lets a model shove its whole stack at will, in any spot. The rule
  to move to: **an all-in happens only when the amount to bet is larger than the
  remaining stack** — a bet, raise or call that the stack cannot cover becomes an
  all-in by itself, and there is no free choice to shove. Measured (October 2026)
  (6 models per band, 3,000 hands at a 6-max table, models of one band against
  each other): all-ins per 100 hands per model were **1.6** at ~1500, **0.5** at
  ~1600, **0.4** at ~1700 and **4.5** for the top models (range 1.0-8.5), i.e.
  15.5% of the top models' aggressive actions against 1.4-4.4% below them, almost
  always postflop. The choice is rare in the lower bands and not rare where the
  ratings are highest, so it is part of what the top models do.
  - **The catch, and why it is not a one-line change:** `action_dim` is part of
    what `check_compatible` (`rl/ppo.py`) compares, so deleting the bin changes the
    shape of the policy head and makes **every stored model and anchor unloadable
    at once** — the same population reset described under "Observation v2". The
    non-breaking route is to keep all 11 bins and **mask the all-in bin unless the
    sizing rule above applies** (the mask is already how out-of-range bins are
    handled: masked, never clamped), which needs no new feature version and no
    migration; only a real removal needs the versioned-encoder groundwork first.
  - Decide the rule for a **call** that costs the whole stack too: it is
    `CHECK_CALL_BIN` today, and `ALL_IN` only when the action maps there.
  - Whatever is chosen changes how every rating was earned (the models were
    selected with the shove available), so readings from before are not
    comparable with later ones.
- **Vary the starting stack.** Every hand begins with all seats at 200 chips
  (100 bb), reset after each hand (`rebuy=True`, `SelfPlayCollector`), so the
  model never sees a genuinely short or deep starting stack — only the depth a hand
  itself reaches. Randomising the stack per hand (in big blinds) would teach
  those regimes; `reward_scale` (`big_blind / starting_stack`) would then have to
  follow the stack in use, and it shares the rating question above.
  - **The reward scale is the part that breaks first.** The features already adapt
    (pot, stack and bets are divided by the big blind), but the critic's target is
    `MSE` on returns scaled by one constant and is not normalised adaptively, so a
    stack that changes per hand changes the spread of the target hand by hand and
    a single `reward_scale` stops being right. Following the stack in use is the
    minimum; a running normalisation of the returns (running mean/std, or PopArt)
    is the robust option and also serves "Train on 2 to 9 players". Its statistics
    would have to be saved in the checkpoint, or a resumed or inherited model
    starts on a different scale from its own training.
- **Rolling workers: drop the generation barrier entirely.** A generation today
  is a *barrier*: N workers are launched together and the supervisor does not
  start the next N until the last one is done. That was harmless while every
  worker ran the identical settings and therefore finished at about the same
  time. It is no longer: `--hands` is swept per worker over a 2.5x range, so a
  worker drawing 320 finishes in well under half the time of one drawing 800,
  and the barrier turns that difference into waiting.
  - **The Elo fill-in phase is the second-best answer, deliberately chosen
    first.** It spends the wait on rating rounds instead of idling, which is
    genuinely useful work and — the reason it came first — needed no change to
    the loop's structure at all: one extra phase in the worker and a two-file
    handshake in the supervisor. The proper answer is to stop making fast
    workers wait: as each worker exits, launch its replacement immediately, so
    the machine holds N runs at all times and nothing is ever idle.
  - **What the barrier is currently load-bearing for**, and all of it would need
    a home:
    - **Parent selection happens once per generation.** `inheritance_plan` reads
      the ranking once and deals `pick_parents` out to N workers, taking care
      that they are *distinct*, which is what makes the population competing
      lineages rather than N copies of one model. Rolling launches would pick a
      parent one at a time, and "distinct from whom" stops having an obvious
      answer -- distinct from the other runs currently in flight is the likely
      replacement, and it needs the in-flight set to be tracked.
    - **So does the hyperparameter plan**, for the same reason: it is drawn once
      per generation from the parents chosen in the same breath. Rolling, it
      would be drawn one worker at a time, which for a pure inherit-and-perturb
      plan is a smaller change than it was while an arm split had to be held at a
      fixed ratio across a batch.
    - **`--status`, the dashboard and `worker_progress` all key on the
      generation number**, reading `gen<N>-w<K>.log` for one N. With rolling
      workers there is no single current N, so the watcher would have to track a
      set of live runs instead -- the biggest single piece of the work, and all
      of it in `rl/monitor.py`.
    - **The generation record and the benchmark** (`run_generation_benchmark`,
      `GenerationRecord`, the `benchmark_new_*` series) are per-generation by
      construction: "the best model this generation produced" is the only
      cross-generation signal the loop has. It would become "the best model of
      the last N runs", which is a rolling window and needs a window size
      decided.
    - **The post-prune `benchmark_arena` run** is launched *between* two
      generations precisely because the workers have exited and the machine is
      free -- it takes every core. Rolling, there is no such moment, so it would
      need one manufactured: stop launching replacements, drain, run, resume.
  - **Start here if it is picked up**: the fill-in phase already proves the
    supervisor can hold a conversation with a worker that has not exited
    (`wait_for_workers`, `FILL_STOP`/`draining`), which was the piece the loop
    most obviously lacked. Rolling launches are the same polling loop with a
    launch inside it.
  - Not a bug, and not urgent while the fill-in is absorbing the difference. The
    thing to measure before starting is how much time the fill-in is *actually*
    absorbing per generation, which the worker logs now report directly
    (`riempimento elo: <n> giri, <n> sessioni`): if it is a few minutes a
    worker, the barrier costs little and this stays here.
- **Unify the main parameters in one config file.** Today every parameter that
  matters exists in *three* places: a `DEFAULT_*` constant in some module, an
  argparse flag in each CLI that touches it, and — for anything the loop passes
  down — a string in the command line `loop.py` builds for `poker-train`. The
  constants themselves are **52 `DEFAULT_*` definitions spread over twelve
  modules** — `duel_power.py` (9), `global_arena.py` (8), `pool_registry.py` (7),
  `benchmark_arena.py` (6), `train.py` (5), `benchmark.py` (5),
  `dashboard.py` (3), and two each in `training_pool.py`, `loop.py`,
  `killswitch.py` and `global_store.py` — and the count keeps growing: the Elo
  fill-in phase added three (`DEFAULT_FILL_MIN_SESSIONS`,
  `DEFAULT_FILL_DEADLINE_MINUTES`, `DEFAULT_FILL_GAMES_PER_MODEL`), each of them
  spelled out again as an argparse default in *both* CLIs and again in the
  command line the loop builds, which is precisely the triplication this entry
  is about. The same name can also mean two different things:
  `DEFAULT_GAMES_PER_MODEL` is 50 in `global_arena` and 20 in `benchmark_arena`.
  The costs are not hypothetical, all three were paid in this project:
  - **A constant that described nothing that ran.** `DEFAULT_GAMES_PER_MODEL` was
    500 while both CLIs *and* `run_population_round` carried a hard-coded 12.
    That is why a test now pins all three together by reading the argparse
    defaults out of the source — a workaround for the duplication, not a fix.
  - **Documentation drift.** This file described the K schedule as 12/8/6/4/3
    while the code held 4.0/1.0/0.5/0.1, and the benchmark layout as 14 series of
    10 models while the disk held 7 of 5. With one file holding the values, the
    prose can point at it instead of restating it.
  - **A fleet outage.** Because `loop.py` serialises its parameters into a
    command line that `train.py` must parse, removing one flag from `train.py`
    killed every worker of every running supervisor (see "Never delete a
    `poker-train` CLI flag while a supervisor is running"). A config both read at
    launch removes that coupling entirely.

  **The load-bearing design constraint: do not move the reasoning.** Those
  constants carry long comments recording *why* each value is what it is — the
  measurements behind `DEFAULT_HANDS_PER_GAME = 1000`, `DEFAULT_K_SCHEDULE`,
  `reward_scale`, `opponent_probability`. That reasoning is worth more than the number,
  and a config file full of bare values would strand it. So the constants stay
  exactly where they are, comments included, as the **defaults**; the config file
  only *overrides* them, and an absent key means "use the documented default".
  - **Format: stdlib only**, the project's one firm rule about dependencies (the
    GUI is Tkinter for this reason). `tomllib` is in the standard library from
    3.11 and this project requires 3.13, so a `pokerlab.toml` needs nothing new;
    JSON would do as well and is writable too.
  - **Open decision 1 — shared or per machine.** The project directory is one NFS
    export mounted by every VM, so a single shared config means one edit retunes
    the whole fleet: powerful, and dangerous in the same breath. Anything genuinely
    local must stay local — `--machine`, the worker count (`default_workers`
    measures the box), `--state-dir`/`--work-dir`/`--log-dir` — and `run.sh`
    already derives all of those.
  - **Open decision 2 — read once, or re-read every generation.** Re-reading per
    generation is what would let the fleet be retuned without restarting
    supervisors, which is exactly the pain the outage exposed. It also means a
    half-saved file can reach a live supervisor, so it needs the same
    write-then-rename care as every other shared write here.
  - **Keep the CLI able to override the file**, or the parallel-sweep recipe and
    every one-off experiment stop working.
  - **Each run must record the values it actually resolved**, the way a published
    checkpoint already records `hidden`/`num_layers`/`feature_version`. Today a
    run's parameters are recoverable from its command line in the supervisor log;
    with a config file that is no longer true, and a rating earned under unknown
    settings is worth much less.

- **Bigger networks — measured, blocked on one thing, deferred to a new version
  of the project.** `hidden`/`num_layers` have been 512/3 since day one and have
  never been anything else: `train.py` builds `PokerActorCritic()` with no
  arguments, and a 250-model sample of the store shows exactly two file sizes
  (3,129,575 and 3,129,923 bytes, differing by metadata, not shape) with all 25
  checkpoints opened in full reading `(512, 3, 480, 1)`. So the axis is entirely
  unused across all ~9,600 models.
  - **Almost everything already supports mixed shapes.**
    `build_model_from_checkpoint` rebuilds at the shape the checkpoint records,
    and *every* consumer goes through it — the training pool
    (`registry_opponents`), the bb/100 benchmark, the population round,
    `benchmark_arena`, the GUI, `poker-play`. Verified directly: a 768x4 model
    seats at a table and plays. Nothing batches across models, so different
    sizes coexist at one table.
  - **Exactly one thing breaks: inheritance.** `load_checkpoint` is called in a
    single place in the whole project, the `--resume` path
    (`train.py`), and it loads into an *already built* model, so
    `load_state_dict` raises a bare `RuntimeError` on a shape mismatch —
    verified in both directions. It is not an edge case: `pick_parents` draws
    parents by *rating*, not by shape, so a differently-sized worker would
    almost always draw a mismatched parent and die at startup. Compounding it,
    `check_compatible` does not look at the shape at all (only `obs_dim`,
    `action_dim`, `feature_version`), so the failure arrives as a raw torch
    error rather than the project's own `IncompatibleCheckpointError`.
  - **Two possible fixes, and they are not equivalent.** Building the learner at
    the *parent's* shape on `--resume` is the smaller change, but then `--hidden`
    becomes a suggestion that inheritance silently overrides. Filtering
    `pick_parents` by shape keeps each lineage at its own size, which is also
    what makes the experiment readable — a 768 lineage with a 512 ancestor in it
    measures nothing. Either way `check_compatible` should learn to say so
    clearly.
  - **The costs, measured** (forward pass at batch 1, `OMP_NUM_THREADS=1`; the
    hands/s compose that with the two thirds of a hand that is pure Python and
    does not change — 11.9 actions per hand, 12.4 ms of which ~32% torch):

    | shape | parameters | file | us/forward | est. hands/s |
    |---|---|---|---|---|
    | 256x3 | 259k | 1.0 MB | 370 | 77.9 |
    | **512x3 (today)** | **781k** | **3.1 MB** | **473** | **71.1** |
    | 512x4 | 1.04M | 4.2 MB | 609 | 63.8 |
    | 768x3 | 1.56M | 6.3 MB | 801 | 55.7 |
    | 768x4 | 2.16M | 8.6 MB | 1,491 | 38.2 |
    | 1024x3 | 2.61M | 10.4 MB | 2,018 | 30.8 |

    Speed falls faster than linearly. **768x3 is the sweet spot** — 2.2x the
    parameters for 22% fewer hands/s — while 768x4 halves throughput and
    1024x3 does worse, which doubles the cost of every rated session, every
    benchmark and every population round.
  - **Two operational consequences to size before trying it.** The population
    round loads 55 models per worker, measured at 203 MB today; at 768x4 that is
    ~470 MB, and with 25 workers arriving together ~12 GB of transient
    allocation instead of ~5, on 31 GB boxes already holding ~13 GB of training
    — the same profile as the unexplained fleet incident. And the store is
    ~30 GB of the 107 GB free at 3.1 MB a model, so bigger models grow the
    backlog several times faster while **the pruning trigger counts files, not
    bytes** — at a larger shape that threshold stops being the right measure.
  - Deferred at the user's decision: no compatibility shims and no code written
    only to paper over a mismatch, so this belongs to a new version of the
    project rather than to the running fleet.
- **Study how the network evaluates the state, to size it (decided to do, not
  yet done).** The network size (`hidden`/`num_layers`, 512/3 today, see "Bigger
  networks") has never been chosen from evidence about what the network can
  actually *see* in a state. The study should find out at what capacity, and at
  what point in training, the networks start to recognise the structures that
  decide a hand: straights, flushes, full houses and the other made hands and
  draws, and how strong the hand is relative to the board.
  - **What to measure.** Probe the value head and the policy (e.g. linear probes
    on the trunk's activations, or the value/action response to controlled
    states) for each concept: pair/two pair/trips, straight and straight draw
    (the wheel included), flush and flush draw, full house, quads, hand rank
    percentile on the board. For each one, the capacity (a ladder of
    `hidden`/`num_layers`) at which it becomes decodable, and the training
    iteration at which it appears within a run.
  - **Why it matters.** The card input is 6 binary 4x13 planes, so a straight is
    a pattern across ranks and a flush a pattern across suits; the first `Linear`
    has to build those from raw planes. If a concept only becomes readable above
    some size, a 512x3 network may be capped below it, and a bigger one would be
    worth its cost (see the throughput table in "Bigger networks"). If every
    concept is already read at 256, the current size is wasteful.
  - **Feeds three other items:** "Bigger networks" (which shape to try),
    "Observation v2" (whether explicit hand-strength features would help or the
    network already derives them) and the suit-isomorphic canonicalisation idea
    in "Where to extend".
- **Aggression/style constraints on the models, to keep the population's
  strategies diverse (idea for the next version, not yet designed).** Today
  every model is trained toward the same objective (bb won against the drawn
  field) and the fleet's selection pressure is a single number, the Elo, so the
  population tends to collapse onto one style — the measured fold rate of the
  best model (70.6% of the decisions where folding is legal) is already a sign of
  it. The idea: impose constraints on playing-style metrics during training, so
  that different lineages are trained to different profiles (VPIP, PFR, AF
  aggression factor, 3-bet %, fold-to-bet %, WTSD, c-bet %, ...) and the pool
  keeps genuinely different opponents instead of near-copies.
  - **Why it would help.** Opponent diversity is what the training pool exists
    for, and `HP_LADDERS` diversifies only the *settings*, not the resulting
    *behaviour*. A model forced to play loose-aggressive and one forced to play
    tight-passive are different tests for a learner, and a top-ranked model that
    beats only one style is a weaker anchor than its rating says.
  - **Things to decide before building it.** (1) *How to constrain*: a penalty or
    Lagrangian term in the PPO loss on the measured metric vs. a target band, a
    reward bonus, or masking/biasing action bins — the first is the cleanest, the
    last repeats the "masked, never clamped" argument in reverse. (2) *Where the
    target lives*: as a per-run hyperparameter in the sweep (a new axis of
    `HP_LADDERS`, inherited and perturbed like the others) or as a fixed list of
    named profiles. (3) *How a style interacts with the Elo*: a constrained model
    will rate below an unconstrained one, so parent selection (`pick_parents`, top
    100 by rating) would discard every constrained lineage — selection would have
    to be per-style, or rate relative to the style's own niche (quality-diversity
    in the MAP-Elites sense). (4) *Measuring the metrics*: they can be computed
    from the hand history / `ActionRecord`s with no engine change, but VPIP and
    AF need a definition fixed once and tested, and a metric measured over a
    rollout is noisy, so the penalty needs a window like `REWARD_WINDOW_HANDS`.
  - **The metrics must carry their sample size, over a window of at most ~200
    hands.** Wherever VPIP, AF and the like are measured — as the quantity being
    constrained, or later as per-opponent stats the agent sees in its observation
    (a HUD) — two rules apply. (1) *Report the number of hands behind each
    figure*, not the figure alone: a VPIP of 60% over 8 hands and over 200 hands
    are different facts, and a model fed the bare percentage would trust both
    equally. In an observation it means a hands-played count (normalised) next to
    each stat, or the stat shrunk toward a prior in proportion to the count. (2)
    *Keep only the most recent ~200 hands per player* (a sliding window, not a
    cumulative mean), because in a real room players change often — a new person
    sits in the seat, or the same one changes style — and a long history describes
    someone who is no longer there. The same window is the right size for the
    constraint's own measurement: the rollout's metric is noisy over few hands, so
    the penalty should read the window and know how full it is, as
    `REWARD_WINDOW_HANDS` and its partial-window marker already do for
    `train/100`. Note the training setup resets stacks and swaps the opponents
    seat by seat (`SeatProxy`), so a "player" in training has no persistent
    identity across hands; per-opponent stats would need that identity defined
    first.
  - **Likely needs a population reset or the versioned encoder**: a style target
    given to the network as an input feature would change `OBS_DIM` (see
    "Observation v2"); constraining only the loss changes no shape and keeps
    every stored model loadable.

## Where to extend each future section

- **RL**: implemented end to end (see the "RL" section above). Natural next
  steps, roughly in order of value: multi-process rollout collection (N
  `Table`s in N workers — note `make_policy_fn` returns a closure, which is not
  picklable, so this needs `fork` or a module-level callable); duplicate/mirror
  deals (replay a seeded deck with rotated seats and average) for variance
  reduction; suit-isomorphic canonicalisation of the card planes for a ~4x
  sample-efficiency win; and a per-action history encoder replacing today's
  per-street aggregates — specified in full as "Observation v2" in the TODO
  section, where a flat encoding of the last 20 actions is argued for over a GRU
  and the versioned-encoder prerequisite is spelled out.
  **Trained models are seatable by path**: `model:<path>` is the only bot spec
  `build_players`/`validate_bot_key` understand (see "Heuristic bots,
  removed"), so a bad path fails only for whoever asked for it.
  `discover_trained_models()` (pure Python — `pool_registry` carries no torch)
  ranks the checkpoints found across *every* machine's pool, since the best
  model is usually not on the host you are sitting at; torch is imported only
  inside `make_model_bot`, so `poker-play` and the GUI still run without the
  `rl` extra until someone actually picks a model. `build_players` grew an
  optional `game: GameConfig` for this: `RLAgentPlayer` normalises its features
  by `big_blind`/`starting_stack`, which an `Observation` deliberately does not
  carry, so seating a model without it raises rather than guessing.
- **Vision**: see the section "Vision (`vision/`)" below for what exists. Next
  is recognition (`CardRecognizer`, taking a crop and returning cards), then the
  dealer button and the opponents' actions, each filling the same spot screen. The
  integration points are unchanged: an `Observation`-compatible read or a
  `HandHistory`-style record, no engine changes.
- **GUI**: implemented — see the "GUI" section above, including a visual
  bot builder, opponent card-graphics, a spy toggle, step-through bot
  actions, a scrollable raise slider with pot-fraction shortcuts, a tinted
  dealer seat, a paced all-in runout, an end-of-hand reveal of every
  participant's cards, and a full per-action/showdown log naming the
  winning combination. Remaining rough edges are
  documented inline: there's no mid-hand stop.

## Testing strategy already in place

Ordered by correctness risk (highest first — see `tests/unit/`):
`test_evaluator.py` (hand ranking), `test_side_pots.py` (side-pot math and
uncalled-bet refund), `test_betting_legal_actions.py` (legal actions,
min-raise/short-all-in), `test_blinds_and_stacks.py` (button rotation, short
blinds, busted players), `test_hand_history.py` (JSONL round-trip).
`tests/integration/test_full_hand_flow.py` fuzzes 2-9 players across 5 seeds
each with a chip-conservation invariant on every single hand — this is the
test most likely to catch a subtle new bug, run it after any engine change.
These engine-level tests fill their non-learner seats with the minimal
`make_random_legal_bot`/`make_always_call_bot` in `tests/support.py` (see
"Heuristic bots, removed") rather than any product-facing bot.

For the RL section the equivalent gate is
`tests/integration/test_rl_env_flow.py` (same 2-9 × 5-seed fuzz, asserting the
policy's action bins never produce an `IllegalActionError`); run it after any
change to `rl/action_space.py` or `rl/features.py`. `test_rl_policy.py` is the
only test file that needs the `rl` extra and skips itself cleanly without it.
`test_rl_global_arena.py` covers the cross-machine population Elo/pruning
system (see "Population-wide Elo and pruning" above); its discovery/
sampling/lock/promotion/member-store tests need no torch, only the
`play_global_round`/`run_population_round` tests do.
