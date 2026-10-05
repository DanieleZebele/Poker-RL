# pokerlab — notes for future Claude Code sessions

## What this project is

A No-Limit Texas Hold'em engine built as the foundation for a future
reinforcement-learning poker project. Five sections:

1. **Game engine** (`src/pokerlab/engine/`, `cards/`, `evaluator/`) — **implemented and tested.**
   Configurable players (2-9), stacks, blinds; full betting logic including
   side pots and the min-raise/short-all-in edge case; JSONL hand history.
2. **Players** (`src/pokerlab/players/`) — **implemented and tested.**
   `ManualPlayer` (terminal input). Every non-human seat, in the CLI and the
   GUI alike, is a trained model given by `model:<path>`
   (`cli/play.py::build_players`, `discover_trained_models`). With no
   `--bots` given they cycle through the best-rated models found across every
   machine's pool; with no trained model on disk, seating a non-human seat
   raises a clear error. `--list-bots` prints the discovered models.
3. **RL training** (`src/pokerlab/rl/`, `players/rl_agent.py`) —
   **implemented and tested.** Observation encoding, discrete action space +
   legal-action mask, `RLAgentPlayer`, trajectory collection with GAE, a
   PyTorch actor-critic, self-play PPO with an opponent pool, checkpointing,
   evaluation in bb/100, a Gym-shaped `TablePokerEnv`, and the `poker-train`
   CLI. Requires the `rl` extra. See "RL" below.
4. **Live table vision** (`src/pokerlab/vision/`) — **implemented.** Screen
   capture (`mss`), a mouse-drag region selector, zones saved to
   `vision_data/regions.json`, a screen of its own ("Collect vision data" on the
   main menu) that collects labelled crops, and recognisers for cards, the
   dealer button, seat states, bets, the pot and stacks that fill the spot
   screen. Requires the `vision` extra (opencv-python/numpy/mss). See "Vision"
   below.
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
- **Big blind option**: `compute_legal_actions` branches on
  `current_bet_to_match == 0` (offer BET, a fresh open) vs. `> 0` (offer RAISE
  off the BB's own posted bet). Gating on `to_call == 0` would be wrong: preflop,
  when everyone just calls around to the BB, `to_call == 0` but
  `current_bet_to_match > 0`, and the BB must still be able to raise. Regression
  test: `test_big_blind_option_can_raise_when_everyone_just_called`. If a "why
  can't seat X raise here" report comes up, check this exact branch first.
- **Engine code never imports from `players/`** (one lazy, function-local
  import in `table.py` is the deliberate exception, to avoid a real import
  cycle) — keeps `engine/` reusable by RL/vision/GUI without dragging in
  CLI or terminal-input concerns.

## Test-only opponents

Engine-level tests (`test_full_hand_flow.py` and others) fuzz correctness (chip
conservation, action legality) against cheap, deterministic opponents:
`make_random_legal_bot`/`make_always_call_bot` in `tests/support.py`. They are
test-only and decoupled from anything product-facing.

## RL (`rl/`, `players/rl_agent.py`)

Target algorithm is **self-play PPO** (actor-critic). Requires the `rl` extra
(`pip install -e ".[rl]"`).

- **The network's shape is four numbers, set from `config.toml`'s `[network]`**
  (`hidden`, `num_layers`, `head_hidden`, `head_layers`; flags of the same names in
  `poker-train` and `poker-loop`, which forwards them to every worker, resolved when
  the generation starts). The trunk is `num_layers` blocks of `Linear -> LayerNorm ->
  ReLU`, `hidden` wide; **each head is `head_layers` such blocks, `head_hidden` wide,
  then its output `Linear`** (11 logits; 1 value), with separate weights for the two
  heads, so the critic's gradient reaches the trunk and its own layers and not the
  policy's. `head_layers = 0` is the original network (1,241,612 parameters at 1380
  inputs); the shipped file asks for 1 layer of 256, which is 1,502,220 parameters and
  a batch-1 forward ~41% slower (446 -> 630 us; `head_layers = 2` is 769 us).
  Checkpoints record all four, `build_model_from_checkpoint` rebuilds at the recorded
  shape (so models of different shapes coexist at a table), and a checkpoint without
  the head keys is refused by `check_compatible` with a clear message. **A worker whose
  parent has another shape starts from scratch** (`train.parent_shape_mismatch`, the
  message "parent has a different network ... starting from scratch", `args.resume`
  turned off, so no sweep observation is written either): `pick_parents` draws by
  rating, not shape, so after the shape in the file changes this is an ordinary event,
  not a crash. **Change the shape with a fleet reset when you can**: until the old
  shape's models are pruned, every worker that draws one as a parent throws its weights
  away. The sweep does not touch the shape.
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
- **`OBS_DIM` is 1380** (`FEATURE_VERSION` 2; it was 480 before each seat got 100 optional
  statistics slots, see "Opponent statistics" below), defined as the sum of named per-section constants so it
  cannot silently drift. Cards are **6 binary 4x13 planes** (hole, flop, turn,
  river, whole board, hole∪board): a street not yet dealt is an all-zero plane,
  which solves the 0/3/4/5-board-cards problem for free and keeps the door open
  to a `Conv2d` over `(6, 4, 13)` with no change to `encode_observation`.
  Seats are indexed **relative to me** (slot 0 is always me), so the encoding is
  invariant to absolute seat numbering. Action history is aggregated per street,
  **not sequenced**, which loses the order, who did each action, and the
  individual bet sizes — the planned replacement (for version 3) is a flat
  positional encoding of the last 20 actions behind a versioned encoder, not a
  GRU; see "Observation v3" in the TODO section.
- **`encode_observation` takes `big_blind` and `starting_stack` as keyword
  arguments** because `Observation` deliberately carries no table config. Do
  *not* try to recover the blind from the `POST_BLIND` records: a short blind
  posts less than the big blind. `legal_mask` is a *required* argument (echoing
  it into the input helps the value head) — a default would silently produce
  different features at training and inference time.
- **Field size is encoded twice, on purpose**: as aggregate counts (live,
  still-actionable, and all-in opponents, plus the static table size) and as the
  per-seat slots (9 base features each; plus 100 optional statistics slots), which say *which* seats folded and where they sit relative
  to me. Both are recomputed from the live `Observation` at every decision, so a
  hand naturally tightens and loosens as players fold.
- **Positional value does not follow the seating order from the button**, and
  getting this wrong is easy: postflop action opens on the small blind and
  closes on the button, so the seat *before* the button is second-best while the
  small blind is worst. Ranking by distance clockwise from the button is
  inverted for every seat except the button itself (it scores the cutoff 0.00
  and the big blind 0.60), so the feature ranks by how many seats act after me;
  regression test:
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
  **Episode = one hand**, and `SelfPlayCollector` redraws every seat's stack for
  every hand (see "Tables: sizes and stacks") because `Observation` has no memory
  of earlier hands, so a multi-hand episode would not be Markov w.r.t. the
  features.
- **Gotcha, encoded as a test**: mask logits with a large *finite* negative
  (`MASK_FILL = -1e9`), never `-inf`. With `-inf`, `Categorical.entropy()`
  computes `0 * -inf = NaN` whenever only one action is legal — which happens
  routinely in poker.
- **A chopped pot legitimately leaves every stack unchanged**, so "a played hand
  must move chips" is *not* a valid invariant (it cost one wrong test
  assertion). The real invariants are chip conservation and per-hand rewards
  summing to zero.
- **The reward scale is not cosmetic** (`SelfPlayCollector.reward_scale`, wired
  from `TrainConfig.reward_scale`, defaulting to `big_blind / starting_stack`,
  where `starting_stack` is the *deepest* stack of the mixture: 1/100).
  With an unscaled big-blind reward and stacks up to 100 bb, value targets span
  ±100, the squared value loss sits in the thousands and — against a policy
  loss of order 0.01 — the critic's gradient is all the network ever sees
  (`value_loss` climbs unscaled and stays near ~1 scaled).
  `HandTrajectory.reward` stays in big blinds for reporting;
  `DecisionRecord.reward` carries the scaled training signal. **One constant for
  every table size and every stack depth, on purpose**: the reward stays in big
  blinds, the same unit as the bb/100 every result is measured in, and a
  per-hand scale (dividing by the seat's own stack, say) would re-weight deep and
  short hands in the gradient away from that objective. Whether the spread of the
  target is balanced across sizes is *measured, not assumed* — see the TODO
  entry on the value scale. If you ever change the stack range or the blinds and
  training stalls, check this first.
- **PPO reuses the mask stored at rollout time** (`DecisionRecord.legal_mask`),
  it does not recompute it. Recomputing or omitting it would measure
  `pi_new/pi_old` against a different distribution than the one that acted, and
  the gradient would be silently wrong.
- **The run's own past selves are not in the training field.** A frozen snapshot
  of the learner is a copy of the network being trained, so it drifts with it
  and anchors nothing, which is precisely the job the fixed pool exists to do;
  every seat it took would be a seat *not* facing an independently trained
  model. Every opponent seat faces a previously trained model from the store.
  A parent whose metadata carries axes `perturb_hyperparameters` does not know
  is fine: it reads the axes of `HP_AXES`, so an unknown key costs nothing.
- **`OpponentPool` fills the seats the learner is not in**, drawing **uniformly**
  from the previously trained pool models it was given (`extra_opponents`, see
  `train.py::registry_opponents`). With none available (an empty store, the very
  first run ever), `OpponentPool.sample` returns `None` and every seat goes to
  the learner — plain self-play is the graceful starting point, not a special
  case. If a curriculum over opponent strength is ever wanted, it belongs in
  `SelfPlayTrainer`, by choosing which pool models to draw from per phase.
  Seats are swapped between hands through `SeatProxy` rather than by rebuilding
  the `Table`, which would reset the button and stacks. Exactly one seat is always the learner, so a hand can
  never yield zero training data.
- **Two kinds of saved model, and the distinction matters.**
  `checkpoints/agent.pt` (`--checkpoint`) is the *live* one: weights plus
  optimizer state, **overwritten every save**, and what `--resume` reads.
  `checkpoints/models/<machine>-[<prefix>-]agent-<runid>.pt` is a *published*
  model in the one shared store: **written once, never modified, and only one per
  run** — the run's **final** checkpoint. When training ends `main()` writes the
  run's **scratch** file (`--scratch-dir/agent-<runid>.pt`, plus a `.json` sidecar
  holding the rating and iteration), rewrites it once with the benchmark outcomes,
  and then the scratch copy is **published**
  (`global_store.publish_model`: a dotted `.partial` copy renamed into place, so a
  reader on another machine never sees half a checkpoint) and registered in the
  global ranking at the rating the learner measured. Publishing once at the end
  is what makes a label mean one set of weights forever — overwriting a model that
  already has a rating history would blend two different networks into one Elo.
  There is no mid-run archiving and no cadence flag.
  If a run dies before it can publish, the loop's sweep publishes the scratch
  file from the sidecar (see "Continuous training loop"). The name carries the
  machine (`--machine`) and, from `poker-loop`, the generation and worker
  (`--archive-prefix gen0155-w07`), so two hosts can never collide. **The run id
  is only second-resolution**, so genuinely concurrent runs of `poker-train` on
  one machine need distinct `--archive-prefix` *and* `--scratch-dir` (the parallel
  -sweep recipe below does this), or the later one finds its name taken and its
  model is not published. Both `checkpoints/` subtrees are gitignored.
- **The published checkpoint is the run's *last*, not its best-rated.** This is
  deliberate and counter-intuitive. The in-run `learner_rating` comes from a few
  thousand hands against the run's own drawn pool, a noisy measurement, while
  training improving the model is a reliable prior: selecting on the rating
  throws away part of the improvement the later checkpoint carries. The general
  principle: **a reliable prior beats an unreliable measurement of the right
  quantity.** Swap it only for a measurement good enough to also catch the runs
  where the later model genuinely is worse — the benchmark bb/100 is the
  candidate, being deterministic and scored against models never seated in
  training. Pinned by
  `test_a_run_publishes_its_last_checkpoint_not_its_best_rated`.
- **Elo is applied pairwise** (`pairwise_elo_delta`): a session's participants
  are scored against each other from their chip deltas, win/loss/draw, and the
  per-pair changes are divided by the number of opponents faced — otherwise
  table size would silently rescale K and a 6-handed session would move ratings
  five times as far as a heads-up one. Every update reads the ratings as they
  stood *before* the session, so the result does not depend on iteration order.
  A draw is not a degenerate case here: a chopped pot genuinely leaves two
  stacks equal. **The ranking criterion is deliberately not bb/100**: a single
  evaluation has a spread of hundreds of bb/100 on one fixed model, so ranking
  on it would rank the luckiest; ratings instead *accumulate* one result
  per rated session until they reflect strength rather than one draw from a
  heavy-tailed distribution.
- **Every training run draws its own opponents** from the shared store
  (`rl/training_pool.py::draw_training_pool`), so workers do not all train against
  the same field — runs are diverse by construction, not by accident. The draw
  mixes two sources for `--pool-models` (20) seats: **top** (`--pool-top-share`,
  50%, uniformly from the `--pool-top-n` = 100 best-rated models) and **random**
  (every remaining seat, uniformly from the whole store, which keeps weak and odd
  opponents — never-rated newcomers included — in the mix and stops the field
  being only the current elite); it reaches a newcomer in proportion to how many
  there are rather than by reserving seats for them. The random source is also
  what makes a short top source harmless: with nothing rated yet the top share has
  nobody to draw and the pool still comes out full; with fewer models than seats it
  is short and `fill_slots` cycles it. Seeded by
  `--seed`, so a given seed reproduces its draw. The ranking it reads is the
  `registry.json` snapshot (one file; see "Continuous training loop"),
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
  model enters the global ranking with. At a flat K a session moves a rating by
  only a few points and the learner would sit near the baseline whatever its
  bb/100, so it is rated through `DEFAULT_K_SCHEDULE` on its own session count,
  which is what makes the number mean something inside a run — see "K follows a
  staircase".
  - **It starts at the parent's rating, not at 1500** (`train.inherited_rating`,
    read off the resumed checkpoint's `publish_rating` metadata, falling back to
    `pool_rating` and then to the baseline). A child is its parent plus some
    training, not an unknown quantity, so starting at 1500 would spend the run's
    first rated sessions re-discovering where its own lineage already sits. The
    *session count* still starts at zero, and the two together are exactly what
    the schedule's gain expects: a starting point worth something, and no claim to
    having earned it.
  The live learner is never itself a member: it changes every
  iteration, so persisting it would rate a moving target, and that is exactly why
  its K has to be handed to the registry rather than looked up. With an *empty* store
  (the very first run) there is nobody to evaluate against: the run is plain
  self-play against the current policy, skips evaluation, and still saves and
  publishes.
- **`evaluate_against_pool()` produces the learner's rating.** It plays
  `--eval-sessions` (**10**) rated sessions of `--session-hands` (**1000**) hands
  each, re-drawing the opponents from the pool and re-seating the learner before
  every one. At `--eval-every 100` over a 1000-iteration run that is **100 rated
  sessions**, which then continue into the pass against the frozen anchors on one
  count. Seats are swapped through `SeatProxy` (made public for this) so one
  `Table` lives across the whole pass and the button keeps rotating; rebuilding
  it per session would reset the button to the same seat every time and hand
  whoever sits there a systematic positional edge.
  - **A pass is a session count, full stop** (`--eval-sessions`), not a hand
    count that has to be a multiple of a block size: how many rated results a
    pass produces is what a reader wants to know.
  - **There is one session length, `--session-hands` (`session_hands` in
    `config.toml`, 1,000 by default), and every program that plays a rated session
    reads it from the same flag**: validation, the pass against the anchors, the
    population and fill-in passes, both arenas, and the shards `play_sharded`
    launches (which get it as an explicit flag). The flag is registered by
    `table_mix.add_table_arguments`, the constant is `table_mix.DEFAULT_SESSION_HANDS`,
    and `test_one_parameter_sets_the_session_length_everywhere` pins that every
    parser carries the same default. The Elo scale is *defined* by how often a
    session of that length picks the stronger model, so changing it changes the
    scale: do it with a fleet reset.
  - **What the validation rating is not.** It is measured against *live* models
    whose own ratings carry error, and it rates a moving target: the weights change
    between passes, so a session played at iteration 100 scored a network that no
    longer exists. That is why the validation is deliberately the *smaller* half
    of a run's rated sessions — 100 against 500 — and why the published rating is
    earned against the pinned anchors; the more sessions come from the validation
    phase, the more of the final rating is a measurement of a moving target.
  - **Session length is the only lever that moves the Elo equilibrium.** Elo sees
    only the *sign* of a chip delta: at a few dozen hands the stronger side
    finishes ahead barely more often than not, so the rating equilibrates close
    to the pool instead of where the strength deserves, and a lower K shrinks the
    jitter around that equilibrium and cannot touch it. Exactly the argument on
    `table_mix.DEFAULT_SESSION_HANDS`.
  - **`--eval-every` (100)**: ten passes a run, ~10% of a run's time. What the
    interval buys is *rated sessions in the count* (100 is what the schedule's
    gain was sized against). **`run.sh` must not override it**: an explicit flag
    beats a default, so a pinned value would quietly keep the fleet on a
    different cadence; see the note on `--hands` under "Every worker inherits weights".
  - **The last iteration always evaluates**, whatever the interval. Without it
    a run whose iteration count is not a multiple of
    `--eval-every` ends with no final reading at all — and that reading is also
    the rating the model is published with whenever the benchmark pass cannot
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
  reordered what slot 340 means while keeping the vector exactly as long, so a
  dimension check alone would have happily loaded a model that reads the wrong
  thing from every slot. Checkpoints record it and `check_compatible` rejects a
  mismatch. Checkpoints also record `hidden`/`num_layers`, so
  `build_model_from_checkpoint` can rebuild an archived agent at whatever shape
  it was trained with.
- Rough health indicators for a run (re-check them when the encoder or the
  reward scale changes): `value` around 1 and flat, `kl` well under 0.05 per
  iteration, a moderate `clip` fraction, and `entropy` drifting slowly down
  (ln 11 ≈ 2.4 is the uniform-over-all-bins ceiling). A climbing value loss or a
  `kl` above ~0.05 per iteration means the step size is too large.
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

## Tables: sizes and stacks (`rl/table_mix.py`)

The models train and are rated on a *mixture* of tables, not on one `GameConfig`.
`TableMix` is the whole description (pure Python, no torch): `table_weights` (one
relative weight per size 2-9, default 25/20/15/10/10/10/5/5, mean 4.35 seats),
`stack_min_bb`/`stack_max_bb` (1.0 and 100.0) and the blinds. Everything is set in
`config.toml` `[game]` or by the flags `add_table_arguments` registers
(`--table-weights`, `--stack-min-bb`, `--stack-max-bb`, `--sb`, `--bb`), which
every CLI that plays hands shares and `poker-loop` forwards to workers and shards
(`table_arguments`). `--players` and `--stack` no longer exist.

- **Every seat draws its own stack, independently, uniform in big blinds with
  decimals, and the stacks are redrawn after every hand** (`TableBank.play_hand`).
  Stacks never carry over between hands, in training or in any rated result: hands
  stay i.i.d., which `duel_power`, the duplicate-deck measurements and the
  derivation of the K staircase all rest on, and a table cannot bleed down to
  heads-up halfway through a session. Independent draws mean a big table almost
  always contains someone short (the minimum of nine uniform draws is ~10 bb);
  that is accepted, and it is what the models will meet at a real table.
- **Chips stay `int`.** Blinds are 50/100, so a chip is a hundredth of a big blind
  and 45.6 BB is 4560 chips; the float exists only in `TableMix.draw_stacks`, at
  the moment a stack is drawn. `TableMix.starting_stack` (the *deepest* stack in
  chips, 10,000) is what the features are normalised by and what
  `reward_scale` divides by; it replaces the old fixed starting stack wherever a
  normalisation constant was wanted, which is why `game.big_blind` and
  `game.starting_stack` kept working on a `TableMix`. No `FEATURE_VERSION` bump.
  `env.py`, `duel_power` and `poker-play` still play one explicit `GameConfig`.
- **Size per hand in training, per session in every rated result.** The collector
  draws a size for every hand (`TableBank` keeps one `Table` per size, each with its
  own button, so positions stay uniform). Validation, the benchmark and the
  population passes draw it once per *session* and hold it for all 1,000 hands: a
  session compares the same participants over the same hands, which is what
  `pairwise_elo_delta` assumes. A size that changed hand by hand would have a
  heads-up hand involve two of a session's nine models and leave every pair with
  a different number of shared hands. Over a pass the share of *hands* per size is
  still the weights, since sessions have equal length.
- **A pass needs enough models for the largest table that can be drawn**
  (`mix.max_players`; a size with weight 0 does not count). A set too small for it
  returns nothing (`None`/`[]`) rather than quietly playing only the small tables,
  which would measure a different mixture than the weights say.
- **Stacks come from a separate rng than the shuffle** (`TableBank(stack_rng=)`):
  `Table` touches its own `random.Random` only to shuffle, so a fixed seed deals
  the same cards whatever the stacks are. Pinned by
  `test_stacks_do_not_touch_the_card_sequence`.
- **One rating, per-size diagnostics.** There is one Elo scale over the whole
  mixture, because one network is trained on the mixture and the question is how
  strong it is on that, and eight scales would need eight times the games.
  `rate_against_benchmark` also returns the learner's bb/100 per size
  (`BenchmarkRating.bb_per_100_by_size`), printed as `per tavolo:` and recorded in
  the model as `benchmark_bb100_by_size`: a diagnostic for a lopsided model, never
  an input to the ranking.
- **A fleet reset comes with it.** Every rating on disk was earned at one
  6-handed table with identical stacks and means something else on this scale, so
  the ledger and the frozen anchors start again (the project restarts from zero
  at v2; there is no code to read old ratings).

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
  repeat. **Nothing is seeded, merged, re-ranked or trimmed by the loop**: each
  worker draws its own opponents from the shared store, trains, publishes its
  final model back, and plays a cross-population rating pass, so the ratings
  live entirely in the global registry.
- **Where things live.** Models in `checkpoints/models/` (shared, write-once);
  ratings in `checkpoints/global/`; and per machine only `--state-dir`
  (`loop_state.json` and the `STOP` file), `--work-dir` (each worker's scratch
  directory `work/gen<N>-w<K>/` and live checkpoint `work/agent-gen<N>-w<K>.pt`)
  and `--log-dir`. `run.sh` points all of it at `checkpoints/machines/<host>/`.
- **Published names are unique by construction**:
  `<machine>-gen<N>-w<K>-agent-<runid>`. `poker-train` derives the run id from a
  start-time timestamp with second resolution, so workers launched together in a
  generation share it; the machine/generation/worker prefix is what keeps them
  apart, including across hosts that happen to be on the same generation number.
- **Monitoring is `poker-loop --status`**, or `--status --watch 30` to redraw in
  place every 30 seconds. It reads `loop_state.json` from `--state-dir`,
  written after *every phase* (starting, training, publishing, benchmark_arena) so a
  check mid-generation still reports something true. It shows the per-worker
  table (below) and nothing else: **deliberately no leaderboard, no
  per-generation history**, because `--watch` answers
  what this machine is doing *now* and those pushed the live worker table off
  the screen. The history stays in `loop_state.json`, and the ranking is
  printed into the supervisor log as each generation ends (`format_leaderboard`).
  - **Per-worker progress is parsed from the workers' own logs**
    (`worker_progress` → one `WorkerProgress` per worker), not reported by the
    supervisor. The supervisor is blocked waiting for the entire training phase
    and learns nothing until a worker exits, so without reading the logs a
    twenty-minute generation would show only "phase: training". The parser has
    to tolerate a half-written final line, since the logs are read while the
    workers are still appending to them; `parse_worker_log` is split out from
    the file handling precisely so every case can be tested on a string.
  - **What each worker's row carries**: iterations done out of the target, the
    stage it is in, the training win rate, the last pool evaluation and rating,
    the **mean rating of the field it drew**, the bb/100 against the frozen
    anchors (`ancore/100`) and the rating it published with (`elo ancore`),
    entropy, and how long ago its log was last written.
    - **The `pool` column is what makes `eval` and `rating` readable at all.**
      Those two say how the learner did; this says *against whom*. Every worker
      draws its own opponents (`draw_training_pool`), so two rows of the same
      table are not comparable without it. `train.py` prints
      `pool rating: media <n> min <n> max <n>` right after the draw and
      `monitor.py` parses it; "-" for a run against an empty store.
    - **`train/100` is a mean over the last `REWARD_WINDOW_HANDS` (100,000)
      hands**, in **bb per 100 hands**. One iteration swings far too much to read
      on its own, and a cumulative mean would bury the present under the first
      iterations of the run. A fixed hand budget (not a fixed number of
      iterations) puts every worker's noise on the same footing even though
      `--hands` is swept. It measures the learner against the whole training
      field, the copies of itself in the seats `opponent_probability` left
      unfilled included, which is *not* the same opposition as `eval` (the drawn
      pool) or `ancore/100` (the frozen set) — the columns are not comparable
      with each other. Treat it as "is this run alive and not collapsed", not as
      a measurement: the measurement is `ancore/100`.
      - **Measure its noise from first differences, not from the raw spread.** A
        run improves over its life, and a plain standard deviation over its
        iterations charges that trend as noise.
      - **The window is partial early in a run, and that is shown rather than
        blanked**, because "is it alive" is exactly what the column is read for
        minute to minute. `WorkerProgress.train_hands` says how many hands are
        behind it; the dashboard puts that in the cell's tooltip and marks a
        filling window with a `*`.
      - **The window is sized from the header line** every worker prints as it
        starts (`device cpu | tables 2:25% 3:20% ... | 512 hands/iteration`). With no header at
        all the parser averages every iteration recorded and reports
        `train_hands` 0, claiming no precision it cannot back.
      - **The unit is bb/100, converted at the parse boundary**
        (`monitor.HANDS_PER_RATE`), not in `train.py`: the `iter` line on disk
        carries bb per hand. The field is `WorkerProgress.train_bb100` /
        `WorkerHistory.train_bb100`, named for its unit. Pinned by
        `test_the_training_rate_is_converted_to_bb_per_100_hands`.
    - **An age column, and a warning above the table.** A killed worker leaves
      its log behind, so without the age its bar would sit at 40/100 forever and
      read as a slow worker rather than a dead one. Past `STALE_LOG_SECONDS` (10
      minutes) in a stage that should still be writing, the row is flagged with
      `!` and named in an "ATTENZIONE" line. Generous on purpose: a population
      pass plays thousands of hands between two lines.
  - **The stage comes from `phase: <name>` lines the worker prints itself**
    (`rl/phases.py`, `train.py::announce`), because several stages of a run
    print *nothing* for minutes and the last ordinary line would otherwise make
    a worker deep in the Elo pass look like it were still training.
    `rl/phases.py` is the whole shared vocabulary, so the name a worker prints
    and the name the watcher parses cannot drift apart; the population pass
    announces its own three stages through the `on_phase` callback threaded down
    from `run_population_sessions`. The watcher adds three states of its own:
    `avvio` (no iteration yet), `training`, and `ERRORE` — set by a traceback in
    the log and deliberately **sticky**, since nothing printed after a crash
    undoes it. At the target iteration count `stage_label` renames the last
    stretch ("fine training"), because "training" at 100/100 really means the
    wrap-up — the saving, the publishing and the pass against the anchors. The
    run ends with a `done` marker.
  - **The two long stages also report how far into themselves they are**
    (`phases.py::progress_marker`/`parse_progress`, `train.py::PhaseProgress`).
    A marker says *what* a worker is doing; these say *how much is left*. The
    benchmark pass and the population pass print very few lines between them
    — long enough that the 10-minute stale-log warning would fire on a perfectly
    healthy worker. The line is
    `avanzamento <stage>: <done>/<total> (<pct>%), <detail>, ~<n>m rimasti`,
    written and parsed in `phases.py` for the same reason the stage names are.
    - **Both passes count sessions.** A pass is a fixed number of sessions
      (`--global-sessions`, `--fill-sessions`), so the denominator is exact up front
      and the bar ends at exactly 100%. (It used to count *owed games*, because a
      per-model debt made the session count unknowable; see "A pass is N
      sessions" below.)
    - **The pass reports every session; `train.py` decides the cadence**
      (`PROGRESS_EVERY_SECONDS`, 30), always printing the first line and the
      last, so `--status` is never more than half a minute stale and the watcher,
      which re-reads the whole log on every poll, stays cheap.
    - **Progress is cleared when a new stage is announced**, so a percentage
      shown next to a stage always belongs to that stage. `stage_label` answers
      "what is it doing" (the dashboard and the phase counts use it);
      `stage_cell` adds "and how far", which only a table cell wants.
    - **`--status` also prints a `fine :` line and the dashboard a `fine ~38m`
      pill**, both the *longest* ETA rather than the mean: a generation ends when
      its slowest worker does, and that is the number a supervisor waiting to
      stop the machine is actually asking for.
    - Only the in-process path reports (`workers <= 1`), which is the one
      production uses. A sharded pass (`benchmark_arena`, run by hand) spreads
      its playing over subprocesses with their own scratch logs.
  - The only benchmark is the per-worker pass below; `--benchmark-dir` names the
    frozen set it uses.
- **Every published model is rated against the frozen anchors before the
  population pass** (`benchmark.py::rate_against_benchmark`,
  `--benchmark-sessions` **500** sessions of 1,000 hands — **500,000 hands**,
  forwarded by `poker-loop`). A model is otherwise published with the rating its
  own training run measured and only earns real games when a population pass
  happens to draw it — a small chance per pass out of a large store — so it
  could sit in the ranking on an unearned number for many generations. This
  plays it against the frozen anchors right after training instead, so it enters
  the global ranking with 600 rated sessions behind it.
  - **The pass draws its opponents at random from the whole frozen set** and
    measures one number, `benchmark_bb100` (over 500,000 hands), which is also
    the hyperparameter sweep's response variable.
  - **Why 500 sessions.** A 1000-hand session measures a rating with a standard
    deviation of about 238 points, so precision improves only as
    `1/sqrt(sessions)`: halving the confidence band again would cost four times
    the sessions. 500 is a compromise between published-rating precision and the
    wall time a worker spends after it stops training.
  - **The K comes from the schedule, continuing the learner's own count.** A flat
    K is an exponential average with a fixed effective window, so more sessions
    buy nothing past a point; the falling K is what turns 500 sessions into 500
    sessions' worth of evidence.
  - **The sessions are deliberately NOT queued for the global merge.** They are
    already in the rating the model publishes with, and `_apply_session` would
    re-apply the identical evidence a second time from the member's own games
    count — worse than applying it once, because re-using the same outcomes
    cannot add information and does add movement. Instead `publish_model` takes
    both the rating **and** the session count (`games_after`), which is what makes
    the local computation authoritative rather than a report. Publishing at
    `games=0` would have a later population pass shove a 600-session model
    around at the schedule's top tier on evidence it already has.
  - **The anchors are the right opponents**: their ratings are pinned, so the
    result is measured against a fixed scale rather than a moving field, and
    they are never seated in training, so it is not memorisation. Opponents near
    the model's own rating are the more informative ones (a session's information
    goes as `p(1-p)`), but the draw is left **uniform**: biasing it would save
    sessions at the price of every model playing a different field, and
    `benchmark_bb100` has to stay comparable across models to serve as the
    sweep's response variable.
  - **Anchors are loaded in rotating slices, not all at once**
    (`--benchmark-resident` 10 anchors, re-drawn every `--benchmark-rotate-every` 50 sessions; both in `config.toml` `[benchmark]`). The
    frozen set only grows, and every worker of a generation reaches this phase at
    about the same moment, so holding the whole set resident in each worker would
    multiply into a large transient allocation across a machine. A slice is small
    and the pass still faces the whole set. A checkpoint that fails to load is
    **struck off** the candidate list rather than merely skipped, so a bad file is
    reported once instead of once per slice.
  - **A frozen set too small to seat a table returns `None`, and that is not an
    error**: a fresh installation has no anchors yet, and the run still publishes,
    at the rating its validation measured.
  - **`play_sharded`** launches `python -m pokerlab.rl.global_arena` shards;
    `global_arena` therefore needs its `if __name__ == "__main__"` block (pinned
    by a test that runs the module as a subprocess). Nothing in production passes
    `workers > 1`. The games split **exactly** between shards (`shard_games`),
    never `ceil(games / workers)` per shard, which would silently overplay.
- **Stopping, the graceful way**: Ctrl-C (or SIGTERM) finishes the generation in
  flight and then exits, rather than killing workers mid-run and losing their
  archives. Creating a file named `STOP` in `--state-dir`
  (`checkpoints/machines/<host>/STOP`, which is what `./run.sh stop` creates)
  does the same, which is how an unattended loop is stopped from another shell.
  Note that "finishes the generation" can mean twenty minutes of waiting.
- **Stopping, the immediate way**: `poker-kill` (`./run.sh kill`,
  `rl/killswitch.py`) kills this machine's supervisor, its workers, their
  children and any global-elo shard at once, whatever stage each is in.
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
    it, publishing any unpublished archive from its sidecar first, and locks
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
  would never be removed, taking trained models no ranking could see with them.
  The sweep first **publishes each abandoned worker's own `agent-*.pt`** into the
  shared store (named as the worker would have named it, registered at the rating
  its sidecar recorded; skipped if that name is already published), then removes
  the directory. The after-training pass is also how a worker that *crashed*
  before publishing still gets its model in. It additionally removes this
  machine's own stale `.<machine>-*.partial` in the models directory (a publish
  that died mid-copy) and stale `global-arena-shard-*` scratch in the temp dir,
  both only once untouched for an hour. **Safety rules that must survive any
  change**: (1) anything a running process still names on its command line is
  left alone — a worker can outlive a supervisor that died, and deleting its
  directory corrupts it; liveness comes from `/proc`, which only sees this host,
  which is correct because `work/` is this machine's own and never shared; where
  `/proc` is missing nothing is deleted; matching is by substring, so anything
  that merely *mentions* a worker's name (a `tail -f` on its log, say) keeps its
  directory around — the safe direction; (2) a directory whose archive could not
  be published is kept; (3) only the exact name shapes the loop itself creates
  are touched. **Never run the sweep on one machine against another machine's
  `work/` directory from a shell** — this host's `/proc` cannot see that
  machine's workers, so a live one would look abandoned.
- **Every worker inherits weights**, always (`inheritance_plan`). Without
  inheritance the loop only ever produces models exactly `--iterations` deep —
  it generates variety, not strength — and a
  from-scratch worker trains worse and is far likelier to blow past the `kl`
  "step too large" threshold early. The diversity that a fresh start would have
  supplied comes instead from the hyperparameter sweep below: the lineages
  differ by their *settings*, which costs nothing, rather than by throwing away a
  trained network. Inheritors take **distinct parents drawn at random from the
  `--pool-top-n` best-rated models on disk** (`training_pool.pick_parents`:
  rated, non-frozen, cycling only if there are fewer than needed), a different
  set every generation, so the population is competing lineages rather than
  copies. A model that has never been rated is not a parent.
  - `--resume` reads `--checkpoint`, so the loop copies the chosen parent from the
    store into the worker's checkpoint path first (the store's file is only read).
    **Only weights carry over**: published models store no optimizer state, so
    Adam restarts cold. That costs a short transient (about one iteration) and is
    far cheaper than discarding the weights too.
  - **`run.sh` must not pass `--hands`** (nor any other
    swept axis). An explicit flag beats a default, so a pinned value silently
    overrides what `loop.py` decides. **The general rule: when a default moves in
    `loop.py`, check whether `run.sh` is overriding it**, because a default only
    governs the paths that do not. Pinned by
    `test_run_sh_does_not_override_the_axes_the_sweep_decides`, which reads the
    launch command out of `run.sh` and requires every swept axis to be absent.
  - **A worker with no parent starts from the fleet's own flags, and so from
    `config.toml`** (`loop.starting_hyperparameters`, then perturbed like any other
    worker: see below). That is how the first generation after an empty store is
    set: write `lr`, `hands`, ... in the `[starting-point]` section and every cold
    worker begins a step or two around them. There is no separate cold-start
    learning rate any more (`--fresh-lr` is gone): a random network does have
    further to travel, so if cold workers should train faster than the 3e-4
    default, set `lr` in the file. The same keys are what a hand-run `poker-train`
    uses as they are. `poker-loop` does **not** ignore them any more; pinned by
    `test_the_file_sets_where_a_worker_with_no_parent_starts`.
  - `--status` marks inheriting workers with `^` and fresh ones with `.`. Every
    row should read `^`; a `.` means the store
    had no parent to give, which is worth noticing rather than hiding.

- **Every worker of a generation trains with its own hyperparameters**
  (`loop.hyperparameter_plan`, recorded in the model it publishes by
  `train.run_metadata`), so a generation is a sweep and an outcome can be
  attributed afterwards to the settings that produced it.
  - **The axes** (`HP_AXES`): `--lr`, `--hands`, `--opponent-probability`,
    `--ppo-epochs`, `--clip-epsilon`, `--minibatch-size`, `--gae-lambda`,
    `--value-coef`, `--max-grad-norm`, `--entropy-coef` and the two that decide
    how strong a field the worker draws, `--pool-top-share` and `--pool-top-n`.
    **`--pool-models`, `--global-sample` and `--global-sessions` are held fixed**:
    the first is the field's *size*, which is also what a run's cost scales with,
    and the last two are how the whole fleet's ratings are earned, so varying them
    would make runs incomparable rather than comparable along one axis. There are
    no ladders of allowed values any more: nothing draws a rung.
  - **Every worker multiplies a starting point by one of `HP_MULTIPLIERS`** —
    **x{0.8, 1.0, 1.2}, unbounded**, one independent draw per axis. That set is
    only the default: `--hp-multipliers` / `hp_multipliers = [...]` in `config.toml`
    (`[sweep]`) replaces it, a repeated value is likelier, `[1.0]` alone switches
    the search off, and a non-positive entry is a `ConfigError` (checked in
    `load_args`, so a bad reload keeps the previous values). The starting
    point is the weights parent's recorded settings, or, for a worker with no
    parent, the fleet's own flags (see above). Nothing is snapped onto a grid, so
    a lineage holds whatever its ancestors' draws multiplied out to and can walk
    arbitrarily far.
    - **Why unbounded.** The ends of any range were never measured, and a clamp
      would make a guess load-bearing: whoever wrote it would decide in advance the
      furthest the fleet could ever go. The real bound is selection — a parent is
      drawn from the `--pool-top-n` best-rated models, so a lineage that walks its
      lr up to something that breaks training rates badly and stops being a
      parent. An arithmetic clamp guesses where the edge is; the ranking finds out.
    - **The cost, accepted: a lineage really can drift far.** The step is a
      random walk in log space, so the spread after N generations is about
      `ln(1.2)*sqrt(2N/3)` — roughly 27x either way over 500 generations. Nothing
      stops that but the ranking.
    - **The multipliers are not symmetric in log space.** 1.2 is not 1/0.8, so
      the set's geometric mean is `(0.96)^(1/3) = 0.9865`: every axis of a
      lineage shrinks ~1.35% per generation with nothing opposing it but
      selection (a median lineage ends at `0.9865^60 = 0.44` of its start after 60
      generations). Accepted as it is.
    - **Only two limits survive, and neither is a range.** A probability axis is
      capped at 1.0 (`_HP_PROBABILITY_AXES`: `opponent_probability`,
      `pool_top_share`), because a probability above 1 is not one and both reach
      code that would quietly do something odd with it. And a **count axis moves
      by at least 1** whenever the multiplier is not 1.0, floored at 1
      (`_moved_count`).
    - **Why count axes need that rule.** Plain rounding looks sufficient and is
      not: `round(2 x 1.2) = round(2.4) = 2` and `round(2 x 0.8) = 2`, so **2 is
      absorbing in both directions** — and 2 is an ordinary `ppo_epochs`. So the
      rule is: round as usual, and only where that would leave the value where it
      started, step exactly one in the multiplier's direction. *Always* rounding
      away from the current value would fix the ratchet too and was rejected: it
      inflates the step wherever rounding was working, turning 6 into 8 or 4
      instead of the 7 or 5 that x1.2 and x0.8 actually ask for.
    - **The floor is 1, not 0.** `ppo_epochs` 0 is not a smaller setting, it is
      the absence of training, and 1 is the integer analogue of the fact that
      multiplying a positive value by 0.8 never reaches 0. 1 is reflecting rather
      than absorbing — the up draw from 1 gives 2. On `hands` and `pool_top_n` the
      floor is unreachable in practice, ~24 consecutive downward draws away.
    - An integer multiplied drifts off any grid an analysis could group by, so
      points are a continuum. That costs nothing that was not already lost,
      because an inherited point never could be read as a response curve (it
      correlates with its parent's quality by construction).
    - `HP_AXES` says *which* axes exist, so adding a name there is all it takes
      for a worker to move on it (it also needs a `Hyperparameters` field, a
      `--<axis>` flag and an entry in `train.REPORTED_AXES`). Pinned by
      `test_an_inheriting_worker_multiplies_every_axis_by_one_of_the_multipliers`,
      `test_a_lineage_is_never_snapped_back_to_any_grid`,
      `test_a_probability_axis_is_capped_at_one` and
      `test_a_count_axis_is_rounded_so_it_can_never_reach_zero`.
  - **Every worker perturbs something and none is drawn independently**
    (`hyperparameter_plan`), so a generation is a pure search. The only workers
    that do not inherit their parent's settings are the ones that *cannot* — see
    the fallback below.
    - **What that costs.** Sampling alone is a controlled experiment that never
      compounds; inheriting alone is a search that cannot be *read*: after a
      couple of generations the surviving settings are whatever the top-100
      happened to carry, so there is no comparison left to make and no way to
      tell a good setting from a lucky lineage. The fleet can find a good
      configuration and cannot say *why* it is good.
    - **And nothing anchors the search** except the ranking that picks parents —
      and `HP_MULTIPLIERS` is not symmetric in log space, so every axis of every
      lineage shrinks ~1.35% per generation on average with no cohort at the
      centre pulling against it. Accepted as it is.
  - **The arm is recorded in every published model** (`hp_arm`: `inherited`,
    `sampled-fallback`, and `sampled` only from a hand-driven `poker-loop` with no
    plan). Not bookkeeping: an inherited point correlates with its parent's
    quality *by construction*, because the parent was drawn from the top 100, so
    it cannot be read as a response curve. `poker-train` records the field and
    never acts on it — which arm a worker is in is a `poker-loop` decision.
  - **The fallback is the whole of the non-inherited population, and it is not a
    choice.** A worker can only inherit from a parent whose checkpoint actually
    carries the metadata; with no parent, or one without it, there is nothing of a
    parent's to perturb, so that worker perturbs the fleet's own starting values
    and is labelled `sampled-fallback` (the name predates the change, when it drew
    a rung of a ladder, and stays because it is recorded in published models and
    read by the monitor and dashboard). The generation header prints the count,
    which is the only signal that inheritance is working at all: a metadata
    regression would quietly turn every worker into a restart from the flags.
  - **Hyperparameters are inherited only from the weights parent.** Perturbing
    the settings of a model this worker is not resuming from would attribute a
    configuration to a run that never had it.
  - The plan is reproducible from `--seed-base`.
  - **`inherit_hyperparameters = false` stops a worker starting from its parent's
    settings.** Every worker then perturbs the fleet's own values (`[starting-point]`
    in `config.toml`, or the flags) by one of the multipliers, whoever its parent
    was, labelled `sampled`: one independent step from the same starting point, so
    nothing drifts off it and the points *can* be read as a response curve (the
    `sampled` arm's original meaning). The weights are still inherited, and the
    optimizer, if switched on, still steers the multipliers. The supervisor prints
    "non ereditati" and `hyperparameter_plan` is called with every parent `None` and
    `fallback_arm=HP_ARM_SAMPLED`. `launch_worker` passes `--parent-label` only to a
    worker whose arm is `inherited`: one that stepped from the toml took no step from
    its parent, and a "toml minus parent" difference would put the parent lineage's
    drift into the optimizer's evidence as if it were an effect, so no observation is
    written. `--no-inherit-hyperparameters` is the flag; re-read every generation.
  - **An optimizer steers the multipliers, toward Elo gained per unit of compute**
    (`rl/sweep_optimizer.py`, `rl/sweep_log.py`; pure Python, no torch; `--sweep-*`
    flags and the `[sweep]` section of `config.toml`; `--no-sweep-optimizer` restores
    the uniform draw). The multipliers are still `HP_MULTIPLIERS`, still per axis;
    what changes is *which one* an axis takes.
    - **The evidence is one file per finished child**, `<global-dir>/sweep/<label>.json`
      (written once, atomically, by `train.py::record_sweep_observation` right after
      the model is published): the parent's label and published rating, the child's
      published rating, the **CPU seconds** the child cost up to publication
      (`time.process_time()`, not wall time: a machine running twenty workers would
      bill a setting for its neighbours) and both sets of axes, so the step is the
      difference. Only a worker that resumed from a named parent with settings on
      record writes one (`--parent-label`, passed by `launch_worker`; a no-parent
      worker has no step to report). **Models published before this existed carry no
      parent and cannot be backfilled.**
    - **Gain is `child rating - parent rating`**, both as published (earned against
      the pinned anchors), frozen in the file, so a pruned parent takes no evidence
      with it. The parents come from the top of the ranking, so their ratings are
      biased up and the mean gain is negative for reasons unrelated to settings; the
      regression carries the parent's rating as a nuisance term for that.
    - **The model**: a Bayesian ridge regression of gain on the *realised* log-steps
      of every axis (`log_step`: the GAE lambda on its complement, a move off exactly
      0 as a fixed step), prior sd `PRIOR_SD` 15 Elo per unit log-step so twelve noisy
      coefficients do not chase noise; plus a ridge regression of log CPU on the log
      *levels*, which gives each axis a **cost elasticity**. The value of a step is
      `beta - G * elasticity` with `G = max(mean gain, --sweep-min-gain)`: the
      derivative of `gain / cost`, floored so a fleet whose children no longer
      improve does not turn the objective upside down (with `G <= 0` more cost would
      look like a benefit).
    - **Thompson sampling, one posterior draw per worker**, so where evidence is thin
      workers disagree and the fleet keeps trying both directions. An axis takes the
      largest multiplier if its sampled value is positive, the smallest otherwise;
      with probability `--sweep-explore` (0.35) it is drawn uniformly anyway, because
      an axis every worker moves the same way stops varying and stops being
      estimable. **Uniform until `--sweep-warmup` (150) children have been seen**, and
      then identical to the old draw from the same rng.
    - **Expect it to be slow and weak.** A gain is the difference of two ratings each
      with sd ~10 Elo, a x1.2 step is worth a couple, so firming up twelve axes takes
      hundreds of children (~6 generations of 25 workers just to warm up). The model
      is linear in the step and fitted over the last `--sweep-window` (1500) children:
      it says which way is uphill *where the fleet has been lately*, not where the
      optimum is.
    - **It never stops a generation**: a failure reading or fitting prints
      "ottimizzatore saltato per errore" and the draw is uniform. **`sweep_optimizer =
      false` (the shipped setting) still fits and prints the report** every generation,
      marked "spento: solo osservazione", and only withholds the policy, so the draw is
      the old uniform one while the evidence and the estimate keep accumulating (the
      observation files are written by `poker-train` whatever the switch says); that
      is how one decides whether to turn it on. Each generation
      prints the estimate per axis (effect of a x1.2 step, gross, cost %, net, and
      the probability it is worth it) in the supervisor log -- **read it before
      trusting it**; an estimate whose intervals all straddle 0 is the optimizer
      saying it has learned nothing, and the fleet is then just the old sweep with
      35% fewer holds. `parent_label` is in `config.LOCAL_ONLY` (a per-run value).
  - **The response variable is `benchmark_bb100`**, recorded in the same
    metadata: the pass against the anchors plays the published model over 500
    sessions of 1,000 hands, so 500,000 hands, at no extra cost because those
    hands are played anyway.
  - **Run lengths are deliberately ragged**, since `--hands` is swept: a worker
    drawing 320 finishes in well under half the time of one drawing 800. The Elo
    fill-in phase below is what the difference is spent on.
  - **Each worker prints what it was configured with, into its own log**
    (`phases.hyperparameters_marker`, `iperparametri: hp_arm=... lr=... hands=...`).
    The values exist in three places -- the worker's command line, the
    supervisor's log and the metadata inside the published model -- and *none of
    the three is what a watcher reads*: `--status` and the dashboard parse the
    worker's own log and nothing else. It also makes an archived log
    self-describing.
    - Deliberately `key=value` pairs, not a fixed-shape line:
      `parse_hyperparameters` hardcodes no key, so adding an axis needs no change
      in the watcher. The line is built from `run_metadata`, so it cannot
      disagree with what rides inside the checkpoint.
      `test_every_axis_of_the_sweep_is_one_the_worker_reports` fails if an axis
      is added to `HP_AXES` and not to `train.REPORTED_AXES`.
    - **Where to read it**: `--status` gains one `sweep :` line (how the
      generation splits between the arms, and the range of `lr` and `hands`) --
      one line, because the per-worker table has no room for more columns. The
      dashboard carries the full per-worker set in the **expanded panel** (click a
      row), above the training curves, plus a per-machine pill with the arm
      split. It rides in the `/api/status` payload the page already polls, so
      opening a panel costs no extra request. A log without the line records
      nothing, and every reader treats that as normal rather than as an error.

- **A worker that finishes early fills the wait with Elo passes instead of
  idling** (`train.run_elo_fill_in`, `--elo-fill-in`; on by default in
  `poker-loop`, off by default in `poker-train`). Run lengths are deliberately
  ragged — `--hands` is swept — so several workers a generation finish well
  before the slowest, and on a big box that would be hours of cores doing
  nothing every generation.
  - **Why Elo and not more training.** Training more would publish more models,
    and the store's problem is not that it holds too few but that few of them
    have a rating worth anything: a pass seats ~50, so a given model comes up
    rarely and can sit for many generations on the number its own training run
    published. And a longer run is not comparable with the others in its
    generation, which is the whole point of the sweep — the fast workers are fast
    *because* they drew fewer hands, so extra iterations would erase the very
    axis being measured. Rating passes touch no published model's weights, only
    what is known about them. The fast workers become the fleet's rating engine.
  - **When it stops: a flag, or a cap in minutes.** The supervisor's stop flag
    ends the phase as soon as it is seen (checked between passes, so the pass in
    progress finishes; a flag already up means no pass at all -- there is no
    minimum, `--fill-min-sessions` was removed). `--fill-deadline-minutes`
    (**150**) is the *only* cap, and it
    is expressed in minutes on purpose: a maximum number of passes would cap the
    work, while what actually has to be bounded is how long a worker can hold its
    core when nobody is coming to release it (a supervisor killed
    mid-generation, a state directory that moved). Pinned by
    `test_the_cap_is_only_ever_expressed_in_minutes`.
  - **The handshake, and the deadlock it exists to break.** A filling worker
    does not exit — it is waiting for its supervisor — and a supervisor that
    blocked on the worker's exit would wait for exactly that worker, each holding
    the other until the worker's deadline expired. So the worker creates
    `draining` in its own scratch directory, meaning "I am done with my own work
    and only killing time"; the supervisor polls every `FILL_POLL_SECONDS` (10)
    and, when **every worker still alive** says that, creates `FILL_STOP` in its
    state directory, which the workers check between passes. Both names live in
    `rl/phases.py`, with the stage names and for the same reason: a name that
    drifted between the two modules would not fail, it would leave every worker
    filling until its deadline.
    - Both files are on the machine's **own** disk, never the shared volume: a
      shared stop flag would have the first machine to finish a generation
      release every other machine's workers too.
    - The stop flag is **removed at the start of every generation**, before
      anything is launched — a flag left by the previous one would release this
      generation's workers the instant they reached the phase.
    - The `draining` marker is removed in a `finally`, the error path included,
      or the next generation's supervisor would read a stale directory as a
      worker already draining.
    - `wait_for_workers` polls instead of blocking and still reports every
      non-zero exit, which is what the generation record and the fleet-outage
      diagnosis both read.
  - **A fill-in pass is small and biased: `--global-sample` 50 models,
    `--fill-sessions` 10, so 10,000 hands and a few minutes.** The
    end-of-run pass is 100 sessions (`--global-sessions`) and takes far longer; a fill-in pass is a
    unit of *waiting*, so it has to be short enough that the worker notices the
    stop flag soon after it goes up and short enough to re-draw often — a long
    wait is then many independent draws rather than one stale one, and the
    ranking is re-read every pass because the previous one just moved it. What
    it does not avoid is loading the drawn models, which is why it is not made
    shorter still.
  - **It draws through `tiered_draw`, and so does the end-of-run pass.** The
    seats are split equally between the tiers of `--draw-tiers` (`draw_tiers` in
    `config.toml`, same syntax as `parent_tiers`: `"10, 100, 1000, all"` is a
    quarter each of ranks 1-10, 11-100, 101-1,000 and everyone else). Tiers are
    nested cutoffs turned into disjoint bands, a tier listed twice gets twice the
    share, three tiers get a third each, and unrated models join the `all` band.
    The workers get the value from the supervisor like every other flag. With the
    default the top ten cannot fill their 12-13 seats, so they are all seated in every pass
    and the shortfall spills uniformly over the models not yet drawn. The top is
    the only part of the ranking anything reads — `pick_parents` draws from the
    best 100. **The fill-in passes lift `trigger_size` to `NO_PRUNE_TRIGGER` and
    can never prune**; the end-of-run pass does *not*, see the next section for
    what that costs. Pinned by `test_a_fill_in_pass_can_never_prune`.
  - **Torch-free at import** (it reaches torch only through `global_arena`'s
    function-local imports), so its tests run in the ordinary suite and it starts
    instantly. Pinned by a test.
- **Trigger for a pass: the end of every single `poker-train` run**, not once
  per `poker-loop` generation — `--global-elo` (on by default;
  `--no-global-elo` to opt out) is checked as the very last thing `main()`
  does, in both standalone `poker-train` and every worker `poker-loop` spawns.
  Many workers finishing together is fine: they merge concurrently.
- **A pass is N sessions, with no per-model debt.** `play_global_sessions(sessions=N)`
  plays N sessions of `--session-hands` (1,000) hands; each seats
  `num_players` distinct models drawn uniformly at random from the sample (the
  ~50 drawn by `tiered_draw` plus the anchors), so how often a model plays is
  whatever the draw gives it (~`N x players / sampled` on average, with the
  spread chance brings). It replaced "every drawn model owes G games", which
  needed a queue to guarantee the count, a progress bar in owed games, and a
  second knob that meant "per model" in one tool and "per pass" in another.
  Defaults: `DEFAULT_GLOBAL_SESSIONS` 100 (end of run; the old 12 games each was
  ~110 sessions), `DEFAULT_FILL_SESSIONS` 10 (fill-in; the old 1 game each was ~10),
  and the two hand-run tools measure their length in total sessions, not passes:
  `benchmark_arena` by its convergence rule (`--arena-min-sessions` ..
  `--arena-max-sessions`) and `poker-elo` by `--elo-sessions`, which it plays as
  successive samples of `--global-sessions` of them (the last takes what is left).
  `--games-per-model` and `--passes` are gone from those two CLIs. The sharded path
  splits the sessions exactly (`shard_sessions`).
- **Sampling is by rating band, in every pass that draws from the population**
  (`global_arena.tiered_draw`, passed through the `draw` hook): `run_population_sessions`
  draws `--global-sample` (~50) models from `discover_population(root)` — every
  model in `checkpoints/models/` — plus `--global-benchmark-sample` (~5)
  uniformly from `discover_benchmark_population`. The population draw gives
  each tier of `--draw-tiers` an equal share of the seats (by default a quarter to
  each of ranks 1-10, 11-100, 101-1,000 and the rest; the end-of-run pass and the
  fill-in passes alike). Pinned by `test_three_tiers_give_a_third_of_the_seats_each`.
  - **Cost, accepted: the end-of-run pass prunes under a biased draw.**
    Eligibility for deletion is a percentile of `games`, so the top bands (seated
    every pass or every other pass) become eligible sooner and the tail stays
    below the percentile and immune; `eliminate_lowest_rated` then removes the
    lowest rated *among the eligible*.
  - A model never before seen is bootstrapped at the default rating when it
    first plays; a benchmark draw is bootstrapped `frozen=True`.
- **K follows a 20-step staircase approximating the optimal gain, and one
  schedule rates everything.** A flat K made a model with a thousand games move
  as much per session as a newcomer, so a well-determined rating still jumped
  around. `PoolRegistry(k_schedule=...)` gives each registered member its own K
  from its own `games` (`k_for_games`, tiers in `DEFAULT_K_SCHEDULE`): **20 tiers
  from 16 down to 0.01**, each a factor 1.4745 below the one above — 16 below 20
  games, then 11, 7.4, 5.0, 3.4, 2.3, 1.6, 1.05, 0.72, 0.49, 0.33, 0.22, 0.15,
  0.10, 0.07, 0.047, 0.032, 0.022, 0.015 and 0.01 from 60,000.
  - **Where those numbers come from, because it is derived rather than tuned.** A
    1000-hand rated session measures a rating with a standard deviation of about
    **238 points** — Elo reads only the *sign* of each pair's chip delta, and the
    zero-sum structure correlates the learner's five pairwise comparisons at
    exactly 0.5, so they are worth 2.18 independent Bernoullis. For a quantity
    that does not move, the gain that extracts all of that evidence and no more is
    Kalman's, and in Elo's parametrisation it is exactly hyperbolic:
    `K_t = 1/(slope·(t + V/P0))` with `slope = 0.001421` and `V = 238² = 56,864`.
    The staircase is geometric in `(t + 35)`, and the two ends fix the ratio:
    `16 / 1.4745^19 = 0.01`, with 0.01 sitting at `704/0.01 − 35` ≈ 70,000 rated
    games. A staircase is necessarily low at the start of a tier and high at the
    end; at this ratio the worst tier is ~4.5% off the curve.
    Pinned by `test_the_staircase_tracks_the_hyperbolic_gain_it_approximates`,
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
    loop. At this offset the inherited value carries only ~6% of the weight of
    the final rating after 600 rated sessions.
  - **The bottom tiers are headroom, not live tiers.** The thresholds at the
    bottom exist because game counts only grow; the bottom tier is inside the
    range the *frozen anchors* occupy, and their deltas are discarded anyway — so
    read anything below ~0.2 as a promise the curve keeps rather than as a live
    tier. Count only the **non-frozen** members when asking which tiers are live:
    counting the anchors in makes the low tiers look live when they are not.
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
    **There is no flat K anywhere** (`DEFAULT_K_FACTOR` and `PoolRegistry.k_factor`
    were removed): every K comes from the schedule in `config.toml`, and
    `pairwise_elo_delta` *requires* a K for every participant and raises
    `ValueError` for one without, rather than guessing.
  A session is rated at the experience the model had *when it sat down* (K read
  before `games` is incremented), an explicit `k_factors` entry wins over the
  schedule, and an unregistered participant nobody names is an error. Two
  consequences: a session is not exactly zero-sum in rating (each side moves by
  its *own* K). That does not matter here: frozen anchors never move, so exact
  conservation never held, and a ranking only needs the *order* to be right. A
  veteran converges more slowly to a genuinely changed strength, which is the
  intended trade. A registry built without a schedule (`k_schedule=None`) can
  only be read, or rate sessions whose every K the caller names in `k_factors`:
  the in-memory registry of a training run is one, and `SelfPlayTrainer` names
  the K of the learner (which is not a member, so there are no `games` to look
  up) and of each frozen member it seats.
- **Frozen Elo anchors**: `PoolMember.frozen` pins a member's rating forever (the
  benchmark anchors, and the opponents in a training run's in-memory registry) —
  `record_session_with_ratings` still scores it normally against everyone else
  and still counts its `games`, it just never applies its own delta. A frozen
  member is a fixed reference point the rest of the scale is measured against.
  - **`rl/benchmark_arena.py` is the single deliberate exception**, and the only
    code in the project that writes an anchor's rating. Nothing imports it; it
    is run by hand (`python -m pokerlab.rl.benchmark_arena --workers 10`), rarely.
    It exists because an anchor is pinned at whatever rating it held the instant
    it was promoted — one number from the ordinary passes, earned against a field
    that no longer exists — and the anchors are never rated against *each other*,
    so nothing else checks that they sit correctly relative to one another. It
    plays them among themselves and settles that, printing every anchor with its
    rating at every checkpoint. It bypasses `record_session_with_ratings` (which
    would refuse) and writes the member files itself, one at a time under its own
    lock, re-reading each so only `rating` is overwritten and
    `games`/`ref`/`frozen` survive.
    - **How long it runs is measured in sessions, never in passes** — the
      convergence rule is in sessions too. The run breaks when
      `sessions >= --arena-min-sessions` **and** `drift <= --arena-tolerance`,
      where `drift` is the **net** movement — where a rating sits now against where
      it sat at the latest checkpoint at least `--arena-window` sessions ago, not
      the distance travelled in between, so a rating jittering around a settled
      value scores ~0 however far it wandered. It is the **worst** anchor, not the
      mean: one anchor still moving means the order is not settled however quiet the
      others are. Until a checkpoint that old exists `drift` returns `inf`, so
      convergence can first fire after `max(min-sessions, window)` sessions.
      Defaults: tolerance 1.0, window 4,000, min 10,000, max 100,000 (the old 20, 50
      and 500 passes of 200 sessions).
      - **`--arena-max-sessions` is a safety cap, not a failure, and it is exact.**
        Hitting it prints "NON CONVERGE" and then does everything a converged run
        does — the ratings were already written at every checkpoint and the snapshot
        is refreshed. It only means the ratings were still moving.
        `--arena-max-sessions 0` plays nothing and only refreshes the snapshot.
      - **`--arena-tolerance` is not scale-free: it has to be read against the K.** An
        anchor moves at most `k x 0.5` per session (`pairwise_elo_delta` divides
        each pair's change by the opponents faced), so at most
        `k x 0.5 x` the sessions an anchor plays in a window (`--arena-window` x
        players / anchors on average). A tolerance of 1.0 is a real test only when
        the K is small enough that a window moves an anchor by about that much; with
        a large K a single window can shove an anchor further than the whole ladder
        spans and the tolerance will essentially never be met, so the run always
        burns `--arena-max-sessions`. Read the tolerance against the K the anchors
        actually sit at.
    - **It plays in batches of `--arena-checkpoint-sessions` (200, `config.toml`),
      and a batch is a checkpoint, not a pass**: it decides nothing about the
      ratings a run ends with — only how much of a killed run is lost (at most that
      many sessions), how often someone watching sees the anchors move, and the
      granularity at which the convergence test runs (so convergence is detected at
      the next checkpoint, up to one batch late). The write is cheap, so a small
      value costs nothing in process; with `--workers` every batch spawns its shards
      afresh and each loads every anchor, so a very small one pays that startup
      over and over (the startup projection counts it). The last batch takes what is
      left of the cap. Pinned by `test_the_arena_never_plays_past_the_maximum`,
      `test_the_arena_stops_when_the_ratings_settle_but_not_before_the_minimum` and
      the `drift` tests.
    - **At every checkpoint it prints the per-group line and the full per-anchor
      table** (`format_series_line`: each directory group's mean rating, its span,
      and how far its mean has moved since the run started), because watching the
      anchors sort themselves out is the whole point of the run.
    - **It writes the ratings at every checkpoint, not once at the end.** The
      write is one small file per anchor, each under its own lock and re-read so
      only `rating` is overwritten -- milliseconds against minutes of play. Writing
      once at the end would lose everything if a multi-day run were killed.
      - **The shared `registry.json` snapshot is refreshed in the same breath**,
        forced, at every checkpoint. The member files are the truth but nothing reads
        them one at a time: `--status`, the dashboard and the GUI's default table
        all read the snapshot, and `write_snapshot`'s own staleness rule only
        rewrites it when some other writer happens along. A long run holds
        the anchors for *days*, so refreshing only at the end would leave the whole
        fleet reporting the ratings the run started from. A failure is caught and
        reported rather than allowed to end a multi-day run. `--dry-run`
        refreshes nothing, like every other write. Pinned by
        `test_every_checkpoint_refreshes_the_shared_snapshot` and
        `test_a_dry_run_refreshes_nothing`.
    - **`--workers N` shards each batch across N local processes**, reusing
      `global_arena.play_sharded` -- playing parallelises perfectly, rating does
      not, so the sessions come back and are rated here in one place. It is the
      only thing that makes a run finish in hours: it is the sessions
      of `--session-hands` hands up to the cap. A projection of the minimum and the
      cap is printed at startup, so a two-day run cannot be launched by accident.
      - **The projection has to include the shard startup.** Every batch spawns
        `workers` fresh processes and each imports torch and loads *every* anchor
        before playing a hand — `workers x anchors` checkpoints pulled over one NFS
        mount — so the cost goes with their product, not with one shard's startup.
        The projection adds `batches x workers x anchors x 0.0039` minutes (calibrated on a
        single point, so read it as an order of magnitude). Note how the startup
        term can dominate: *more* workers stops helping well before the cores run
        out, and the number grows faster than the anchor count since both terms
        scale with it.
      - **Never raise `OMP_NUM_THREADS` instead.** `OMP_NUM_THREADS` sizes
        torch's *intra-op* pool -- how many threads split one tensor operation --
        and here that operation is a batch-of-one forward pass through a small
        MLP. Threads inside one process buy almost nothing, while one process per
        core scales with the core count (measured on a 32-core box: one process
        with 15 threads ≈ 1.1x one thread, fifteen one-thread processes ≈ 15x),
        because torch is only about a third of a hand's cost; the rest is
        `encode_observation`, the legal-action masks and the betting state —
        Python bytecode no torch thread can touch, and which the GIL would
        serialise even if they were Python threads. Shards are therefore launched
        with `OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`, since torch may route
        an op through MKL rather than OpenMP and each reads its own variable).
        Under load it is far worse than 1.1x: threads of several processes
        contending for the same cores collapse, and a fleet machine running 10-20
        workers is always in that regime, so the safe setting is one thread per
        process, always.
    - **A flat K is what keeps the anchors' mean fixed** (the arena uses the
      staircase, see below, so this is a property to know rather than one it has):
      `pairwise_elo_delta` is zero-sum for a flat K (a pair's two deltas are equal,
      opposite and divided by the same count), so a closed population playing
      itself redistributes rating without moving its centre. The per-experience K
      schedule breaks that cancellation.
    - Every rating is written to a timestamped JSON backup under
      `global/benchmark_arena_backups/` before anything is touched, and
      `--restore <file>` puts them back (`games` and `frozen` preserved).
      `--dry-run` plays and reports without writing.
- **Pruning trigger: the real on-disk population reaches
  `--global-trigger-size` (10,000).** That is the whole rule: whenever
  `len(discover_population(...))` — the number of files in `checkpoints/models/`
  (one per network) — is at or above the trigger at the end of a merge, a
  pruning pass runs; after it removes ~a quarter of the eligible models the count
  drops below the trigger, and it fires again when the backlog has grown back to
  it. It is *not* keyed to how many models the ledger has rated (the ledger only
  grows by the handful each pass samples, the real population is the whole
  backlog). Pruning is serialised by an exclusive `__prune__` lock (30-minute
  TTL) that stops two machines pruning — or admitting the same new benchmark
  anchor — at once; it does **not** stop ordinary merges, and the population is
  recounted *after* taking it so a pass that another machine just finished is not
  repeated on a stale count.
- **Elimination physically deletes the checkpoint file. This is deliberately
  destructive, not bookkeeping.** `_eliminate` removes `--global-eliminate-
  fraction` of the *eligible* members: the non-frozen ones whose `games` is at or
  above the `--global-protect-percentile` (**25th**) of games played among the
  models rated so far — never below it, since a rating built on a handful of
  games is not evidence. The percentile is low on purpose: eligibility is a
  *percentile of games*, so any draw that seats some models more often than
  others — and `tiered_draw` deliberately does — makes the well-played eligible
  sooner and leaves the rarely-drawn tail permanently immune, which would have
  the pass eat the middle of the population instead of its bottom. A lower
  percentile widens the eligible set; the accepted cost is that a pass removes
  more models, so prunes fire less often and bite harder. **There is no absolute
  minimum-games floor**; with nothing rated yet the percentile is undefined and
  no pruning happens. Each doomed model is then locked individually and re-read
  fresh, so one being rated at that moment is skipped rather than deleted from
  under a merge. Every confirmed model has its `.pt` removed by
  `delete_checkpoints` (resolved through a fresh `discover_all_copies()`, never
  through `ref`) and its member file removed. **Nothing is promoted to the
  benchmark by pruning** — see the next section.
- **How a model becomes a benchmark anchor** (`global_arena.py::
  add_benchmark_candidates`, called by `train.py` right after the run's
  population pass):
  - Population = non-frozen members whose checkpoint is in `models/`. A model
    qualifies when its `games` are **above the 90th percentile**
    (`--benchmark-games-percentile`) of that population's games and its rating is
    **more than 10 points** (`--benchmark-margin`) above the best anchor; both
    are in `config.toml` `[anchors]` and `poker-loop` forwards them to the workers,
    the only place `add_benchmark_candidates` runs.
  - Candidates are taken from the lowest rating upwards and each must also clear
    the previously added one by more than the margin, so anchors stay at least 10
    apart instead of a cluster joining at once.
  - An added model is *moved* into the flat `checkpoints/benchmark/`, every other
    copy deleted, `frozen=True` at the rating it holds, `ref` rewritten. No
    re-settling by hand. Held under the `__prune__` lock so a prune cannot
    delete it mid-move.
  - **Every addition requests a `benchmark_arena` run**
    (`global_store.request_benchmark_arena`, claimed by one supervisor between
    two generations, `loop.py::run_requested_benchmark_arena`): launched with no parameter flags, so it
    runs with what `config.toml` sets under `[benchmark-arena]` (`arena_min_sessions`,
    `arena_max_sessions`, `arena_tolerance`, `arena_window`, `arena_checkpoint_sessions`; the supervisor only forwards `--config` when it was given
    one), logged to `arena-gen<N>.log`. A failed
    run is not retried. **The arena always uses the hyperbolic K staircase**:
    each anchor's K comes from its own `games`, so a session is not zero-sum and
    the anchors' mean is not exactly fixed — veterans at the bottom tier barely
    move, a new anchor moves most.
  - **Do not use `delete_checkpoints` on an anchor**: `discover_all_copies` only
    looks in `models/`, so it silently deletes nothing from `benchmark/`. Removing
    an anchor's member file and leaving the `.pt` behind lets the live fleet
    re-register the orphan as a frozen anchor at 1500 with few games. Delete the
    file *and* the member together, and never one without the other.
- **`ref` is a real, maintained path, not a hint.** A member's `ref` is where its
  checkpoint is *now*, relative to the same root the passes run from
  (`checkpoints/...`). It is kept current three ways: promotion into the
  benchmark rewrites it; every pass that plays a model refreshes its `ref` from
  the disk discovery it already did (`_apply_session`); and
  `repair_member_refs(global_dir, root, machine=...)` sweeps all members at once
  for the models that have not played since their file moved. A stale `ref`
  would make `discover_global_top_models` silently drop top models from the
  GUI's default table, so that function also falls back to finding the
  checkpoint by label, degrading to "a bit slower", never to "missing".
- **Hard rule: `registry.json` under `checkpoints/global/` is a snapshot, never a
  source of truth.** The truth is the member files. Write it only through
  `write_snapshot`, and never read it to *decide* anything that changes state
  (merging and pruning read `load_global_registry`); it may be minutes stale, and
  anything hand-edited there is overwritten by the next snapshot. The corollary
  is that **whatever writes member files has to refresh the snapshot itself**, or
  its work is invisible to every reader: a training merge does it as its last
  step, and `benchmark_arena` and `poker-elo` force one after every pass,
  because both can run for hours or days without any other writer coming along
  to do it for them.
- **Ghosts**: `prune_ghost_members` drops the member files of models whose
  checkpoint exists nowhere any more, compared against the on-disk label set and
  not against `ref`. An *empty* label set (an unreadable volume) drops nothing,
  so a transient NFS problem cannot wipe the ledger.
  - **The label set is a snapshot, and a snapshot goes stale.**
    `apply_pending_population_sessions` lists the disk, then merges pending
    sessions, which on a busy fleet can run for tens of minutes while other
    machines keep publishing new models and registering them. Each newcomer
    appears in `list_member_labels` but not in the snapshot, and would be swept as
    a ghost: a live model, freshly rated, silently reset to rating 1500 with zero
    games. The signature in a `poker-elo` log is bursts of "fantasmi rimossi"
    landing on the longest passes and nothing on the short ones.
  - **How to tell a false positive from a real ghost**: nothing in this project
    deletes a checkpoint except a prune. With pruning off (`poker-elo` without
    `--prune`) and the population under `--global-trigger-size`, **any ghost at
    all is a false positive.**
  - **The fix is to ask about one label at the moment it matters.** The sweep
    re-lists the disk immediately before running (instead of reusing the
    pre-merge snapshot), and `checkpoint_on_disk(root, label)` re-checks each
    candidate **after its lock is taken and immediately before removal**, which
    is the only instant the answer is authoritative; the snapshot is then just a
    shortlist. It looks in `models/<label>.pt` and in `benchmark/` recursively,
    because a check that looked only in the store would delete every anchor's
    rating, which is the fixed point the whole scale rests on. Three tests cover
    it, including that a genuinely missing checkpoint is still dropped: the
    re-check must narrow the sweep, not switch it off.
- **Why pruning exists**: nothing else ever removes a model, so without it the
  store grows forever (models are added every generation, ~3 MB each). Real
  physical deletion, gated behind an Elo signal reliable enough to trust (the
  percentile-of-games protection) and a real disk-population trigger, is what
  bounds it.

## Ranking resolution at the top

The global Elo orders the population correctly over large gaps and cannot
resolve fine differences near the top, which is the part everything the loop does
reads: `pick_parents` draws uniformly from the top 100.

- **It is a power limit, not an Elo bug.** At a gap of a few bb/100 a session of
  `--session-hands` = 1,000 picks the stronger of two adjacent strong
  models only a little more than half the time, and ninety per cent would need
  tens of thousands of hands a session. No K schedule fixes that: the evidence
  per session simply is not there (see "How long must a session be?").
- **What matters for `pick_parents`** is whether the top-100 *band* is assembled
  correctly, not whether #20 outranks #100 within it: a draw that treats the
  whole band identically is indifferent to fine ordering inside it. The coarse
  ordering must be sound.
- **How to check the ordering with duplicate decks**: 3 copies of one model
  against 3 of another, then the same shuffles replayed with the teams swapped,
  so card luck cancels. `Table` consumes its `random.Random` only to shuffle,
  once per hand, so a fixed seed deals the identical sequence no matter how the
  betting goes; `(a - b) / 2` over the two arrangements is the skill effect
  alone. Always run a null control (a model against itself) — it must read ~0 —
  so the estimator is known to invent nothing. `rl/duel_power.py` does this.
- **Tried and rejected**: weighting the Elo update by chip margin (two variants,
  both turned out to be a lower effective K in disguise — see the TODO entry on
  the bb difference), and lowering `DEFAULT_K_SCHEDULE` is what improved ordering
  in simulation. The K schedule is derived from the session's statistical power,
  not tuned, so do not edit its numbers by hand.

## How long must a session be? (`rl/duel_power.py`)

Everything the ranking does assumes that the model finishing a session ahead is
the better one. That is not a fact, it is a probability, and it depends entirely
on the session length. `poker-train`, the population pass and every rated
result rest on it, so it is worth measuring rather than assuming. Run by hand,
nothing imports it:

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
  `Table` touches its `random.Random` only to shuffle, once per hand. Mirroring
  does not move the estimate (the pooled mean is exactly the mean of the paired
  differences), it sharpens it, by roughly an order of magnitude.
- **The normal approximation extrapolates.** With a per-hand edge `mu` and noise
  `sigma`, a session of `H` hands is won with probability `Phi(mu*sqrt(H)/sigma)`,
  and both are measured from the *total* hands played. So the report shows a
  measured frequency wherever there are enough blocks and a prediction
  everywhere, including at lengths nobody played. The two columns side by side
  are also the check on the approximation: they must disagree at ten hands (the
  per-hand delta is heavy-tailed) and agree from a few hundred on.
- **The null control works**: the same model against itself returns an edge of
  exactly +0.00 bb/100 and every session an exact 50/50.

What to read from a run: the real edge per seat (with its interval), the
per-hand noise `sigma`, how often a `--session-hands` = 1,000 session picks
the stronger model, and the hands needed for 95% and 99%. The frequency at 1,000
hands is the one that matters: whatever fraction of sessions picks the wrong
model moves the ratings the wrong way. For two adjacent models the edge is small
and the hands needed scale with its *square*: halving the edge quadruples the
hands. It also says the right direction for any fixed hand budget is fewer,
longer sessions -- the same conclusion reached on
`table_mix.DEFAULT_SESSION_HANDS`. **Re-run it on the current population**
before trusting any figure from an earlier one: the edges depend on the models.

- **The distribution, not just the frequency.** `format_distributions` draws
  what 100 sessions of each length actually looked like. The median barely
  moves with the length because it is the real difference between the two
  networks -- only the scatter around it shrinks, as `1/sqrt(hands)`. Sessions are drawn
  round-robin across streams, not as consecutive blocks of one, since part of
  what makes two sessions differ is which seats each model drew.
- **`--mode field` measures what a population pass actually measures**: one
  seat each and the rest filled from the frozen anchors, instead of half the
  table each. Its per-hand noise is expected to be higher than the 3v3 duel's (a
  3v3 duel averages over three seats), so it needs correspondingly more hands for
  the same confidence; measure it before resting anything on that.
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

- **The bot builder seats trained models, and only trained models**:
  `AddBotDialog` offers a "Modello addestrato" combobox listing the 20
  best-rated checkpoints found across every machine's pool, newest ratings
  first, and nothing else. The spec it produces is
  `{"key": "model", "path": ...}`, which `_bot_spec_to_key_string` renders as
  `model:<path>` — the same string the CLI understands, so nothing in
  `build_players` needs a GUI-specific branch. "Aggiungi" is disabled when no
  checkpoint is found, rather than offering a choice that would fail on
  confirm.
- **Setup screen bot builder**: `SetupFrame` has no free-text bot field and no
  "number of players" field; `self.bot_specs: list[dict]`
  (each `{"key": "model", "path": ...}`) drives a row of rectangles (one per
  configured bot, each with a "-" to remove) plus a trailing "+" that opens
  `AddBotDialog` to pick a trained model. `num_players` is *derived*
  (`human_seats + len(bot_specs)`, capped by `GameConfig`'s own 2-9
  validation) rather than typed separately — keeps the visual builder as the
  single source of truth. `_bot_spec_to_key_string` turns a spec back into
  the exact `model:<path>` string `build_players` and `validate_bot_key`
  understand.
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
    **On the `v2` branch the models in `top_models/` are the v1 ones and are
    provisional**: the folder stays, but its contents must be replaced with the
    first good v2 models (via `push_top_models`) as soon as they exist. Once v2
    changes `OBS_DIM` or `FEATURE_VERSION`, `check_compatible` rejects these files,
    so the GUI's default table, the bot picker and the spot advisors would find
    nothing loadable until they are replaced.
- **Busted players disappear from the table**: `TableFrame._hide_seat`
  calls `grid_remove()` (not just blanking the labels) on a seat's box once
  its stack hits 0, called from both `_render_observation` (seat missing
  from `Observation.seats`) and `_render_final_stacks` (`final_stacks[seat]
  == 0`). `grid_remove()` hides the widget but does **not** clear its
  Canvas contents — a hidden seat's card canvases still have stale drawn
  items sitting in them. Filter out `seat in self._busted_seats` before
  trusting a `canvas.find_all()` count on a seat that might be hidden.

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
  (`Table.on_action_applied` -> `gui/app.py::ActionReporter`). A wrapper
  around `Player.act()` is structurally incapable of showing what an action
  did: the engine applies the action *after* `act()` returns, so a wrapper
  only ever sees the state from before it, the table would show every move
  as not yet made and be corrected only by the *next* action -- so the last
  action of a street would **never** be drawn at all, the next event being
  the new community card. `on_action_applied` carries `{hand_id, seat, player_id, name, action,
  record, observation}` with the Observation from *after* the action, and
  `ActionReporter` publishes it and only then paces -- so what the pause
  holds on screen is the finished action, chips in and pot updated. The
  Observation is built only when a hook is installed, so training pays
  nothing for it.
  - **The human is reported the same way**, which is why `GuiPlayer.act()`
    does not publish anything itself: one path, one place, post-action
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
    action.
  - Toggling step mode off pushes one release onto `step_gate` (in case a
    bot is mid-block right then) and toggling it back on drains any
    leftover release first -- without that pairing, either a bot could stay
    stuck until one extra "Avanti" click, or a stale release could silently
    skip the next real pause. Step mode can also be preset from
    `SetupFrame` before a session starts (`start_session(..., step_mode=...)`),
    since toggling it only after `start_session` already returned can lose
    the race against instant bot-only hands finishing before the checkbox
    click even lands.
- **"Who am I" is a property of the GUI, never of the Observation being
  rendered.** `_render_observation` takes the human's seat from
  `self._human_seat`, not from `observation.my_seat`. Most Observations it
  draws belong to whichever player just acted, so reading `my_seat` would label
  *that bot* "(tu)" and pass `is_me=True` to `_draw_seat_cards`, which draws
  its hole cards as an empty slot -- the seat would look folded every time it
  acted, and in spectator mode one bot would always wear the marker. The same
  rule applies to the full-size "Le tue carte" panel: with no human seated it
  draws nothing, instead of flipping between opponents' hands.
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
- **The all-in runout is invisible without an engine hook.** When everyone still in the hand is all-in, `Table` calls no
  `Player` at all for the remaining streets -- `_run_betting_round` returns
  immediately -- so a spectator that only watches `Player.act()` saw
  *nothing* between the last bet and the final stacks: the board never
  appears and the hand looks like it skipped its own ending. `Table` therefore
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
  - **Gotcha: it is easy to invert the scroll direction.** X11 reports the
    wheel as `<Button-4>`/`<Button-5>` and Windows/macOS as `<MouseWheel>`
    with a signed `delta`, so the reflex is one handler reading `event.num`.
    That field is not dependable -- on this build, a synthesised `<Button-4>`
    arrives with `num=8`. Each sequence
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
  for BET/RAISE) rather than re-deriving them from an Observation. The table
  view does not lag the log -- see the `ActionReporter` bullet above.
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
  regardless of window height -- packed as a normal top-to-bottom widget
  below the log's `expand=True` `Text` widget it would be silently clipped on
  a window shorter than the summed content height. Any future widget added below existing content should go through
  the same `side="bottom"` treatment rather than plain `.pack()`.
- **Reuses `cli/play.py`'s `build_players`** (with its optional
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
  and `ActionReporter`'s pacing and step-gating in `test_gui_app.py`. Some
  `TableFrame`/`SetupFrame` *logic* (seat hiding, spy-mode card visibility,
  bot-spec formatting) is testable headlessly too and is covered in
  `tests/unit/test_gui_app.py` — `tk.Tk()` + `.withdraw()` works without a
  real display; what's not covered is full click-driven end-to-end flows (those stay manually smoke-tested by
  driving `PokerGuiApp` headlessly and invoking rendered buttons
  programmatically, or counting a card Canvas's drawn items via
  `canvas.find_all()` — but see the busted-seat caveat above about hidden
  canvases first). **Gotcha**: creating/destroying multiple separate
  `tk.Tk()` instances in rapid succession *within one process* is flaky
  on this machine's Tcl/Tk install (an intermittent "couldn't read file
  ...button.tcl" or `invalid command name "tcl_findLibrary"` error despite
  the file existing) — the GUI tests work around it with **one
  session-scoped `app` fixture in `tests/conftest.py`** that every GUI test
  file shares, destroying only its own frames (not the Tk root) between
  tests. Never add another `Tk()` or `PokerGuiApp()` fixture: request `app`.
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
    screen, never shrinking). Centred on its chair like the others it would be
    the tallest box (cards plus actions) and run past the canvas, cutting the
    last action off.
  - **Chairs may overlap; nothing is ever cut off, and a press brings a box to
    the front.** A top chair grows upwards when its actions appear and would
    leave the canvas: `_fit_canvas` slides the whole table right/down when
    anything sticks out of the top or left, measured from each chair's
    *requested* size (`_content_bbox` -- `canvas.bbox` lags right after the
    buttons are rebuilt). Overlap between neighbours is accepted: a press anywhere on a box, buttons included, `lift()`s it
    (`_make_raisable`: a per-chair bindtag placed *first* on every descendant,
    so the clicked button still works; re-applied after every refresh because
    the action buttons are rebuilt), and whoever is to act is lifted
    automatically. The tag is a process-unique counter, never `winfo_id()`:
    Windows reuses handles, so a new chair inherited a destroyed one's binding
    whose `lift()` failed and aborted the script. Tk delivers no pointer events
    to a withdrawn root, so the tests for this `deiconify()` for their duration.
  - **Every amount on the spot screen is in big blinds**: pot, bets on the chairs, action buttons, the raise field and its
    wheel, the log, the models' advice and the status line, written as the
    client writes them ("18,5 BB", comma decimals). The engine still counts
    chips at fixed blinds of 1/2 (`ENGINE_SMALL_BLIND`/`ENGINE_BIG_BLIND`, the
    blinds the models trained at; one chip = 0,5 BB) and converts only at the
    edges: `spot.format_bb`/`bb_number` to show, `spot.parse_bb` to read input
    (comma or dot, rounded to the nearest chip). "Blind 0,5 / 1 BB" is a fixed
    label (there are no blind entries) and stacks are typed in
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
from the poker client on screen. Each reader below is built from labelled crops
collected in the vision screen; the accuracies quoted are against the crops in
`vision_data/` and hold only for the client's current look.

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
  **It must always be closable.** The overlay is override-redirect, some Linux
  window managers give such a window no keyboard focus, and `focus_force`
  before the window is mapped does nothing — so Return and Esc could be ignored
  on a full-screen window holding the mouse. `present()` therefore waits
  for the window to be mapped (with a time limit; **never `wait_visibility`**, which
  blocks forever if it never maps), then grabs and focuses it; there are also
  on-screen Conferma/Annulla buttons, double-click to confirm, right-click to
  cancel, and a 180 s self-cancel.
- **The vision screen is compact and scrolls.** It carries no instructions, for
  space. The sections sit in a canvas (`VisionFrame.body`) under a fixed title
  bar and scroll with the wheel anywhere over them: the wheel is bound with
  `bind_all` because the event goes to the widget under the pointer (a button,
  a label), `_on_wheel` ignores events outside this screen or from a popup,
  and `destroy` unbinds so no other screen inherits it. Both wheel families
  are bound, each to its own direction (`<MouseWheel>` delta, X11
  `<Button-4/5>`).
- **In the GUI: a screen of its own** (`gui/vision_view.py::VisionFrame`, the
  "Collect vision data" button on the main menu, next to "Chiedi ai modelli"),
  so collecting examples does not crowd the table; `vision_view` reuses
  `CardPicker`/`place_near_pointer` from `spot_view`, never the other way
  pass. Tests in `tests/unit/test_gui_vision.py`. Its buttons: "Zona mie
  carte", "Zona board", "Anteprima" (shows what is captured now) and "Salva
  carte" / "Salva board" (one zone each, so a crop of the cards does not need a
  board zone set) — the examples the recognition is built from.
- **Dealer button: zones and examples.** The "Dealer (6
  giocatori)" section sets one zone per seat (`regions.dealer_region_name`,
  `dealer_6_<seat>` in `regions.json`; seat 0 is you at the bottom, then
  clockwise on screen from your left, the spot screen's chair order; only 6-max
  so far, the positions differ at 9). "Salva dealer" captures every zone set and
  opens `DealerLabeler`: all crops side by side, each "Presente"/"Non presente",
  at most one present (none is fine, between hands). On "Conferma" *every* crop
  is written -- the absent ones are examples too -- to **`vision_data/dealer/`**
  with a `{"zone", "dealer": bool}` label (`labels.save_dealer_label`), a folder
  and a label shape of its own so neither dataset is mistaken for the other.
  Card actions ("Salva carte/board") capture only the card
  zones, never the dealer ones that share the regions file.
- **Player seats: zones and examples.** The "Giocatori (6
  giocatori)" section sets one zone per seat around the player box (avatar,
  name, stack; `regions.player_region_name`, `player_6_<seat>`, same numbering
  as the dealer, seat 0 = you included since you can fold too). "Salva
  giocatori" opens a `SeatLabeler` with one state per seat
  (`labels.SEAT_STATES`): `in_gioco`, `fuori` (folded, or just joined and
  waiting), `sit_out` (seated but sitting out) and
  `libero` (empty seat); every crop is written on Conferma to
  **`vision_data/players/`** with `{"zone", "state"}`
  (`labels.save_player_label`). Each seat starts from the state it was given
  last time (`VisionFrame.last_seat_states`), since between captures only a
  seat or two changes; the very first capture starts blank and Conferma waits
  until every seat has a state.
  - **Recognition: `vision/seats.py`, colour rules plus one template**
    (`python -m pokerlab.vision.seats` lists every crop where rule and label
    disagree -- the place to find a mislabelled example). Measured on the
    labelled crops: an opponent in the hand shows pink-magenta card backs (share up to 0.59, 0
    for every other state; threshold 0.10); seat 0 in the hand is read from the
    white of *your* face-up cards (0.08-0.09 live, 0.00-0.02 folded and dimmed;
    threshold 0.05 -- not their colours, which a hand of two black spades
    lacks); an empty seat shows a green chair outline (0.026, 0 otherwise);
    sit-out is the grey "SIT OUT" pill, matched by shape against the lettering
    of labelled sit-out crops (`sit_out_templates`), so that one state needs an
    example; anything else is "fuori". Caveat: the two sit-out examples are the same seat, near-identical, so the
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
    thread on every reading, a visible stutter; one grab takes ~60 ms.
  - **Dealer and player zones are separate and read separately**: the button
    only in `dealer_*` zones, the seat state only in `player_*` zones. The two
    sections' buttons read "Gettone N" / "Giocatore N" so they are not confused,
    the selector overlay names the zone being set (`select_region(label=...)`),
    and a dealer zone with a side over
    `regions.DEALER_ZONE_MAX_SIDE` (80) is flagged in the vision screen and
    **skipped by `ScreenReader`**, since a player box contains a gold stack
    chip that the dealer rule would take for the button.
  - `SeatLabeler` is the one labelling window for per-seat states;
    `DealerLabeler` is it with "Presente"/"Non presente" and an
    at-most-one-present check.
- **Bets and pot: zones and examples.** The "Puntate e
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
  - **Two-stage digit match.** The card ranks are the same typeface, but pooled
    with the bets' digits they can out-match the right answer, so: the bets'
    own digits first, and only when their best match is below `OWN_SURE` 0.8
    the card ranks alone. That
    covers digits 2-9 never seen in a bet; 0 and 1 have no card source (a ten
    is one merged glyph).
  - **Loops decide 3 vs 8** (`glyph_holes`, applied in `_Matcher`): candidates
    are restricted to examples with the same number of closed loops -- 8 has
    two; 0, 4, 6, 9 one; 1, 2, 3, 5, 7 none, measured identical for every
    example in the bets *and* the card ranks. Correlation alone is a coin toss
    between them (same right half). A ±1 px shift-tolerant match tied them
    exactly and grey-level glyphs halved the margin on every other digit, so
    both were rejected. If no example shares
    the loop count (a stray mark broke a loop) all are considered again.
  - The vision screen's amounts "Anteprima" captions each zone with its reading.
- **Stacks: zones, examples, a reader awaiting its first crops.** The "Stack (6
  giocatori)" section sets `stack_6_<seat>` zones (`regions.stack_region_name`)
  and labels crops like the amounts (`AmountLabeler`, `labels.STACKS_DIR` =
  `vision_data/stacks/`, same `{"zone", "text"}` shape). **The stacks are
  written in yellow**, so `amounts.text_mask` picks the
  lettering's colour by zone -- `yellow_mask` for `stack_*`, white otherwise --
  and then everything is shared: digits are compared as black-and-white
  shapes, so white bet digits and yellow stack digits lend each other examples
  (`build_reader` is fed both folders; `python -m pokerlab.vision.amounts`
  checks both). The yellow band (`YELLOW_HUE` 15-40, S > 80, V > 140) isolates
  the lettering, and a stack is written "103,5 BB" like a bet
  (`STACK_SUFFIX_BLOBS` = 2). Evaluate the two folders *together*: run on
  `vision_data/stacks` alone, a digit with a single example has none left once
  it is excluded, which is an artefact of the check, not of the app, which pools
  both. Live, a full reading with stacks takes ~76 ms.
  - The spot screen sets each seated player's stack **at a new hand only**,
    with the seating (`SpotFrame._apply_stacks`): the stack shown plus the
    chips already in front (the blinds, preflop). Never mid-hand: the chair's
    field (now labelled "inizio BB") is the hand's *starting* stack, and the
    engine takes every bet off it itself, so rewriting it after a bet would
    take that bet off twice. Stacks are still *read* every 0.5 s, for a
    check: each chair shows "resta X BB" -- what the engine leaves it -- and
    "≠ schermo Y BB" when the stack read now differs (`_screen_stacks`,
    redrawn whenever it moves), which is how a missed or misread action shows.
- **The spot screen rebuilds the actions from the screen** (rather than a
  show-and-compare mode): `gui/action_sync.py`, pure Python over
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
    table, so a pass of checks would be discovered only when the next card
    came. The vision screen's "Mio turno (barra del tempo)" section collects a
    `turn_timer` zone with "Presente"/"Non presente" crops
    (`vision_data/turn/`, `labels.save_turn_label`); `vision/turn.py` reads it
    with a **fixed colour rule**: the bar is bright green on a dark track, tens
    of per cent saturated-and-bright pixels with it and exactly 0 % without, so
    `PRESENT_SHARE` is 1 % -- low enough for a bar nearly spent -- and *any* hue
    counts, since countdown bars commonly turn yellow then red. Needs no
    examples (`python -m pokerlab.vision.turn` checks it on the labelled ones). `ScreenReading.my_turn` feeds
    `TableView.my_turn`: while it is up and the engine still waits on a seat
    before yours, that seat acted -- a CHECK with chips unchanged, a FOLD if
    out; facing a bet with nothing changed it is still not guessed.
  - **No "Chiedi ai modelli" button inside the spot screen**: the models answer by themselves whenever it is your turn
    with both cards known. The main menu's "Chiedi ai modelli (spot)" button,
    which opens the screen, stays.
  - **The client's pot excludes the current bets** (it reads empty preflop with
    the blinds still in front). So the check
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
  S >= 100, V >= 120) >= `PRESENT_FRACTION` 0.20. On the labelled crops
  present reads well above that and absent exactly 0 % (felt H 67-70); the rule
  needs no "present" example.
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
  run. Keep pokerlab's own window off the captured zones. Not verified on Windows;
  capture and the preview/save path were checked on Linux/X11.
- **Card recognition: `vision/recognize.py`, pattern matching, no network.**
  `python -m pokerlab.vision.recognize` scores it on `vision_data/crops/` by
  leave-one-image-out (each crop read by a recogniser built from all the others);
  `--image <png>` reads one crop. The client uses a **four-colour deck**
  (spades dark grey, hearts red, diamonds blue, clubs green) on a fixed layout:
  - **Slices**: board 5 equal fifths; hand **not** halved but cut at 0.47
    (`SLOT_EDGES`). The right hole card overlaps the left and its rank starts at
    ~83/170 px, so an exact half would leave a 2-px sliver in the left slice that
    is read as its rank.
  - **Empty slot**: white-pixel share below 0.06 (empty slots sit well below it,
    cards well above). The board stops at its first gap; a hand is 2 cards or none.
  - **Rank**: topmost white blob *starting in the left 40% of the slice*, plus
    blobs touching it on the same row (a ten's two digits), padded square to
    32x32 and matched by normalised correlation, 1-NN over every labelled glyph
    (left, right and board pooled -- the slight rotations are covered by
    examples, not modelled). The left-40% and touching rules come from a client
    animation (a player's reaction) drawn white over a 9h, higher than the rank,
    which the looser "whole row" rule merged into a ten.
  - **Suit**: the card's background colour by a **fixed rule** (`SUIT_RULES`): spades
    black, diamonds blue, clubs bright green, hearts red; the symbols on top are
    white. Read as the median non-white colour around the rank glyph. Measured
    bands (HSV): spades S<=38 V 57-64, hearts H 3-4, clubs H 59-60 S ~200,
    diamonds H 106-108; the felt sits at H 66-90 S 45-165, so clubs are told
    from it by hue *and* saturation. The rule's bands are much wider than
    measured and still do not overlap. A colour matching none reads as suit "?".
    Rank and suit are matched **separately**, so a card never collected whole is
    still read, and a suit needs no examples at all. (Closest rank confusions:
    5 vs 6, 3 vs 8.)
  - **No test section in the vision screen**: readings are checked in the spot
    screen, which reads every zone every 0.5 s. There is an **"Anteprima" per
    section** (`_preview` card zones only,
    `_preview_dealer`, `_preview_players`, all through `_show_preview`), which
    shows that section's zones side by side as captured now, captioned with the
    quick reading (gold share for the button, the seat state for a player).
  - **The spot screen reads the screen by itself every 0.5 s** (`SCREEN_POLL_MS`;
    no button): `gui/screen_reader.py` (no Tk) captures both zones and
    recognises them; `SpotFrame.apply_reading` puts them into the spot. A
    reading costs tens of ms (more the first, which builds the recogniser), so
    it runs on the Tk thread with no worker. Rules, each tested:
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

## Configuration file (`config.toml`, `pokerlab/config.py`)

One file of overrides for the fleet's parameters, in the project root (the NFS
export, so every machine reads the same one). The `DEFAULT_*` constants and their long comments stay exactly where they are as
the **defaults**; the file only overrides, and an absent key means the default.

- **Precedence: flag > `config.toml` > default in the code.** Implemented as
  `parser.set_defaults(...)` before `parse_args`, so argparse itself makes the flag
  win. **Five programs read it**: `poker-train`, `poker-loop`, `poker-elo`
  (`population_arena`), `benchmark_arena` and `poker-dashboard`. Each has a
  `build_parser()` and starts through `config.parse_with_config` (the three small
  ones through `config.resolve_cli`, which also handles the error exit and
  `--print-config`). Adding a sixth is `add_config_arguments` + `resolve_cli` + an
  entry in `rl/siblings.py`.
  - **One key, one meaning, in every program.** The file is a single flat namespace,
    so a name cannot mean two things. That is why the arenas call things what
    `poker-train` calls them (`--global-sample`, `--global-sessions`,
    `--global-lock-seconds`, `--global-trigger-size`,
    `--global-eliminate-fraction`, `--global-protect-percentile`, `--k-schedule`) and
    **not** `--sample`/`--hands`/..., where `--hands` is hands per training iteration
    in `poker-train` and the length of a session in an arena. What is specific to a
    tool has its own prefix: `arena_min_sessions`, `arena_max_sessions`,
    `arena_tolerance`, `arena_window`, `arena_checkpoint_sessions`, `elo_sessions`,
    `elo_coverage`, `dashboard_port`, `dashboard_refresh`. `iterations` is shared with
    the dashboard on purpose: it scales every progress bar, and has to equal the
    loop's. Pinned by
    `test_a_key_shared_with_an_arena_or_the_dashboard_means_the_same_everywhere`
    (same type and default in every parser that owns the name).
  - **The torch-free ones read the file `lenient`ly.** `poker-elo`, `benchmark_arena`
    and the dashboard must not import torch (`rl/siblings.py::TORCH_FREE`), and
    `poker-train`'s parser lives in a module that does, so they see only each other
    and *skip* a key nobody they can see knows, instead of refusing it. The typo is
    still caught, by the first `poker-train`/`poker-loop` that reads the file, which
    see everyone. Pinned by
    `test_the_torch_free_clis_read_the_shipped_config_without_importing_torch`.
  - **Not adopted, on purpose: `global_arena`'s shard entry point.** A shard is a
    subprocess that `play_sharded` launches with every value it needs as an explicit
    flag, so the file already governs it through its parent; it is the same argument
    as "Workers never read the file" below.
  - **Per-invocation switches stay flags**: `--dry-run`, `--restore`,
    `--host` and above all `--prune` (`config.LOCAL_ONLY`): a shared file must not be
    able to turn on deletion of checkpoints for the whole fleet.
  - **What the file now governs on the arenas is the rating scale**: `k_schedule`
    and `session_hands` (the session length the Elo scale is defined by) apply
    to `poker-elo` and `benchmark_arena` exactly as they do to a training run, so a
    change is a change of scale for everyone: do it with a fleet reset.
- **TOML, not a `.py`, deliberately**: a supervisor re-reads it every generation for
  weeks, and a data file cannot run inside a live process. Stdlib `tomllib`.
- **A key is the long flag name with underscores**; `[sections]` are decoration (keys are
  flattened, a name may appear once). Values are type-checked against the flag they
  stand for (`hands = 1.5` is an error; `lr = 1` is fine).
- **Strict on purpose.** A key no CLI knows is an error with a "did you mean" (a typo
  that silently did nothing is the failure to prevent). A key the *other* CLI owns is
  accepted and left alone (`siblings=`), since the file is shared. **Per-machine
  keys are rejected** (`config.LOCAL_ONLY` plus every `Path`-typed flag: machine,
  workers, device, seeds, every directory): the file is fleet-wide and those are not.
  They stay flags, and `run.sh` derives them.
- **The sweep's axes in the file are the *starting point*, not decoration.** `lr`,
  `hands`, ... under `[starting-point]` are read by `poker-loop` as the settings a
  worker with no parent starts from (each then multiplied by 0.8/1.0/1.2, like any
  other worker's); workers that inherit start from their parent's and never read
  them. A hand-run `poker-train` uses them as they are. `config.ignore` and the
  "ignorato" line are gone with the one thing that used them.
- **Re-read every generation** (`loop.refresh_args`, called at the top of each
  generation in `run_loop`, given `reload=` by `main`). A file that cannot be read
  (half saved, a typo) prints `config: ...; tengo i valori precedenti` and the loop
  carries on unchanged; a change prints `config riletto, cambiato: key: a -> b`. At
  startup, by contrast, a bad file is a hard exit (code 2). Save the file in one go
  (write elsewhere, rename), not in place on the shared volume. `GameConfig` is
  rebuilt per generation for the same reason.
- **Two settings are *text*, not numbers: `k_schedule` and `parent_tiers`.** Both are
  strings (`"0:16, 20:11, ..."`, `"10, 100, 1000, all"`) so the flag, the file, the
  worker's command line and the `settings` recorded in a model are one
  representation. Their argparse `type=` (`pool_registry.k_schedule_text`,
  `training_pool.parent_tiers_text`) validates and returns the canonical string, and
  `config._coerce` calls a flag's own `type=` on a file value, so the file is refused
  with the flag's message. `k_schedule` is read by the learner, `rate_against_benchmark`
  and the population merge (`run_population_sessions(k_schedule=)`); the supervisor
  forwards it to every worker. `parent_tiers` is read only by the supervisor
  (`inheritance_plan(tiers=)`). `poker-elo` and `benchmark_arena` read `k_schedule` too.
  **Changing `k_schedule` changes what every
  rating means** -- do it with a fleet reset.
- **Workers never read the file**: `launch_worker` passes `--config ""` (empty = no
  file) and every value as an explicit flag, resolved by the supervisor when the
  generation started. A worker reading it again would only add a way to die between
  two reads, and workers of one generation could disagree.
- **A run records what it resolved to**: `run_metadata` now carries
  `settings` (`config.resolved_settings`: every fleet-wide value, no paths or
  per-machine ones) inside the published model, because with a file a run's
  parameters are no longer recoverable from its command line. The worker log's
  `iperparametri:` line is unchanged.
- **`poker-loop --print-config` / `poker-train --print-config`** print every parameter as
  it resolves, as `key = value` ready to paste into the file, marking the ones the
  file set. Every value in the shipped file is active, so a default that moves in the
  code does *not* reach the fleet until the file is changed too.
- **`run.sh` must pin none of the keys `config.toml` sets** (game,
  `--iterations`, `--pool-models`). `POKER_ITERATIONS` and
  `POKER_POOL_MODELS` still override for a one-off run, and `status`/`watch` pass
  the former so the bars scale. Pinned by `test_run_sh_pins_nothing_the_config_file_sets`.
  The rule from "Every worker inherits weights" generalises: a flag only governs
  what it overrides.
- **Tests**: `tests/unit/test_config.py` (no torch: precedence, validation,
  siblings, which file) and the `config.toml` section at the end of
  `test_rl_loop.py` (the shipped file is valid for *both* CLIs, reload behaviour).
  The e2e loop test passes `--config ""` so it does not inherit production values.

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
parallelism just oversubscribes: on a 32-core box the default 32 threads burn
~18x more CPU than one thread for the same wall time and identical results (same
seed) — which is what makes it practical to run many independent training runs in
parallel across the cores instead of one run hogging all of them.

Throughput single-threaded was of the order of **50 hands/s** at a 6-handed table;
a hand costs about as much as it has seats, so the default mixture (4.35 seats on
average) should be faster. Measure it on the machine before planning a run's length.

```bash
mkdir -p checkpoints/logs
OMP_NUM_THREADS=1 nohup .venv/bin/poker-train \
    --iterations 200 --hands 512 --seed 42 \
    --eval-every 100 \
    > checkpoints/logs/run-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

Logs go under `checkpoints/logs/`, which is already covered by the
`checkpoints/` gitignore entry.

#### Parallel sweeps (many seeds at once)

"Use all the threads" means **N single-threaded runs, one per core** — not one
run with N torch threads, which is measurably slower (see above). One run pins
one core, so the box takes about as many concurrent runs as it has cores.

Memory, not cores, is what you have to measure first, and the naive number is
wrong: a `poker-train` process shows a large RSS, but most of that is torch's
shared pages, so the *incremental* cost of one more run is much smaller (measure
it by launching a few and differencing `free -m`). Always measure the increment
with a small probe batch before saturating — do not divide total RAM by RSS.

Each run in a sweep needs **its own `--checkpoint`, its own `--scratch-dir` and
its own `--archive-prefix`**. The checkpoint is obvious (they would overwrite each
other). The scratch dir and prefix matter because the run id is only
second-resolution: runs launched in the same second would share a scratch file
and would try to publish under the same name (the later one finds it taken and
its model is not published). Note that every run in a sweep draws its opponents from the *same* shared store and publishes into
it, so runs do see each other's published models; if a sweep must stay isolated
for a controlled comparison, give each its own `--models-dir` and `--global-dir`
(and `--no-global-elo`).

```bash
for s in $(seq 1 29); do
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 setsid nohup poker-train \
      --iterations 200 --hands 512 --seed $s \
      --eval-every 100 \
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
pytest -v                                    # full suite
pytest --cov=pokerlab                        # with coverage
ruff check .                                 # lint

poker-gui                                    # Tkinter desktop app

poker-train --iterations 200 --hands 512     # self-play PPO (needs the `rl` extra)
poker-train --resume --device cuda
# every run draws its own opponents from checkpoints/models/ and publishes its
# final model there when it ends
poker-train --pool-models 8
# a validation pass is --eval-sessions rated sessions of 1000 hands against the
# drawn opponents (their ratings are fixed reference points; only the learner's
# rating moves), and the count continues into the pass against the frozen anchors
poker-train --pool-models 20 --eval-every 100 --eval-sessions 10
poker-train --benchmark-sessions 500                # the pass that sets the published rating

poker-loop --workers 8                      # continuous training until stopped
poker-loop --status                          # per-worker table
poker-loop --status --watch 30               # live, redraws in place
poker-dashboard                              # browser view of every machine, http://127.0.0.1:8770
poker-dashboard --host 0.0.0.0 --iterations 1000   # reachable from the other machines
# --status/--watch need --iterations to match the running loop's, or every
# progress bar is scaled against the wrong target (the default is 1000 everywhere; `POKER_ITERATIONS` overrides in run.sh)
# the held-out models in checkpoints/benchmark/ are never seated in training
./run.sh stop                                # stop after the current generation (can be ~20 min)
./run.sh kill                                # stop everything NOW, whatever stage it is in
./run.sh kill-dry                            # ... list what that would kill, and kill nothing
# every worker also tries a cross-machine population Elo pass at the end of
# its run (on by default -- see "Continuous training loop" above)
poker-loop --workers 8 --no-global-elo   # opt out fleet-wide

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
and which the ratings in `checkpoints/global/` refer to.

## TODO — decided to postpone, not forgotten

Each item below was discussed and consciously left for later. None is a bug; the
current behaviour is coherent. Any figure quoted comes from a measurement that
must be repeated on the population it will be used against: models change, and so
do the numbers.

- **Opponent statistics (a HUD) so the model can adapt to different players —
  the input and the tracker exist, the training side does not.** VPIP, PFR, 3-bet,
  fold-to-3bet, steal, postflop aggression, c-bet, fold-to-c-bet and WTSD, fed to
  the network as part of the observation.
  - **Done: `engine/stats.py`, `Observation.seat_stats`, 100 slots per seat.** A
    `StatsTracker` (pure Python, in the engine, importing nothing from `players/` or
    `rl/`) keeps a `WINDOW`-hand (200) sliding window per `player_id`; `Table(...,
    stats_tracker=)` feeds it every finished hand and puts each seat's vector in the
    `Observation`s of the hands that follow. Every statistic is *events over
    opportunities*, counted once per hand at the player's first chance (AGG counts
    every postflop chip-in); an all-in for no more than the bet is a call (decided by
    `ActionRecord.amount`, the street total after the action). The vector is
    `USED_SLOTS` = 20 numbers (a "supplied" flag, hands in the window, then a rate and
    a log-scaled opportunity count per statistic, so zero opportunities reads as
    unknown) inside the `STAT_SLOTS` = 100 each seat has room for; **the other 80 are
    reserved and left at zero**. Every seat block is now 9 base features + 100, so
    **`OBS_DIM` is 1380 and `FEATURE_VERSION` 2**: every v1 checkpoint is rejected by
    `check_compatible`, a fleet reset (v2 has no published models to lose). Cost: the
    network goes from 0.78M to 1.24M parameters and a batch-1 forward from ~364 to
    ~438 us (+20%).
  - **Optional, seat by seat.** A seat with no entry in `seat_stats` -- no tracker, a
    player the tracker has not seen, the spot screen -- encodes as zeros in its stat
    slots, which is also what an all-unknown vector looks like; there is no flag and no
    special case. More than `STAT_SLOTS` numbers is a `ValueError` (silently truncating
    would drop some); values are clipped to [0, 1]. **Changing what any slot means is a
    `FEATURE_VERSION` bump.** Pinned by `tests/unit/test_stats.py` and the statistics
    section at the end of `test_rl_features.py`.
  - **Done: how each model plays is measured during training and shown in the
    dashboard.** `SelfPlayCollector` pools every seat the learner took and counts the
    same nine statistics over its last `STYLE_WINDOW` (20,000) seat-hands (about nine
    thousand hands at the default mixture; `TableBank.last_hand` hands it the finished
    hand, `analyse_hand` does the counting). `poker-train` prints, on the first
    iteration and every tenth, `stile (N mani): [vpip e/o | pfr e/o | ...]` -- raw
    counts, closed by a bracket so a line cut inside the last number cannot parse as a
    wrong reading (`rl/style_log.py`, format and parser in one file). `monitor` reads it
    into `WorkerHistory.style_rate/style_latest/style_hands` and a worker's expanded
    panel draws two charts (preflop, postflop) with the last rate and the sample behind
    each in the note. The final style also rides in the published model's metadata
    (`style`, `style_hands`). This describes the *model*, not the opponents it faced: it
    is the learner's own VPIP/PFR/..., which is what tells a collapsed or lopsided
    policy from a healthy one at a glance. It is not the tracker's per-opponent window.
  - **NOT done: nothing in training supplies them.** `SelfPlayCollector` builds its
    `Table`s with no tracker, so the model sees 900 zeros of statistics and has no way to
    learn to read them. What it needs is below, and it is a change to the training
    distribution, not a plumbing job.
  - **Opponents need an identity in training, and today they have none.** Stacks
    reset after every hand and `SeatProxy` swaps who sits where between hands, so
    "the VPIP of the player on my left" means nothing. Opponents must stay the
    same for a block of at least the statistics window (~200 hands); the
    1,000-hand rated session already has that shape, the training collector does
    not.
  - **Where to compute them: at the `Table`, not in the player.** An
    `RLAgentPlayer` only sees the action history when it is its turn, so once it
    folds preflop it stops seeing what the others do and VPIP/3-bet would be
    lost exactly there. A tracker fed by `Table.on_action_applied` (which exists)
    keeps a per-seat window and hands the numbers to the `Observation` as an
    optional field. The engine keeps not importing from `players/`.
  - **Definitions, fixed once and tested.** Percentages are counted over the
    hands where the action was *possible* (opportunities), not over all hands: a
    3-bet% over every hand is misleading. Pure Python, no torch, in the ordinary
    test suite.
  - **Every statistic carries its sample size, over a window of at most ~200
    hands.** A VPIP of 60% over 8 hands and over 200 hands are different facts,
    and a model fed the bare percentage would trust both equally. In the
    observation: an opportunity count (normalised) next to each stat, or the stat
    shrunk toward a prior in proportion to the count; zero samples then reads as
    "unknown" with no special case. Keep a sliding window, not a cumulative mean:
    in a real room players change often, and a long history describes someone who
    is no longer there. The metric of a rollout is noisy over few hands, so
    anything that reads it needs a window and must know how full it is, as
    `REWARD_WINDOW_HANDS` and its partial-window marker do for `train/100`.
  - **Measure before building the training side.** A model can only adapt if the
    opponents actually differ. Write the metrics module first, run it over the
    hands the existing pool models play, and look at the spread of VPIP/PFR/3-bet
    across models. If it is small, adaptation has nothing to learn from and
    opponent diversity (next item) becomes a prerequisite.
  - **The spot screen has no history of previous hands**, so its statistics start
    from "unknown" (zero samples) until a way to supply them exists.
  - **It changes `OBS_DIM`** (new input features), so it needs the versioned
    encoder or a population reset — the same decision as "Observation v3" below.
- **Style constraints on the models, to keep the population's strategies diverse
  (idea, not yet designed).** Every model is trained toward the same objective
  (bb won against the drawn field) and the fleet's selection pressure is a single
  number, the Elo, so the population tends to collapse onto one style. The idea:
  impose constraints on playing-style metrics during training, so that different
  lineages are trained to different profiles and the pool keeps genuinely
  different opponents instead of near-copies. `HP_AXES` diversifies only the
  *settings*, not the resulting *behaviour*.
  - **Things to decide first.** (1) *How to constrain*: a penalty or Lagrangian
    term in the PPO loss on the measured metric vs. a target band, a reward
    bonus, or masking/biasing action bins — the first is the cleanest, the last
    repeats the "masked, never clamped" argument in reverse. (2) *Where the
    target lives*: a per-run hyperparameter (a new axis of `HP_AXES`,
    inherited and perturbed like the others) or a fixed list of named profiles.
    (3) *How a style interacts with the Elo*: a constrained model will rate below
    an unconstrained one, so `pick_parents` (top 100 by rating) would discard
    every constrained lineage — selection would have to be per-style, or rate
    relative to the style's own niche (quality-diversity in the MAP-Elites
    sense). (4) *Measuring the metrics*: same definitions and windows as the HUD
    item above.
  - Constraining only the loss changes no shape and keeps every stored model
    loadable; a style target given to the network as an input feature changes
    `OBS_DIM`.
- **Observation v3: encode the action *sequence*, not per-street aggregates.**
  Postponed from version 2 in favour of the opponent statistics above. Today
  `_history_aggregates` compresses the whole betting history into 40 numbers — 4
  streets x 10 counters — and what it throws away is exactly what poker is played
  on:
  - **The order inside a street.** "check, bet, raise" and "bet, raise, check"
    produce the identical vector. Initiative, who opened, who reacted to whom —
    all gone.
  - **Who did what.** Only the *last* aggressor of each street is identified (by
    relative seat) plus the agent's own participation.
  - **Individual bet sizes.** Only the street's *maximum*, relative to the pot.
  - Per-opponent *amounts* do survive, in the 81 seat slots (`current_bet / pot`
    and `committed / pot`); it is the action *types* and their order that are
    lost.

  Why it matters: bet-sizing patterns are the signature of what the agent most
  obviously cannot do. Small on the flop then large on the turn is value building;
  large on the flop then giving up on the turn is an abandoned bluff. To the
  current encoder those are nearly the same vector. The information is all there
  already: `Observation.action_history` is a list of `ActionRecord` carrying
  `street`, `seat`, `player_id`, `action_type`, `amount`, `stack_before/after`,
  `pot_before`. **No engine change is needed** — only `features.py`.

  **The prerequisite.** `check_compatible` (`rl/ppo.py`) rejects a checkpoint
  whose `obs_dim` *or* `feature_version` differs, so any encoder change makes
  every stored model, the frozen anchors and the Elo scale built on them
  unloadable at once. That is a population reset, not a migration. So the work is
  in two parts, in this order:
  - **Part 1, a versioned encoder.** `encode_observation` takes a `version`; the
    current code becomes the `version=1` branch, untouched. `RLAgentPlayer` reads
    the version off the checkpoint it wraps and encodes at that version.
    `check_compatible` accepts any version this build can serve rather than only
    the newest. Then models of different versions sit at the same table, each fed
    its own vector: the pool stays full, the anchors stay valid, the ratings keep
    meaning something. Small, no behaviour change, directly testable. Worth doing
    on its own merits.
  - **Part 2, the new features.** Keep the 40 aggregates and append a flat
    positional encoding of the **last 20 actions**, 24 features each: action type
    one-hot (7), actor's seat relative to me (9), street one-hot (4),
    `amount / pot_before` (1), `log(amount / big_blind)` (1), was-it-me (1),
    slot-occupied/padding (1). 480 new features, so `OBS_DIM` 1380 → 1860. The
    window of 20 was sized from the distribution of actions per hand (median 12,
    max ~25 including blinds): re-measure it if the table size or stack depth
    changes. When a hand truncates it keeps the most recent actions.
  - **Fix how each new feature is scaled, in the same change as the version.**
    Every feature in `features.py` is a bounded ratio (`_clip01`, with chip
    amounts divided by the big blind and passed through `log1p` over a fixed
    scale), and the network has no input normalisation beyond the `LayerNorm`
    after each `Linear`. The per-action features must follow the same convention:
    `amount / pot_before` and `log(amount / big_blind)` clipped to [0, 1] with a
    stated scale, the one-hots left as they are. A feature left unbounded would
    dominate the first `Linear` and change what `FEATURE_VERSION` means without
    anyone noticing, so the scale is part of the version's definition.
  - **Why not a GRU over `ActionRecord`s**: it also changes `PokerActorCritic`'s
    architecture rather than just the features, multiplying the work and the
    risk, for information the flat encoding already exposes. A recurrent encoder
    only pays off once hands routinely exceed the window.
- **Re-derive the K staircase and the session variance on the table mixture.**
  `DEFAULT_K_SCHEDULE` and the 238-point session sd behind it were measured at one
  6-handed table with identical 100 bb stacks. A session now has a drawn size
  (heads-up is one Bernoulli, 9-handed is eight correlated ones) and per-seat
  stacks that change how many chips a seat can move, so the real `V` is a mixture
  and almost surely larger. Start from the current staircase and re-measure `V`
  with `duel_power --mode field` per table size once a trained population exists
  (`duel_power` itself still plays one fixed table: give it the mixture first),
  then re-derive the numbers from the formula rather than editing them.
- **Check that the value target is balanced across table sizes and stack depths.**
  `reward_scale` is one constant (`big_blind / starting_stack` = 1/100). The spread
  of a hand's result still changes with the table size (a 9-handed all-in can win
  eight stacks), so log the value loss and the target spread per size from the
  rollouts of a real run. If they differ by more than ~2x, the options are a scale
  per table size from the measurement (config, no checkpoint state) or an adaptive
  return normalisation (running mean/std, or PopArt), which has to be saved in the
  checkpoint or `--resume` and inheritance start on a different scale.
  - **The measurement exists now** (`rl/value_diagnostics.py`, pure Python):
    `poker-train` prints two `valore per tavolo:` / `valore per stack (bb):` lines
    on the first iteration and every `VALUE_DIAGNOSTICS_EVERY` (10), each group with
    the sd of its target (the GAE return, in the critic's unit), the explained
    variance of the values the policy collected with, and its decisions, plus the
    spread (widest sd over the narrowest, groups under `MIN_DECISIONS` left out).
    Stack groups are by *effective* stack (`HandTrajectory.effective_stack_bb`: own
    stack or deepest opponent, whichever is smaller) at `<10`/`10-30`/`30+` bb. The
    lines start with `valore` so the status parsers, which read only `iter ` lines,
    are unaffected. The format and its parser (`parse_value_line`) live in the same
    module, so they cannot drift. **Not yet read off a real run.**
  - **The dashboard shows it** in a worker's expanded panel, four charts after the
    training curves: target sd by stack, target sd by table size, explained variance
    by stack, and the two spreads against a 2x line. `monitor.WorkerHistory` carries
    them as sparse `[iteration, value]` pairs (a reading every 10 iterations, anchored
    to the `iter` line printed just before it), empty for a log that predates the
    lines. `--status` does not show it.
  - **First indication, not an answer.** A from-scratch self-play study (4 constants
    x 3 seeds, 300 iterations, duplicate-deck league, scratch script not kept) found
    no variant stronger than the rest: the spread between seeds of one variant
    (up to ~12 bb/100) is larger than between variants. The target sd with the 1/100
    scale is ~8x apart across stack groups and ~2-3x across sizes, so the ~2x
    threshold *is* exceeded, yet a constant 25x apart did not move strength. A
    per-hand scale (`reward / effective stack`) was the only one whose critic had
    non-negative explained variance on every stack group, and it changes what the
    objective weighs, so it was not adopted. Settling it needs ~10 seeds per variant,
    or the `valore` lines from production runs.
- **Make the Elo update take the bb difference into account.** `pairwise_elo_delta`
  reads only the *sign* of each pair's chip delta, so winning by one chip counts as
  much as winning by 100 bb. Decided to do, not yet done. The design has to avoid
  what two earlier variants found when simulated against a synthetic population
  whose true skill is known:
  - A continuous score `0.5 + 0.5 * clip(margin / S, -1, 1)` gained nothing under
    heavy-tailed session noise (one session in ten dominated by a monster pot) and
    a loose `S` hurt it.
  - Multiplying the delta by `|chip delta| / starting_stack` flips the sign unless
    `abs()` is taken and capped, and with the cap its whole gain is a lower
    effective K in disguise (identical to scaling K by the mean weight).
  - `|chip delta|` measures how *eventful* a session was, not how *well* the model
    played; the direction is already in the sign, so an unsigned magnitude adds
    variance, not signal. The new rule has to be judged by whether it orders a
    population of *known* skill better than the sign rule, not by looking
    sensible.
  - K dominates in simulation: with the sign rule unchanged, lowering the whole
    schedule moved the ordering far more than either scoring change. The K
    staircase is now derived rather than tuned (see its section), so any new rule
    means re-deriving it.
  - Whatever is chosen touches the scale everything rests on (the Elo scale is
    defined by how often a 1,000-hand session picks the stronger model, see
    `--session-hands`), so the K staircase and the anchors' ratings would need
    re-deriving, and earlier readings are not comparable.
  - **First, study how many bb/100 a model should earn for a given Elo gap.**
    Nobody has measured the exchange rate between rating points and chip winnings,
    and the update needs it. Play pairs of models spanning a range of Elo gaps with
    `rl/duel_power.py` (duplicate decks, which is what resolves a few bb/100),
    enough pairs per gap to fit a relation, and report the edge with its interval
    per gap. It has to settle whether the relation is roughly linear or saturates,
    how much it depends on the field (`--mode field` vs the 3v3 duel), and whether
    it holds at the top. **The tool exists: `rl/elo_bb_grid.py`**
    (`python -m pokerlab.rl.elo_bb_grid --hands 10000 --jobs 20`, results in
    `checkpoints/studies/elo_bb_grid.json`, resumable). Every benchmark model
    against every other and against itself, 3v3 at a 6-seat table, duplicate
    decks, one cell per unordered pair (the grid is antisymmetric); the diagonal is
    the noise floor; it prints a through-the-origin fit of bb/100 on the Elo gap.
    **A cell is the row model's own bb/100 per seat, which is half the head-to-head
    margin `duel_power` reports.** The cost grows with the square of the anchor
    count, so `--max-models` exists for a first look.
- **Bigger networks — blocked on one thing, deferred to a new version of the
  project.** *Partly done: the shape is now configurable (`[network]` in `config.toml`,
  see the RL section) and a mismatched parent no longer crashes a worker, it starts
  from scratch. What is below was the analysis before that; the open part is a
  `pick_parents` that filters by shape, so that a worker does not lose its weights.*
  `hidden`/`num_layers` were 512/3 since day one and unused.
  - **Almost everything already supports mixed shapes.**
    `build_model_from_checkpoint` rebuilds at the shape the checkpoint records,
    and *every* consumer goes through it — the training pool
    (`registry_opponents`), the bb/100 benchmark, the population pass,
    `benchmark_arena`, the GUI, `poker-play`. Nothing batches across models, so
    different sizes coexist at one table.
  - **Exactly one thing breaks: inheritance.** `load_checkpoint` is called in a
    single place in the whole project, the `--resume` path (`train.py`), and it
    loads into an *already built* model, so `load_state_dict` raises a bare
    `RuntimeError` on a shape mismatch. It is not an edge case: `pick_parents`
    draws parents by *rating*, not by shape, so a differently-sized worker would
    almost always draw a mismatched parent and die at startup. Compounding it,
    `check_compatible` does not look at the shape at all (only `obs_dim`,
    `action_dim`, `feature_version`), so the failure arrives as a raw torch error
    rather than the project's own `IncompatibleCheckpointError`.
  - **Two possible fixes, and they are not equivalent.** Building the learner at
    the *parent's* shape on `--resume` is the smaller change, but then `--hidden`
    becomes a suggestion that inheritance silently overrides. Filtering
    `pick_parents` by shape keeps each lineage at its own size, which is also what
    makes the experiment readable — a 768 lineage with a 512 ancestor in it
    measures nothing. Either way `check_compatible` should learn to say so clearly.
  - **The costs** (forward pass at batch 1, `OMP_NUM_THREADS=1`, for the
    480-input network (1380 now: ~438 us at 512x3); the hands/s compose that with the two thirds of a hand that
    is pure Python and does not change — re-measure before relying on them):

    | shape | parameters | file | us/forward | est. hands/s |
    |---|---|---|---|---|
    | 256x3 | 259k | 1.0 MB | 370 | 77.9 |
    | **512x3 (today)** | **781k** | **3.1 MB** | **473** | **71.1** |
    | 512x4 | 1.04M | 4.2 MB | 609 | 63.8 |
    | 768x3 | 1.56M | 6.3 MB | 801 | 55.7 |
    | 768x4 | 2.16M | 8.6 MB | 1,491 | 38.2 |
    | 1024x3 | 2.61M | 10.4 MB | 2,018 | 30.8 |

    Speed falls faster than linearly: 768x3 was the sweet spot (2.2x the
    parameters for 22% fewer hands/s), while 768x4 halves throughput and 1024x3
    does worse, which doubles the cost of every rated session, every benchmark and
    every population pass.
  - **Two operational consequences to size before trying it.** The population
    pass loads ~55 models per worker, so bigger models multiply the transient
    allocation when many workers arrive at that phase together. And bigger models
    grow the store several times faster while **the pruning trigger counts files,
    not bytes** — at a larger shape that threshold stops being the right measure.
  - Deferred: no compatibility shims and no code written only to paper over a
    mismatch, so this belongs to a new version of the project rather than to the
    running fleet.
- **Study how the network evaluates the state, to size it (decided to do, not yet
  done).** The network size has never been chosen from evidence about what the
  network can actually *see* in a state. The study should find out at what
  capacity, and at what point in training, the networks start to recognise the
  structures that decide a hand: straights, flushes, full houses and the other made
  hands and draws, and how strong the hand is relative to the board.
  - **What to measure.** Probe the value head and the policy (e.g. linear probes on
    the trunk's activations, or the value/action response to controlled states) for
    each concept: pair/two pair/trips, straight and straight draw (the wheel
    included), flush and flush draw, full house, quads, hand rank percentile on the
    board. For each one, the capacity (a ladder of `hidden`/`num_layers`) at which
    it becomes decodable, and the training iteration at which it appears within a
    run.
  - **Why it matters.** The card input is 6 binary 4x13 planes, so a straight is a
    pattern across ranks and a flush a pattern across suits; the first `Linear` has
    to build those from raw planes. If a concept only becomes readable above some
    size, a 512x3 network may be capped below it, and a bigger one would be worth
    its cost. If every concept is already read at 256, the current size is
    wasteful.
  - **Feeds other items:** "Bigger networks" (which shape to try), the encoder
    work (whether explicit hand-strength features would help or the network already
    derives them) and the suit-isomorphic canonicalisation idea in "Where to
    extend".

## Where to extend each future section

- **RL**: implemented end to end (see the "RL" section above). Natural next
  steps, roughly in order of value: multi-process rollout collection (N `Table`s in
  N workers — note `make_policy_fn` returns a closure, which is not picklable, so
  this needs `fork` or a module-level callable); duplicate/mirror deals (replay a
  seeded deck with rotated seats and average) for variance reduction;
  suit-isomorphic canonicalisation of the card planes for a ~4x sample-efficiency
  win; and the opponent statistics and later the per-action history encoder, both
  specified in the TODO section.
  **Trained models are seatable by path**: `model:<path>` is the only bot spec
  `build_players`/`validate_bot_key` understand, so a bad path fails only for
  whoever asked for it. `discover_trained_models()` (pure Python — `pool_registry`
  carries no torch) ranks the checkpoints found across *every* machine's pool,
  since the best model is usually not on the host you are sitting at; torch is
  imported only inside `make_model_bot`, so `poker-play` and the GUI still run
  without the `rl` extra until someone actually picks a model. `build_players`
  takes an optional `game: GameConfig` for this: `RLAgentPlayer` normalises its
  features by `big_blind`/`starting_stack`, which an `Observation` deliberately
  does not carry, so seating a model without it raises rather than guessing.
- **Vision**: see the "Vision (`vision/`)" section above for what exists. The
  integration points are unchanged: an `Observation`-compatible read or a
  `HandHistory`-style record, no engine changes.
- **GUI**: implemented — see the "GUI" section above. Remaining rough edges are
  documented inline: there's no mid-hand stop.

## Testing strategy already in place

**Run only the tests a change can affect.** The full suite takes ~3 minutes, almost
all of it the `rl` area and the real-subprocess e2e. Every test file carries an
area marker taken from its name (`tests/conftest.py`: `test_gui_*` -> `gui`,
`test_vision_*` -> `vision`, `test_rl_*` -> `rl`, `test_config*` -> `config`,
everything else `engine`; `slow` marks the e2e loop test). Use
`python tests/affected.py <changed files> [-- pytest args]` (no arguments: the files
`git diff HEAD` lists), or `pytest -m gui`, `pytest -m "rl and not slow"`, a single
file. Engine/cards/evaluator/players changes fan out to engine+rl+gui; GUI-only
changes run nothing else. The full suite is for before a commit, after a change to
`tests/conftest.py`/`support.py`, or when `affected.py` says it does not recognise
the change. Add a new area by naming the file with its prefix, not by decorating.


Ordered by correctness risk (highest first — see `tests/unit/`):
`test_evaluator.py` (hand ranking), `test_side_pots.py` (side-pot math and
uncalled-bet refund), `test_betting_legal_actions.py` (legal actions,
min-raise/short-all-in), `test_blinds_and_stacks.py` (button rotation, short
blinds, busted players), `test_hand_history.py` (JSONL round-trip).
`tests/integration/test_full_hand_flow.py` fuzzes 2-9 players across 5 seeds
each with a chip-conservation invariant on every single hand — this is the
test most likely to catch a subtle new bug, run it after any engine change.
These engine-level tests fill their non-learner seats with the minimal
`make_random_legal_bot`/`make_always_call_bot` in `tests/support.py` rather than
any product-facing bot.

For the RL section the equivalent gate is
`tests/integration/test_rl_env_flow.py` (same 2-9 × 5-seed fuzz, asserting the
policy's action bins never produce an `IllegalActionError`); run it after any
change to `rl/action_space.py` or `rl/features.py`. `test_rl_policy.py` is the
only test file that needs the `rl` extra and skips itself cleanly without it.
`test_rl_global_arena.py` covers the cross-machine population Elo/pruning
system (see "Continuous training loop" above); its discovery/
sampling/lock/promotion/member-store tests need no torch, only the
`play_global_sessions`/`run_population_sessions` tests do.
