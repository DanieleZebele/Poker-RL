# pokerlab — notes for future Claude Code sessions

## What this project is

A No-Limit Texas Hold'em engine built as the foundation for a future
reinforcement-learning poker project. Four planned sections:

1. **Game engine** (`src/pokerlab/engine/`, `cards/`, `evaluator/`) — **implemented and tested.**
   Configurable players (2-9), stacks, blinds; full betting logic including
   side pots and the min-raise/short-all-in edge case; JSONL hand history.
2. **Players** (`src/pokerlab/players/`) — **implemented and tested.**
   `ManualPlayer` (terminal input) and `ScriptedBot`, with a catalog of five
   named bot personalities ranked by difficulty 1 (weakest) to 5 (strongest):
   Random, CallingStation, Maniac, Rock, Shark. See "Bot catalog" below.
3. **RL training** (`src/pokerlab/rl/`, `players/rl_agent.py`) —
   **implemented and tested.** Observation encoding, discrete action space +
   legal-action mask, `RLAgentPlayer`, trajectory collection with GAE, a
   PyTorch actor-critic, self-play PPO with an opponent pool, checkpointing,
   evaluation in bb/100, a Gym-shaped `TablePokerEnv`, and the `poker-train`
   CLI. Requires the `rl` extra. See "RL" below.
4. **Live table vision** (`src/pokerlab/vision/`) — **stub only.** Same
   treatment as above, in `reader.py`; requires the `vision` extra
   (opencv-python/numpy/mss).
5. **GUI for live testing** (`src/pokerlab/gui/`, `players/gui.py`) —
   **implemented.** A Tkinter desktop app (`poker-gui`): a setup screen
   (players/stack/blinds/hands/bot selection, reusing the same bot catalog
   and `custom:` spec as the CLI) and a table screen (seats, board, pot,
   hole cards, action buttons, hand-by-hand log). See "GUI" below.

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

## Bot catalog (`players/scripted.py`)

All named bots share one implementation, `make_heuristic_bot`, parametrized
along four axes: `tightness` (how strong a hand must be to play at all),
`aggression` (how much of the playable range is bet/raised rather than
checked/called), `bluff_frequency` (independent per-decision chance of
betting/raising anyway despite a weak hand), and `size_variance` (random
jitter on bet sizing, so the same hand doesn't always produce the same bet).
`tightness` and `bluff_frequency` are further scaled per-decision by
`_active_opponent_count` (how many other seats are still live, i.e. not
folded, in the current hand): more live opponents tightens the continue
threshold and shrinks the bluff rate (a bluff only works if everyone folds),
fewer loosens both — see `_field_adjusted_tightness` /
`_field_adjusted_bluff_frequency` and `_BASELINE_OPPONENTS`. Since this
count is recomputed from the live `Observation` on every decision, a bot
naturally loosens up as the hand progresses and players fold, without any
extra state to track.
`BOT_CATALOG` in that file lists the five presets with a `key`, a difficulty
1-5, a human-readable description, and a uniform-signature factory
(`(player_id, name, rng) -> Player`). `list_bot_profiles()` returns them
sorted weakest-to-strongest; `get_bot_profile(key)` looks one up and raises
`KeyError` with the valid key list on a typo. The difficulty ranking is a
subjective, documented judgment call (not a solved/exploitability-proven
ordering) — see the docstrings on `make_loose_aggressive_bot` (Maniac),
`make_tight_passive_bot` (Rock), and `make_tight_aggressive_bot` (Shark) for
the reasoning behind each one's placement. `Random` and `CallingStation`
(difficulty 1-2) stay as their original standalone functions
(`make_random_legal_bot`, `make_always_call_bot`) rather than going through
`make_heuristic_bot`, since neither is really a point on the same
tightness/aggression spectrum. `poker-play --list-bots` prints the catalog;
`poker-play --bots key1,key2,...` picks which bots fill the non-human seats
(cycled if there are more seats than keys). A slot can also be a one-off
custom bot instead of a catalog key, via
`custom:tightness=0.2;aggression=0.8;bluff_frequency=0.3;size_variance=0.5`
(any axis left out gets a moderate default) — parsed and validated by
`parse_custom_bot_spec` in `cli/play.py`. This registry is also the
natural place to plug in future RL agents for curriculum-style training
(e.g. train against Random first, then Rock, then Shark) without needing a
parallel selection mechanism.

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
  not sequenced — a GRU over `ActionRecord`s is the natural v2.
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
  hand naturally tightens and loosens as players fold — the same signal the
  scripted bots get from `_active_opponent_count`.
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
- **`OpponentPool` fills the seats the learner is not in**, sampling **uniformly**
  between all five `BOT_CATALOG` bots and past snapshots of the policy itself
  (`SelfPlayTrainer.snapshot()`, capped by `max_snapshots`). Note this is *not*
  a difficulty-ordered curriculum — Shark shows up from iteration 1. The mix
  does shift on its own as snapshots accumulate: measured over 500 hands 6-max,
  scripted bots hold ~42% of seats at iteration 1 and ~19% once five snapshots
  are in the pool, with the learner itself at ~58% throughout (one seat is
  forced, the rest are coin flips at `opponent_probability`). If a real
  curriculum is ever wanted, it belongs in `SelfPlayTrainer`, by rebuilding the
  pool per phase. Seats are swapped between hands
  through `_SeatProxy` rather than by rebuilding the `Table`, which would reset
  the button and stacks. Exactly one seat is always the learner, so a hand can
  never yield zero training data.
- **Two kinds of saved model, and the distinction matters.**
  `checkpoints/agent.pt` (`--checkpoint`) is the *live* one: weights plus
  optimizer state, **overwritten every save**, and what `--resume` reads.
  `checkpoints/pool/agent-<runid>-iter<N>.pt` (`--pool-dir`, `--archive-every`)
  are *permanent* copies, never overwritten — the run id is a start-time
  timestamp so concurrent or repeated runs cannot collide. Both directories are
  gitignored.
- **`archived_opponents()` seats previously trained agents in later runs**
  (`--pool-models`, default 5, newest first). Without it the pool resets to the
  scripted catalog on every run and the in-memory snapshots die with the
  process, so each new agent relearns from scratch against the same five bots.
  A file that fails to load is skipped with a printed reason, never fatal: the
  directory is user-owned and one bad file must not stop training.
- **`FEATURE_VERSION` (in `features.py`) must be bumped whenever the *meaning*
  of a feature changes, even when `OBS_DIM` does not.** The position fix
  reordered what slot 340 means while keeping the vector exactly 480 long, so a
  dimension check alone would have happily loaded a model that reads the wrong
  thing from every slot. Checkpoints record it and `check_compatible` rejects a
  mismatch. Checkpoints also record `hidden`/`num_layers`, so
  `build_model_from_checkpoint` can rebuild an archived agent at whatever shape
  it was trained with.
- **`SelfPlayTrainer.evaluate(bot_key)` measures bb/100 from stack deltas, not
  from collected trajectories.** Hands the learner wins without ever acting
  (everyone folds to its blind) produce no trajectory, and those are
  systematically *winning* hands — averaging trajectories only would bias the
  win rate downwards.
- **But the eval number is dominated by noise, and more hands barely help.**
  Measured on one fixed checkpoint, varying only the eval seed: spread of
  ~570 bb/100 over 300 hands, ~363 over 1000, ~351 over 5000. Going 300 → 5000
  should cut the spread ~4x if results were well-behaved; it cut it 1.6x,
  because per-hand outcomes are heavy-tailed (one hand can swing a whole stack)
  so the mean converges far slower than 1/sqrt(n). **Do not read progress into
  iteration-to-iteration eval changes** — in a real 40-iteration run the prints
  were -267, +1376, +363, +342, which is one number and noise. The real fix is
  variance reduction (duplicate deals: replay a seeded deck with rotated seats
  and average) is implemented as `evaluate_duplicate()` — **and measurably did
  not help**: at equal compute the spread was 259 bb/100 plain vs 333 duplicate.
  Duplicate cancellation assumes the hand plays out roughly the same whoever
  sits in a seat, which fails badly when the learner's strategy is nothing like
  the opponents'. Keep it for when the agent is less degenerate; do not trust it
  now.
- **The real cause of the noise is the agent's own strategy, not the metric.**
  After 40 iterations the policy shoves 45% of the time and folds 28%, with
  almost no calling (4.8%). The resulting per-hand distribution has median
  **-6 bb** but mean **+5 bb** and standard deviation **144 bb**: 60% of hands
  move more than 50 bb and account for 98% of all chip movement. It loses small
  constantly and occasionally stacks the table. No averaging technique rescues a
  measurement of a lottery ticket. Sampled vs greedy action selection made no
  difference (std 397 vs 398), confirming the policy's stochasticity is not the
  culprit. **Shoving is genuinely near-optimal against this pool** — the
  scripted bots scale their tightness *up* with the opponent count, so five of
  them at a 6-max table fold almost everything and the agent learns to steal
  relentlessly. Fixing the eval means fixing the training signal first: a less
  fold-happy opponent pool, a higher entropy bonus, or deeper stacks relative to
  the blinds so shoving stops dominating.
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

## GUI (`gui/app.py`, `players/gui.py`)

Tkinter was chosen over a local web app specifically to avoid adding a new
dependency (Flask, etc.) — it ships with Python. Architecture:

- **Setup screen bot builder**: `SetupFrame` no longer has a free-text bot
  field or a "number of players" field. Instead `self.bot_specs: list[dict]`
  (each `{"key": <catalog key>}` or `{"key": "custom", "params": {...}}`)
  drives a row of rectangles (one per configured bot, each with a "-" to
  remove) plus a trailing "+" that opens `AddBotDialog` (a modal
  `tk.Toplevel`) to pick a catalog bot or "Custom" with four parameter
  sliders. `num_players` is now *derived* (`human_seats + len(bot_specs)`,
  capped by `GameConfig`'s own 2-9 validation) rather than typed
  separately — keeps the visual builder as the single source of truth.
  `_bot_spec_to_key_string` turns a spec back into the exact string
  `BOT_CATALOG` keys / `custom:...` syntax that `build_players` and
  `validate_bot_key` already understood, so neither needed to change.
  `CUSTOM_PARAM_DEFAULTS` was promoted from a private name in `cli/play.py`
  to a public one specifically so the dialog's sliders could reuse the same
  defaults instead of duplicating them.
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
- **`SteppingPlayer`** (`players/gui.py`) wraps every non-human `Player` so
  the GUI can watch bot decisions live instead of a whole hand resolving
  instantly: it reports every action as an "action_taken" `GuiEvent`
  (Table/engine never see this -- it's a pure wrapper around `act()`), and
  -- only while a shared `step_mode_state["on"]` dict says so -- blocks
  after the action on a shared `step_gate: queue.Queue[None]` until the
  GUI's "Avanti" button pushes a release. `GuiPlayer.act()` also emits an
  "action_taken" event after the human's own choice (unblocked -- they
  already "stepped" by clicking), so the log covers every player uniformly.
  Toggling step mode off pushes one release onto `step_gate` (in case a bot
  is mid-block right then) and toggling it back on drains any leftover
  release first -- without that pairing, either a bot could stay stuck
  until one extra "Avanti" click, or a stale release could silently skip
  the next real pause. Step mode can also be preset from `SetupFrame`
  before a session starts (`start_session(..., step_mode=...)`), since
  toggling it only after `start_session` already returned can lose the
  race against instant bot-only hands finishing before the checkbox click
  even lands (learned the hard way while testing this).
- **Opponent-card "spy" toggle**: `Table` accepts an optional
  `on_hand_started` callback (fired once per hand, right after blinds are
  posted, with `{hand_id, button_seat, sb_seat, bb_seat, small_blind,
  big_blind, hole_cards}`) purely as a spectator/debug hook -- nothing in
  `Player`/`Observation` was touched to add this, since leaking opponents'
  hole cards into the normal per-player interface would be a real design
  flaw for the eventual RL section. The GUI wires this hook to cache
  `hole_cards` and a checkbox flips whether `TableFrame._draw_seat_cards`
  draws opponents' actual cards or a face-down back.
- **Action-by-action + showdown log**: `_log_hand_start` (from
  `on_hand_started`) prints the blind postings, `_log_action` (from every
  "action_taken" event, human included) prints one line per decision via
  `_describe_action` -- which computes CALL/ALL_IN amounts from the
  `Observation` rather than `Action.amount`, since the engine deliberately
  leaves that field meaningless for those two action types (see
  `Action`'s docstring) -- and `_format_hand_summary` (at "hand_complete")
  prints a showdown section with revealed hole cards whenever 2+ seats
  didn't fold, plus "Pot vinto da: ..." and final stacks. One known,
  accepted imprecision: an "action_taken" event's `Observation` is a
  snapshot from *before* that action was applied, so the table view is
  always one action behind the log text for whichever action just fired --
  it self-corrects on the very next event. Fixing this precisely would need
  a HandState-level snapshot hook inside `Table`'s betting loop, not just
  wrapping `Player.act()`; not done, since the lag is a non-issue in
  practice (each event's log text is accurate immediately; the visual
  catch-up happens one tick later, imperceptible except in step mode where
  it just means "the board reflects the previous click's action").
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
  same catalog keys and `custom:...` spec as `poker-play --bots`, with zero
  duplicated parsing logic.
- **Known environment quirk, worked around in code**: this project's own
  Python install has `TCL_LIBRARY`/`TK_LIBRARY` pointing at the wrong
  folder (`<prefix>/lib/tcl8.6` instead of the real `<prefix>/tcl/tcl8.6`),
  which makes a bare `tkinter.Tk()` fail with "Can't find a usable
  init.tcl". `gui/app.py` sets those env vars itself (only if unset and the
  real folder is found) before importing tkinter — see
  `_fix_tcl_tk_library_paths`. If GUI tests ever fail with that exact
  error on a fresh machine, this is almost certainly why.
- `GuiPlayer`/`SteppingPlayer` are unit-tested (thread-safety of the
  block/unblock handoff) in `tests/unit/test_gui_player.py` without needing
  a real Tkinter window. Some `TableFrame`/`SetupFrame` *logic* (seat
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
  works around it with a single `module`-scoped `PokerGuiApp` fixture that
  every test's `TableFrame` is parented to, destroying only the `Frame`
  (not the Tk root) between tests. Follow that pattern for any new Tkinter
  test rather than creating a fresh `Tk()`/`PokerGuiApp()` per test.
- **Cards are drawn on a `tk.Canvas`, not image files** (`gui/cards_canvas.py`):
  a plain rectangle plus rank/suit text (Unicode ♠♥♦♣, red for hearts/
  diamonds, black for spades/clubs) for a face-up card, a solid-fill
  rectangle for a face-down back, and a dashed outline for an undealt slot.
  No Pillow/image-asset dependency needed for this. The human's own hole
  cards are drawn full-size above the seat grid; still-live opponents show
  small face-down backs next to their seat box (folded/busted/self show an
  empty slot there instead).

## Setup and running things

Classic venv (not uv — deliberate user preference):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

```powershell
pytest -v                                    # full suite (245+ tests)
pytest --cov=pokerlab                        # with coverage
ruff check .                                 # lint

poker-gui                                    # Tkinter desktop app

poker-train --iterations 200 --hands 512     # self-play PPO (needs the `rl` extra)
poker-train --resume --device cuda --eval-bot shark
# every run archives into checkpoints/pool/ and seats what earlier runs left there
poker-train --pool-models 8 --archive-every 20

poker-play --list-bots                       # show the bot catalog and difficulty ranking
poker-play --players 6 --stack 200 --sb 1 --bb 2 --hands 10 --human-seats 1
poker-play --players 9 --hands 500 --human-seats 0 --seed 42 --bots rock,shark,maniac
```

Re-run `pip install -e ".[dev]"` after pulling changes that add a new
`[project.scripts]` entry point (like `poker-gui` was) — an editable
install doesn't pick up a newly added console script until reinstalled.

Hand histories are written to `hand_histories/session_<id>.jsonl` (gitignored).
Trained models go to `checkpoints/` (also gitignored): `agent.pt` is the live
one, overwritten each save; `checkpoints/pool/` holds the permanent archives
that later training runs seat as opponents.

## Where to extend each future section

- **RL**: implemented end to end (see the "RL" section above). Natural next
  steps, roughly in order of value: multi-process rollout collection (N
  `Table`s in N workers — note `make_policy_fn` returns a closure, which is not
  picklable, so this needs `fork` or a module-level callable); duplicate/mirror
  deals (replay a seeded deck with rotated seats and average) for variance
  reduction; suit-isomorphic canonicalisation of the card planes for a ~4x
  sample-efficiency win; a per-action history encoder (GRU over `ActionRecord`s)
  replacing today's per-street aggregates. `make_rl_bot_factory`/
  `register_rl_bot` for plugging a trained checkpoint into `BOT_CATALOG` are
  still deliberately *not* built — a hardcoded catalog entry pointing at a
  missing checkpoint would break `poker-play --list-bots` for everyone.
- **Vision**: implement `TableStateReader` in `vision/reader.py`. It should
  produce either an `Observation`-compatible read or a `HandHistory`-style
  record — both integration points already exist, no engine changes needed.
  Install the `vision` extra first.
- **GUI**: implemented — see the "GUI" section above, including a visual
  bot builder, opponent card-graphics, a spy toggle, step-through bot
  actions, and a full per-action/showdown log. Remaining rough edges are
  documented inline: the "action_taken" table view is one action behind
  the log text (see `_describe_action`'s surrounding notes), and there's no
  mid-hand stop.

## Testing strategy already in place

Ordered by correctness risk (highest first — see `tests/unit/`):
`test_evaluator.py` (hand ranking), `test_side_pots.py` (side-pot math and
uncalled-bet refund), `test_betting_legal_actions.py` (legal actions,
min-raise/short-all-in), `test_blinds_and_stacks.py` (button rotation, short
blinds, busted players), `test_hand_history.py` (JSONL round-trip),
`test_scripted_bots.py`. `tests/integration/test_full_hand_flow.py` fuzzes
2-9 players across 5 seeds each with a chip-conservation invariant on every
single hand — this is the test most likely to catch a subtle new bug, run
it after any engine change.

For the RL section the equivalent gate is
`tests/integration/test_rl_env_flow.py` (same 2-9 × 5-seed fuzz, asserting the
policy's action bins never produce an `IllegalActionError`); run it after any
change to `rl/action_space.py` or `rl/features.py`. `test_rl_policy.py` is the
only test file that needs the `rl` extra and skips itself cleanly without it.
