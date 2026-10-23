"""Search at the moment of a decision, by sampled worlds.

The network answers a decision from a single forward pass: an intuition. This module spends
computation on the decision instead, by playing it out: it samples possible worlds (the
hands the opponents might hold, the cards still to come), plays each candidate action in each
world to the end of the hand with the models in every seat, and chooses the action that won the
most. Nothing is trained and no model is touched: it is a layer around a `Player`
(`SearchPlayer`), and the engine is the simulator.

**How a world is played.** The decision is rebuilt from the `Observation` alone: a fresh `Table`
with the stacks the hand started with, the actions already taken *replayed* (a scripted prefix,
in the order the engine asks for them) and the deck dealt as the world says
(`engine.arranged_deal`). Seats are relabelled so the button is seat 0, which is where a fresh
`Table` puts it. The hero's candidate action is played at the first decision after the script;
from there every seat -- the hero's included -- plays its model.

**Who the opponents are.** `make_player` builds the model that sits in every simulated seat.
The search does not know what the real opponents are, so this is a *belief about them*: the
hero's own model is the natural choice, and it is exactly right only when that is who they are.
This is a best response to a model of the opponents, not an equilibrium: if the model is wrong
the search can be exploited, and against the model it is right about it gains the most it can.

**Only the opponents still in the hand are imagined.** A player who has folded no longer
decides anything, so its cards matter to nobody: the worlds are the hands of the opponents still
in, their actions are the only ones weighed, and a seat that folded is dealt whatever cards are
left over (the engine deals every seat). This ignores two small things -- the two cards a folded
player holds are not taken out of the live players' hands, and the fold itself says a little about
them -- and it is what keeps the number of worlds down: one opponent left is 990 at the river,
however many sat down.

**Ranges (`SearchConfig.ranges`).** What an opponent did depends on its own cards and on what is
public, never on another opponent's cards, so the likelihood of a world is the *product* of one
likelihood per opponent. Each opponent's range is therefore found on its own, over every hand it
could hold (about a thousand), instead of over every combination of hands (a thousand to the
power of the number of opponents). The worlds that are played are then drawn from the ranges,
one hand per opponent, redrawing when two of them would hold the same card -- the engine cannot
deal a card twice -- and every world counts the same: the weighing is done by the drawing.

**Belief.** `belief="uniform"` samples the opponents' hands uniformly among the cards not seen.
`belief="weighted"` weights each sampled world by the probability that the opponents' models
would have taken the actions they were seen to take, with those cards (importance weighting,
the hand-reading of a poker player): a world where the raiser held nothing a model would raise
with counts for almost nothing. The effective sample size says how many worlds are left; below
`min_ess` the search declines and the caller falls back on the policy.

**Variance.** Every candidate is played in the *same* worlds, so the comparison between actions
is paired and the cards' luck cancels (the same idea as the duplicate decks of
`studies/agents/duel_power.py`).

**Stopping at the hero's next decision (`leaf`).** A rollout from the flop plays the turn and the
river too, with their cards and their luck. With `leaf` it stops when the hero is asked again
(or the hand ends) and takes what the hero's position is worth from `leaf_value`: in the studies,
the model's own critic, which was trained on exactly that kind of state -- a decision of its own --
and is given every player's equity, known in a world. Shorter rollouts and no luck of the cards to
come; the price is the critic's own error, which does not average out (it explains the river well
and the flop much less).

**Many hands at once (`threads`, `SimModel`).** Most of a rollout's time is the model's forward
pass on one decision at a time. With `threads` > 1 the rollouts (and the replays that weigh a
range) run in threads, and every question the simulated seats ask the model goes through a
`BatchGate`: when every running rollout is waiting for an answer, one forward pass answers them
all. The threads do not make the engine faster -- Python runs one at a time -- they make the
questions wait together. A `SimModel` is what the gate needs: the model's batched probabilities.
With threads the order of the draws depends on the scheduling, so a search is no longer
reproducible draw for draw; with one thread it is.

Torch-free: it needs only players and policies, the same argument as `features.py`.
"""

from __future__ import annotations

import itertools
import math
import random
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from pokerlab.cards.card import Card, Rank, Suit
from pokerlab.engine.actions import (
    FORCED_ACTIONS,
    Action,
    ActionType,
    IllegalActionError,
    LegalAction,
)
from pokerlab.engine.arranged_deal import ArrangingRandom, deal_positions
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import PlayerStatus, Street
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import PolicyDecision, PolicyFn, RLAgentPlayer
from pokerlab.rl.action_space import action_index_to_action, legal_action_mask

BELIEFS = ("weighted", "uniform")
# A probability of 0 would put a world out of the sample for good over one unlikely action.
PROBABILITY_FLOOR = 1e-9
_FULL_DECK = tuple(Card(rank, suit) for suit in Suit for rank in Rank)


@dataclass(frozen=True)
class SearchConfig:
    worlds: int = 100  # sampled worlds per decision
    candidates: int = 4  # the most likely action bins, by the policy, that are tried
    belief: str = "weighted"
    min_ess: float = 8.0  # fewer effective worlds than this and the search declines
    streets: frozenset[Street] = frozenset({Street.RIVER})
    # Every world there is instead of `worlds` sampled ones, when there are at most
    # `max_exhaustive` of them (a heads-up river has 990; a turn 45,540): above that, sampled.
    exhaustive: bool = False
    max_exhaustive: int = 5000
    # A world is dropped, and its rollouts never played, as soon as the probability of the
    # opponents' actions replayed so far (the product of the probabilities their models gave each)
    # falls below this: 0 keeps every world and weighs it by that probability.
    world_threshold: float = 0.0
    # Each opponent's hands weighed on their own (see the module docstring), then `worlds` worlds
    # drawn from the ranges. The threshold then drops a *hand* of an opponent.
    ranges: bool = False
    # With ranges: the threshold is relative to the opponent's likeliest hand instead of absolute
    # (a hand is dropped when it is that much less likely than the best of its range). An absolute
    # threshold drops ever more of a range as a hand gets longer -- after five or six actions
    # played half the time each, every hand that mixes is under 1% -- and keeps only the hands
    # that play almost deterministically, which is a bias, not a saving: the worlds are drawn in
    # proportion to the weights anyway.
    relative_threshold: bool = False
    # Stop early and drop candidates that are clearly worse (`stop_z` > 0): every `check_every`
    # worlds, once `min_worlds` have been played, a candidate whose paired difference from the best
    # is more than `stop_z` standard errors below it is no longer played; the search ends when one
    # is left. 0 plays every world for every candidate.
    stop_z: float = 0.0
    min_worlds: int = 100
    check_every: int = 50
    # Play a world that came out more than once a single time, counted as many times as it came
    # out; with one opponent left at the river, the worlds are its range's hands themselves, each
    # weighed by how likely it is (with ranges only).
    dedup: bool = False
    # No search in a pot smaller than this many big blinds: the policy plays it.
    min_pot_bb: float = 0.0
    # The first answer to the hero's candidate (whoever acts next) is not drawn: every answer its
    # model gives at least `branch_min` is played, and the results are averaged with the model's
    # probabilities. The world's value then carries no luck of that draw.
    branch: bool = False
    branch_min: float = 0.02
    # A rollout stops at the hero's next decision and takes its value from the `Searcher`'s
    # `leaf_value` (a hand that ends before it is played to the end as ever).
    leaf: bool = False

    def __post_init__(self) -> None:
        if self.belief not in BELIEFS:
            raise ValueError(f"belief must be one of {BELIEFS}, got {self.belief!r}")
        if not 0.0 <= self.world_threshold < 1.0:
            raise ValueError("world_threshold is a probability: at least 0 and under 1")
        if self.ranges and self.belief != "weighted":
            raise ValueError("ranges are made of weighed hands: they need the weighted belief")
        if self.relative_threshold and not self.ranges:
            raise ValueError("a threshold relative to the likeliest hand needs ranges")
        if self.world_threshold > 0 and self.belief != "weighted":
            raise ValueError("a world can only be dropped for its probability under the weighted belief")
        if self.worlds < 1 or self.candidates < 2:
            raise ValueError("a search needs at least one world and two candidates")
        if not 0.0 <= self.branch_min < 1.0:
            raise ValueError("branch_min is a probability: at least 0 and under 1")
        if self.stop_z < 0 or self.min_worlds < 1 or self.check_every < 1 or self.min_pot_bb < 0:
            raise ValueError("stop_z and min_pot_bb are at least 0, min_worlds and check_every at least 1")


@dataclass(frozen=True)
class Choice:
    """What a search came to: the action, and what it saw."""

    action_index: int
    action: Action
    policy_top: int  # the bin the policy finds most likely
    values: dict[int, float]  # expected result in big blinds of each candidate bin
    ess: float  # effective number of worlds behind the values
    worlds: int
    # Standard error of each candidate's paired difference from the policy's likeliest bin, over
    # the worlds both were played in (0 for that bin itself): how sure the search is of the gap.
    errors: dict[int, float] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.action_index != self.policy_top


@dataclass
class SearchCounts:
    """Kept by a `Searcher`, to say afterwards what it did."""

    decisions: int = 0  # asked, on a street it searches
    searched: int = 0  # answered with a choice
    changed: int = 0  # ... that was not the policy's likeliest bin
    declined_ess: int = 0  # too few effective worlds
    declined_other: int = 0  # nothing to choose between, or a hand it cannot rebuild
    worlds_failed: int = 0  # a replay that did not reach the decision
    worlds_pruned: int = 0  # dropped for falling under `world_threshold`
    worlds_kept: int = 0  # played out and counted
    rollouts: int = 0  # hands played out to the end, candidates times worlds
    range_replays: int = 0  # replays to weigh a hand of an opponent's range
    probes: int = 0  # replays to find who answers the candidate, and how (with `branch`)
    leaves: int = 0  # rollouts stopped at the hero's next decision and valued by `leaf_value`
    dropped_candidates: int = 0  # candidates no longer played once clearly worse
    skipped_small_pot: int = 0  # decisions left to the policy for a pot under `min_pot_bb`
    ess_total: float = 0.0


class _FixedStats:
    """Hands the statistics the hero saw to the simulated `Table`, which puts them in the
    observations of the models, and records nothing (the same as the spot screen's)."""

    def __init__(self, by_seat: dict[int, tuple[float, ...]]) -> None:
        self._by_seat = dict(by_seat)

    def vectors(self, player_ids) -> dict[int, tuple[float, ...]]:
        return {seat: vector for seat, vector in self._by_seat.items() if seat in player_ids}

    def record_hand(self, *args, **kwargs) -> None:
        return None


class _Diverged(Exception):
    """The replay is not the hand that was seen."""


class _Pruned(Exception):
    """The world became less probable than the threshold: stop replaying it."""


class _Leaf(Exception):
    """The hero is asked again: the rollout stops here (with `leaf`)."""

    def __init__(self, features: list[float], observation: Observation) -> None:
        super().__init__("the hero's next decision")
        self.features = features
        self.observation = observation


class _Reached(Exception):
    """The replay got to the hero's decision (carries it, to compare with the real one)."""

    def __init__(
        self, observation: Observation, legal_actions: list[LegalAction], player: RLAgentPlayer | None = None
    ) -> None:
        super().__init__("reached the decision")
        self.observation = observation
        self.legal_actions = legal_actions
        self.player = player  # the model of the seat that was asked


@dataclass(frozen=True)
class _Hand:
    """A decision rebuilt from its observation, in simulation seats (the button is seat 0)."""

    config: GameConfig
    stacks: tuple[int, ...]  # what each simulated seat started the hand with
    steps: tuple[tuple[int, Action], ...]  # the voluntary actions taken so far, in order
    hero: int
    hole: tuple[Card, Card]
    board: tuple[Card, ...]
    stats: dict[int, tuple[float, ...]]
    sim_of: dict[int, int]  # real seat -> simulated seat
    live: tuple[int, ...]  # the opponents (simulated seats) still in the hand

    @property
    def players(self) -> int:
        return self.config.num_players


def _action_of(record) -> Action:
    if record.action_type in (ActionType.BET, ActionType.RAISE):
        return Action(record.action_type, record.amount)  # the raise-to level, as the engine logs it
    return Action(record.action_type)


def _reconstruct(observation: Observation) -> _Hand:
    """The hand as the engine would have to be set up to ask this decision again."""
    history = observation.action_history
    seats = [seat.seat for seat in observation.seats]
    button = seats.index(observation.button_seat)
    order = seats[button:] + seats[:button]  # clockwise from the button: simulated seats 0, 1, ...
    sim_of = {real: index for index, real in enumerate(order)}

    first_stack: dict[int, int] = {}
    for record in history:
        first_stack.setdefault(record.seat, record.stack_before)
    # A seat that has not acted has put nothing in yet: what it has is what it started with.
    started = {seat.seat: first_stack.get(seat.seat, seat.stack) for seat in observation.seats}

    posts = [record for record in history if record.action_type is ActionType.POST_BLIND]
    big_blind = observation.big_blind or max((record.amount for record in posts), default=0)
    if big_blind <= 0:
        raise ValueError("the big blind of the hand is unknown")
    small_blind = posts[0].amount if posts and posts[0].stack_after > 0 else 0
    if not 0 < small_blind < big_blind:  # a short small blind posts less than the blind
        small_blind = max(1, big_blind // 2)

    config = GameConfig(
        num_players=len(order),
        starting_stack=max(started.values()),
        small_blind=small_blind,
        big_blind=big_blind,
    )
    steps = tuple(
        (sim_of[record.seat], _action_of(record))
        for record in history
        if record.action_type not in FORCED_ACTIONS
    )
    return _Hand(
        config=config,
        stacks=tuple(started[real] for real in order),
        steps=steps,
        hero=sim_of[observation.my_seat],
        hole=observation.hole_cards,
        board=tuple(observation.community_cards),
        stats={sim_of[seat]: tuple(vector) for seat, vector in observation.seat_stats.items() if seat in sim_of},
        sim_of=sim_of,
        live=tuple(
            sorted(
                sim_of[seat.seat]
                for seat in observation.seats
                if seat.seat != observation.my_seat and seat.status is not PlayerStatus.FOLDED
            )
        ),
    )


def _bin_of(action: Action, observation: Observation, legal_actions: list[LegalAction]) -> int | None:
    """The action bin an action falls in, for the probability the policy gave it: the bin of
    the same kind whose amount is closest (a human's raise to 7 is the bin nearest to 7)."""
    mask = legal_action_mask(observation, legal_actions)
    best: int | None = None
    best_gap = 0
    for index, allowed in enumerate(mask):
        if not allowed:
            continue
        candidate = action_index_to_action(index, observation, legal_actions)
        if candidate.action_type is not action.action_type:
            continue
        gap = abs(candidate.amount - action.amount)
        if best is None or gap < best_gap:
            best, best_gap = index, gap
    return best


class _Script:
    """The prefix to replay, and what to do when it runs out. Shared by every seat of a
    simulated table: it is the engine that decides whose turn it is, so the steps are
    consumed in the engine's own order."""

    def __init__(
        self,
        steps,
        hero: int,
        *,
        forced: Action | tuple[tuple[int, Action], ...] | None,
        weigh: bool,
        limit: float | None = None,
        weighed: frozenset[int] = frozenset(),
        weigh_from: int = 0,
        start_log: float = 0.0,
    ) -> None:
        self.steps = steps
        self.weigh_from = weigh_from  # the steps before this one are already in `start_log`
        self.limit = limit  # log of the least probability a world may fall to; None: no floor
        self.weighed = weighed  # the seats whose actions count: the opponents still in the hand
        self.hero = hero
        # What is played after the script, each `(seat, action)`: the hero's candidate, and with
        # `branch` the answer to it. None: stop at the hero's decision instead.
        if forced is None:
            self.forced: tuple[tuple[int, Action], ...] = ()
        elif isinstance(forced, Action):
            self.forced = ((hero, forced),)
        else:
            self.forced = tuple(forced)
        self.stop = forced is None  # raise `_Reached` at the first decision after `forced`
        self.leaf = False  # raise `_Leaf` when the hero is asked again after `forced`
        self.weigh = weigh
        self.position = 0
        self.forced_played = 0
        self.log_weight = start_log


class _ReplayPlayer(Player):
    """Plays the script, then the hero's candidate action, then its model."""

    def __init__(self, player_id: str, name: str, inner: RLAgentPlayer, script: _Script) -> None:
        super().__init__(player_id, name)
        self._inner = inner
        self._script = script

    def _probability(self, action: Action, observation: Observation, legal_actions: list[LegalAction]) -> float:
        probabilities = self._inner.action_probabilities(observation, legal_actions)
        if probabilities is None:
            return 1.0
        index = _bin_of(action, observation, legal_actions)
        return PROBABILITY_FLOOR if index is None else probabilities[index]

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        script = self._script
        if script.position < len(script.steps):
            seat, action = script.steps[script.position]
            if seat != observation.my_seat:
                raise _Diverged(f"the script expects seat {seat}, the engine asked {observation.my_seat}")
            script.position += 1
            if script.weigh and seat in script.weighed and script.position > script.weigh_from:
                probability = self._probability(action, observation, legal_actions)
                script.log_weight += math.log(max(probability, PROBABILITY_FLOOR))
                if script.limit is not None and script.log_weight < script.limit:
                    raise _Pruned
            return action
        if script.forced_played < len(script.forced):
            seat, action = script.forced[script.forced_played]
            if observation.my_seat != seat:
                raise _Diverged(f"seat {seat} was to play {action}, the engine asked {observation.my_seat}")
            script.forced_played += 1
            return action
        if script.stop:
            raise _Reached(observation, legal_actions, self._inner)
        if script.leaf and observation.my_seat == script.hero:
            raise _Leaf(self._inner.features_for(observation, legal_actions)[0], observation)
        return self._inner.act(observation, legal_actions)


class BatchGate:
    """Gathers the questions many threads ask a model into one evaluation.

    A thread that is running work calls `enter` first and `leave` after. `ask` adds its question
    and waits; when every running thread is waiting, the last one to ask evaluates them all at
    once (`evaluate`, a batch of features and masks to a batch of probabilities) and wakes the
    others. With no thread running (called from a single thread) it answers at once."""

    def __init__(
        self, evaluate: Callable[[list[list[float]], list[list[bool]]], list[list[float]]]
    ) -> None:
        self._evaluate = evaluate
        self._condition = threading.Condition()
        self._pending: list[tuple[list[float], list[bool], list]] = []
        self._running = 0
        self.batches = 0
        self.questions = 0

    def enter(self, count: int = 1) -> None:
        with self._condition:
            self._running += count

    def leave(self) -> None:
        with self._condition:
            self._running -= 1
            if self._pending and len(self._pending) >= self._running:
                self._flush()

    def ask(self, features: list[float], mask: list[bool]) -> list[float]:
        slot: list = []
        with self._condition:
            self._pending.append((features, mask, slot))
            if len(self._pending) >= self._running:
                self._flush()
            while not slot:
                self._condition.wait()
        answer = slot[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def _flush(self) -> None:
        """Answer every pending question (the condition is held)."""
        pending, self._pending = self._pending, []
        try:
            answers = self._evaluate([f for f, _m, _s in pending], [m for _f, m, _s in pending])
        except BaseException as error:  # noqa: BLE001 - every waiting thread must hear of it
            answers = [error] * len(pending)
        for (_f, _m, slot), answer in zip(pending, answers, strict=True):
            slot.append(answer)
        self.batches += 1
        self.questions += len(pending)
        self._condition.notify_all()


@dataclass(frozen=True)
class SimModel:
    """The model of the simulated seats, as the batched search needs it: its probabilities for a
    batch of decisions (`make_search_policy_fn(...).batch_distribution`), the encoding it reads,
    and the normalisers its seats are built with (as `RLAgentPlayer` takes them)."""

    batch_distribution: Callable[[list[list[float]], list[list[bool]]], list[list[float]]]
    feature_version: int
    big_blind: int
    starting_stack: int


def gated_policy(gate: BatchGate, feature_version: int, rng: random.Random) -> PolicyFn:
    """A policy whose every question goes through `gate`; the action is drawn with `rng`."""

    def policy(
        features: list[float], mask: list[bool], bias: Sequence[float] | None = None, temperature: float = 1.0
    ) -> PolicyDecision:
        if bias is not None:
            raise ValueError("a seat of the batched search plays no style")
        probabilities = gate.ask(features, mask)
        return PolicyDecision(action_index=rng.choices(range(len(probabilities)), weights=probabilities)[0])

    def distribution(features: list[float], mask: list[bool], *_style) -> list[float]:
        return gate.ask(features, mask)

    policy.distribution = distribution  # type: ignore[attr-defined]
    policy.feature_version = feature_version  # type: ignore[attr-defined]
    return policy


class Searcher:
    """Chooses an action by playing the candidates out in sampled worlds.

    `make_player(player_id, name)` builds the model that sits in every simulated seat: a
    `RLAgentPlayer`, with its own policy (the belief about who the opponents are, see the
    module docstring). `choose(..., seat_models=)` can name another one seat by seat, for a
    table of different models: an opponent's range is then weighed with its own model and every
    rollout seats each one where it sits. One `Searcher` is for one process: it keeps a random
    stream and counts."""

    def __init__(
        self,
        make_player: Callable[[str, str], RLAgentPlayer] | None,
        config: SearchConfig | None = None,
        rng: random.Random | None = None,
        *,
        batched: SimModel | None = None,
        threads: int = 1,
        leaf_value: Callable[[list[float], Observation, dict[int, tuple[Card, Card]]], float] | None = None,
    ) -> None:
        if make_player is None and batched is None:
            raise ValueError("a search needs the model of its simulated seats: make_player or batched")
        if config is not None and config.leaf and leaf_value is None:
            raise ValueError("a search that stops at the hero's next decision needs a leaf_value")
        # What the hero's position is worth at its next decision, in big blinds: the features its
        # model reads there, the decision, and every seat's two cards (simulated seats).
        self._leaf_value = leaf_value
        self._make_player = make_player
        # Simulated seat -> the model that sits there, for the decision being searched (`choose`'s
        # `seat_models`); a seat not in it gets `make_player` (or the gate).
        self._seat_models: dict[int, Callable[[str, str], RLAgentPlayer]] = {}
        self.config = config if config is not None else SearchConfig()
        self._rng = rng if rng is not None else random.Random()
        self.counts = SearchCounts()
        self._lock = threading.Lock()
        # Batched: the simulated seats ask through a gate, and the work runs in `threads` threads.
        self._threads = threads if batched is not None else 1
        self._gate = BatchGate(batched.batch_distribution) if batched is not None else None
        self._batched = batched
        self._draws = random.Random(self._rng.random())  # the gated seats' draws
        # The ranges of this decision and of the one before, to carry to the next decision of the
        # same hand: `(hand key, steps covered, {simulated seat: (hands standing, log weights)})`.
        self._current: tuple[tuple, tuple, dict[int, tuple[list, list[float]]]] | None = None
        self._previous: tuple[tuple, tuple, dict[int, tuple[list, list[float]]]] | None = None

    # ---- threads ---------------------------------------------------------------------

    def _add(self, field: str, amount: int = 1) -> None:
        """Count something from any thread."""
        with self._lock:
            setattr(self.counts, field, getattr(self.counts, field) + amount)

    def _seat(self, player_id: str, name: str) -> RLAgentPlayer:
        if self._gate is None:
            return self._make_player(player_id, name)
        model = self._batched
        return RLAgentPlayer(
            player_id,
            name,
            policy_fn=gated_policy(self._gate, model.feature_version, self._draws),
            big_blind=model.big_blind,
            starting_stack=model.starting_stack,
        )

    def _map(self, function: Callable, items: list) -> list:
        """`[function(item) for item in items]`, in threads that ask the model through the gate
        when the search is batched; in order either way."""
        if self._threads <= 1 or len(items) <= 1:
            return [function(item) for item in items]
        gate = self._gate
        workers = min(self._threads, len(items))
        results: list = [None] * len(items)
        errors: list[BaseException] = []
        taken = itertools.count()
        lock = threading.Lock()

        def run() -> None:
            try:
                while not errors:
                    with lock:
                        index = next(taken)
                    if index >= len(items):
                        return
                    results[index] = function(items[index])
            except BaseException as error:  # noqa: BLE001 - raised again below
                errors.append(error)
            finally:
                gate.leave()

        # Every worker is counted before any starts: a worker that asked before the others had
        # entered would be answered alone.
        gate.enter(workers)
        threads = [threading.Thread(target=run) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return results

    # ---- one world -----------------------------------------------------------------

    def _unseen(self, hand: _Hand) -> list[Card]:
        seen = {*hand.hole, *hand.board}
        return [card for card in _FULL_DECK if card not in seen]

    def _deal(
        self, hand: _Hand, dealt: dict[int, tuple[Card, Card]], future: tuple[Card, ...], spare: list[Card]
    ) -> dict[int, Card]:
        """Where each card sits in the deal: the hero's and the board's as seen, the two cards
        given in `dealt` to the seats named there, `future` for the rest of the board, and the
        cards of `spare` (which nobody else holds) for every other seat -- the ones that folded,
        and whoever this deal is not about."""
        positions = deal_positions(hand.players)
        placements: dict[int, Card] = {}
        hero_slots = positions[f"seat{hand.hero}"]
        placements[hero_slots[0]], placements[hero_slots[1]] = hand.hole
        extra = iter(spare)
        for seat in range(hand.players):
            if seat == hand.hero:
                continue
            first, second = positions[f"seat{seat}"]
            placements[first], placements[second] = dealt[seat] if seat in dealt else (next(extra), next(extra))
        coming = iter(future)
        for index, slot in enumerate([*positions["flop"], *positions["turn"], *positions["river"]]):
            placements[slot] = hand.board[index] if index < len(hand.board) else next(coming)
        return placements

    def _placements(self, hand: _Hand, groups, left: list[Card]) -> dict[int, Card]:
        """`_deal` from `groups`: the two cards of each opponent still in (in seat order), then
        the cards still to come on the board."""
        return self._deal(hand, dict(zip(hand.live, groups[:-1], strict=True)), tuple(groups[-1]), left)

    @staticmethod
    def _hand_key(hand: _Hand) -> tuple:
        """What stays the same from one decision of a hand to the next: the hero's cards, the
        stacks the hand started with and the seating. A new hand changes it."""
        return (hand.hole, hand.stacks, tuple(sorted(hand.sim_of.items())))

    def _range_of(self, hand: _Hand, seat: int, limit: float | None) -> tuple[list[tuple[Card, Card]], list[float]]:
        """The hands one opponent could hold and how likely its actions make each: every
        two-card hand from the cards not seen, replayed with only that opponent's actions weighed
        (the others, and the cards to come, are filler: they cannot change what `seat` did).

        Within a hand the work is carried from one decision to the next (`_previous`): the
        actions already weighed have the same probability with the same cards, so only the new
        ones are weighed, and only for the hands still standing -- a hand dropped under the
        threshold cannot come back, since a weight only goes down, and a hand holding a card
        that has since come on the board is out."""
        unseen = self._unseen(hand)
        missing = 5 - len(hand.board)
        key = self._hand_key(hand)
        if self._current is None or self._current[0] != key or self._current[1] != hand.steps:
            self._previous, self._current = self._current, (key, hand.steps, {})
        prior = None
        for remembered in (self._current, self._previous):
            if remembered is None:
                continue
            known_key, known_steps, known = remembered
            if known_key == key and hand.steps[: len(known_steps)] == known_steps and seat in known:
                prior = (len(known_steps), known[seat])
                break
        if prior is None:
            candidates = [(pair, 0.0) for pair in itertools.combinations(unseen, 2)]
            weigh_from = 0
        else:
            weigh_from, (old_hands, old_logs) = prior
            board = set(hand.board)
            candidates = [(pair, log) for pair, log in zip(old_hands, old_logs, strict=True) if not board & set(pair)]

        def weigh(candidate):
            (first, second), start_log = candidate
            rest = [card for card in unseen if card is not first and card is not second]
            placements = self._deal(hand, {seat: (first, second)}, tuple(rest[:missing]), rest[missing:])
            return self.reach(
                hand, placements, weigh=True, limit=limit, weighed=frozenset({seat}),
                weigh_from=weigh_from, start_log=start_log,
            )

        hands: list[tuple[Card, Card]] = []
        logs: list[float] = []
        self.counts.range_replays += len(candidates)
        for ((first, second), _start), reached in zip(candidates, self._map(weigh, candidates), strict=True):
            if reached is None:
                self.counts.worlds_failed += 1
            elif reached[0] is None:
                self.counts.worlds_pruned += 1
            else:
                hands.append((first, second))
                logs.append(reached[2])

        self._current[2][seat] = (hands, logs)
        return hands, [math.exp(log) for log in logs]

    def _joint_worlds(
        self,
        hand: _Hand,
        ranges: dict[int, tuple[list[tuple[Card, Card]], list[float]]],
        count: int,
        *,
        keyed: bool = False,
    ) -> Iterator:
        """`count` worlds with one hand per live opponent drawn from its range, redrawn when two
        would hold the same card; the cards to come are drawn from what is left. `keyed` yields
        `(key, placements)`, the key being what makes two worlds the same one: the live opponents'
        hands and the cards to come (a folded seat's cards change nothing)."""
        unseen = self._unseen(hand)
        missing = 5 - len(hand.board)
        cumulative = {seat: list(itertools.accumulate(weights)) for seat, (_, weights) in ranges.items()}
        produced = attempts = 0
        while produced < count and attempts < 50 * count:
            attempts += 1
            dealt: dict[int, tuple[Card, Card]] = {}
            used: set[Card] = set()
            for seat in hand.live:
                hands, _ = ranges[seat]
                chosen = self._rng.choices(hands, cum_weights=cumulative[seat])[0]
                if used & set(chosen):
                    break
                used |= set(chosen)
                dealt[seat] = chosen
            else:
                rest = [card for card in unseen if card not in used]
                self._rng.shuffle(rest)
                produced += 1
                future = tuple(rest[:missing])
                placements = self._deal(hand, dealt, future, rest[missing:])
                if keyed:
                    yield (tuple(dealt[seat] for seat in hand.live), frozenset(future)), placements
                else:
                    yield placements

    def _sample_world(self, hand: _Hand) -> dict[int, Card]:
        """A world drawn from the cards not seen."""
        unseen = self._unseen(hand)
        self._rng.shuffle(unseen)
        drawn = iter(unseen)
        groups = [(next(drawn), next(drawn)) for _ in hand.live]
        groups.append(tuple(next(drawn) for _ in range(5 - len(hand.board))))
        return self._placements(hand, groups, list(drawn))

    def _all_worlds(self, hand: _Hand) -> Iterator[dict[int, Card]] | None:
        """Every world there is, or None if there are more than `max_exhaustive`: each live
        opponent's two cards and the cards still to come, each set of cards once."""
        sizes = [2] * len(hand.live) + [5 - len(hand.board)]
        unseen = self._unseen(hand)
        total, left = 1, len(unseen)
        for size in sizes:
            total *= math.comb(left, size)
            left -= size
        if total > self.config.max_exhaustive:
            return None

        def groups(cards: list[Card], remaining_sizes: list[int]):
            if not remaining_sizes:
                yield (), cards
                return
            for chosen in itertools.combinations(cards, remaining_sizes[0]):
                rest = [card for card in cards if card not in chosen]
                for tail, leftover in groups(rest, remaining_sizes[1:]):
                    yield (chosen, *tail), leftover

        return (self._placements(hand, chosen, leftover) for chosen, leftover in groups(unseen, sizes))

    def _table(self, hand: _Hand, placements: dict[int, Card], script: _Script) -> Table:
        players = [
            _ReplayPlayer(
                f"sim{seat}",
                f"S{seat}",
                self._seat_models[seat](f"sim{seat}", f"S{seat}")
                if seat in self._seat_models
                else self._seat(f"sim{seat}", f"S{seat}"),
                script,
            )
            for seat in range(hand.players)
        ]
        table = Table(
            hand.config,
            players,
            rng=ArrangingRandom(placements, seed=0),
            stats_tracker=_FixedStats(hand.stats),
        )
        table.stacks = list(hand.stacks)
        return table

    def reach(
        self,
        hand: _Hand,
        placements: dict[int, Card],
        *,
        weigh: bool,
        limit: float | None = None,
        weighed: frozenset[int] | None = None,
        weigh_from: int = 0,
        start_log: float = 0.0,
    ) -> tuple[Observation | None, list[LegalAction] | None, float] | None:
        """Replay the hand up to the hero's decision: what the hero is asked there and the log
        weight of the opponents' actions on the way (None: the replay is not the hand seen).
        A world that fell under `limit` on the way is `(None, None, log weight so far)`."""
        script = _Script(
            hand.steps,
            hand.hero,
            forced=None,
            weigh=weigh,
            limit=limit,
            weighed=frozenset(hand.live) if weighed is None else weighed,
            weigh_from=weigh_from,
            start_log=start_log,
        )
        table = self._table(hand, placements, script)
        try:
            table.play_hand()
        except _Reached as reached:
            return reached.observation, reached.legal_actions, script.log_weight
        except _Pruned:
            return None, None, script.log_weight
        except (_Diverged, IllegalActionError):
            return None
        return None  # the hand ended before the decision

    def rollout(
        self, hand: _Hand, placements: dict[int, Card], forced: Action | tuple[tuple[int, Action], ...]
    ) -> float | None:
        """The hero's result in big blinds when it plays `forced` at the decision (an action, or
        `(seat, action)` pairs: the hero's and what follows) and the models play the rest (None:
        the replay is not the hand seen)."""
        script = _Script(hand.steps, hand.hero, forced=forced, weigh=False)
        script.leaf = self.config.leaf
        table = self._table(hand, placements, script)
        self._add("rollouts")
        try:
            result = table.play_hand()
        except _Leaf as leaf:
            self._add("leaves")
            positions = deal_positions(hand.players)
            holes = {
                seat: (placements[positions[f"seat{seat}"][0]], placements[positions[f"seat{seat}"][1]])
                for seat in range(hand.players)
            }
            return self._leaf_value(leaf.features, leaf.observation, holes)
        except (_Diverged, IllegalActionError):
            return None
        return (result.final_stacks[hand.hero] - hand.stacks[hand.hero]) / hand.config.big_blind

    def _probe(self, hand: _Hand, placements: dict[int, Card], action: Action):
        """Who answers `action` and with what in front of them: `_Reached` at that decision, the
        hero's result (a float) if the hand ends with nobody else asked, None if the replay is
        not the hand seen."""
        script = _Script(hand.steps, hand.hero, forced=((hand.hero, action),), weigh=False)
        script.stop = True
        table = self._table(hand, placements, script)
        self._add("probes")
        try:
            result = table.play_hand()
        except _Reached as reached:
            return reached
        except (_Diverged, IllegalActionError):
            return None
        return (result.final_stacks[hand.hero] - hand.stacks[hand.hero]) / hand.config.big_blind

    def branched_rollout(self, hand: _Hand, placements: dict[int, Card], action: Action) -> float | None:
        """`rollout`, with the first answer to `action` not drawn: each answer the model of
        whoever acts next gives at least `branch_min` is played, and the results are averaged
        with its probabilities (renormalised over the answers played)."""
        probe = self._probe(hand, placements, action)
        if probe is None or isinstance(probe, float):
            return probe
        observation, legal, model = probe.observation, probe.legal_actions, probe.player
        probabilities = model.action_probabilities(observation, legal) if model is not None else None
        if probabilities is None:
            return self.rollout(hand, placements, action)
        mask = legal_action_mask(observation, legal)
        answers = [(i, p) for i, p in enumerate(probabilities) if mask[i] and p >= self.config.branch_min]
        if not answers:
            answers = [max(((i, p) for i, p in enumerate(probabilities) if mask[i]), key=lambda pair: pair[1])]
        total = value = 0.0
        for index, probability in answers:
            answer = action_index_to_action(index, observation, legal)
            result = self.rollout(hand, placements, ((hand.hero, action), (observation.my_seat, answer)))
            if result is None:
                return None
            value += probability * result
            total += probability
        return value / total

    # ---- one decision --------------------------------------------------------------

    def _worlds(self, hand: _Hand, ranges) -> Iterator[tuple[dict[int, Card], float]]:
        """The worlds to play, each with the log of the weight it starts with (before any
        replay weighs it): 0 for a world drawn as often as it should be, the log of how many
        times it came out when duplicates are folded together, the log of its likelihood for a
        hand of the range played as a world of its own."""
        config = self.config
        if ranges is not None:
            if config.dedup and len(hand.board) == 5 and len(hand.live) == 1:
                # Nothing left to come and one opponent: a world is a hand of its range.
                (seat,) = hand.live
                hands, weights = ranges[seat]
                unseen = self._unseen(hand)
                order = list(range(len(hands)))
                self._rng.shuffle(order)
                for index in order:
                    held = set(hands[index])
                    rest = [card for card in unseen if card not in held]
                    yield self._deal(hand, {seat: hands[index]}, (), rest), math.log(weights[index])
                return
            if config.dedup:
                counts: dict = {}
                first: dict = {}
                for key, placements in self._joint_worlds(hand, ranges, config.worlds, keyed=True):
                    counts[key] = counts.get(key, 0) + 1
                    first.setdefault(key, placements)
                for key, placements in first.items():  # in the order they first came out
                    yield placements, math.log(counts[key])
                return
            for placements in self._joint_worlds(hand, ranges, config.worlds):
                yield placements, 0.0
            return
        worlds = self._all_worlds(hand) if config.exhaustive else None
        if worlds is not None and config.stop_z > 0:
            # Stopping early on a prefix needs the prefix to be a random sample, not the
            # enumeration's order.
            worlds = list(worlds)
            self._rng.shuffle(worlds)
        if worlds is None:
            worlds = (self._sample_world(hand) for _ in range(config.worlds))
        for placements in worlds:
            yield placements, 0.0

    @staticmethod
    def _weighted_mean(values: list[float], weights: list[float]) -> float:
        total = sum(weights[: len(values)])
        return sum(w * v for w, v in zip(weights, values, strict=False)) / total

    def _race(self, active: list[int], per: dict[int, list[float]], log_weights: list[float]) -> list[int]:
        """The candidates still worth playing: those not clearly worse than the best so far, by
        their paired differences over the worlds both were played in."""
        top = max(log_weights)
        weights = [math.exp(value - top) for value in log_weights]
        best = max(active, key=lambda index: self._weighted_mean(per[index], weights))
        kept = [best]
        for index in active:
            if index == best:
                continue
            differences = [b - c for b, c in zip(per[best], per[index], strict=True)]
            total = sum(weights)
            mean = sum(w * d for w, d in zip(weights, differences, strict=True)) / total
            variance = sum(w * (d - mean) ** 2 for w, d in zip(weights, differences, strict=True)) / total
            effective = total * total / sum(w * w for w in weights)
            error = math.sqrt(variance / effective)
            if mean > 0 and mean > self.config.stop_z * error:
                self.counts.dropped_candidates += 1
            else:
                kept.append(index)
        return [index for index in active if index in kept]

    def choose(
        self,
        observation: Observation,
        legal_actions: list[LegalAction],
        base: RLAgentPlayer,
        seat_models: dict[int, Callable[[str, str], RLAgentPlayer]] | None = None,
    ) -> Choice | None:
        """The searched action, or None to leave the decision to the policy: not a street this
        search covers, a pot too small to be worth it, nothing to choose between, too few
        effective worlds.

        `seat_models` (real seat -> a maker of that seat's `RLAgentPlayer`) is who the search
        believes sits where, the hero's seat included; seats it leaves out get `make_player`. The
        ranges carried from an earlier decision of the hand assume the same models."""
        config, counts = self.config, self.counts
        if observation.street not in config.streets:
            return None
        too_small = config.min_pot_bb * observation.big_blind
        if config.min_pot_bb > 0 and observation.big_blind > 0 and observation.pot_size < too_small:
            counts.skipped_small_pot += 1
            return None
        counts.decisions += 1
        mask = legal_action_mask(observation, legal_actions)
        probabilities = base.action_probabilities(observation, legal_actions)
        legal_bins = [index for index, allowed in enumerate(mask) if allowed]
        has_ante = any(r.action_type is ActionType.POST_ANTE for r in observation.action_history)
        if probabilities is None or len(legal_bins) < 2 or has_ante:
            counts.declined_other += 1
            return None
        candidates = sorted(legal_bins, key=lambda index: -probabilities[index])[: config.candidates]
        try:
            hand = _reconstruct(observation)
        except ValueError:
            counts.declined_other += 1
            return None
        self._seat_models = {hand.sim_of[real]: maker for real, maker in (seat_models or {}).items()
                             if real in hand.sim_of}
        actions = {index: action_index_to_action(index, observation, legal_actions) for index in candidates}

        weigh = config.belief == "weighted"
        limit = math.log(config.world_threshold) if config.world_threshold > 0 else None
        belief_ess: float | None = None
        ranges = None
        if config.ranges:
            ranges = {}
            belief_ess = float("inf")
            relative = config.relative_threshold and config.world_threshold > 0
            for seat in hand.live:
                # A relative threshold needs the whole range first (and a hand under it now can
                # come back above it later, so the carried range keeps every hand).
                hands, weights = self._range_of(hand, seat, None if config.relative_threshold else limit)
                if relative and hands:
                    floor = max(weights) * config.world_threshold
                    kept = [(pair, weight) for pair, weight in zip(hands, weights, strict=True) if weight >= floor]
                    self._add("worlds_pruned", len(hands) - len(kept))
                    hands, weights = [pair for pair, _ in kept], [weight for _, weight in kept]
                if not hands:
                    counts.declined_other += 1
                    return None
                ranges[seat] = (hands, weights)
                belief_ess = min(belief_ess, sum(weights) ** 2 / sum(w * w for w in weights))
            weigh = False  # the weighing is done: the worlds are drawn from the ranges

        active = list(candidates)
        per: dict[int, list[float]] = {index: [] for index in candidates}
        log_weights: list[float] = []
        play = self.branched_rollout if config.branch else self.rollout
        chunk_size = config.check_every if config.stop_z > 0 else max(16, 2 * self._threads)
        worlds = self._worlds(hand, ranges)
        while True:
            chunk = list(itertools.islice(worlds, chunk_size))
            if not chunk:
                break
            if weigh:
                weighed = self._map(lambda item: self.reach(hand, item[0], weigh=True, limit=limit), chunk)
                kept = []
                for (placements, start_weight), reached in zip(chunk, weighed, strict=True):
                    if reached is None:
                        counts.worlds_failed += 1
                    elif reached[0] is None:
                        counts.worlds_pruned += 1
                    else:
                        kept.append((placements, start_weight + reached[2]))
            else:
                kept = chunk
            tasks = [(world, index) for world in range(len(kept)) for index in active]
            values = self._map(lambda task, kept=kept: play(hand, kept[task[0]][0], actions[task[1]]), tasks)
            outcomes: list[dict[int, float | None]] = [{} for _ in kept]
            for (world, index), value in zip(tasks, values, strict=True):
                outcomes[world][index] = value
            for (_placements, log_weight), outcome in zip(kept, outcomes, strict=True):
                if any(value is None for value in outcome.values()):
                    counts.worlds_failed += 1
                    continue
                log_weights.append(log_weight)
                for index, value in outcome.items():
                    per[index].append(value)
                counts.worlds_kept += 1
            if config.stop_z > 0 and len(active) > 1 and len(log_weights) >= config.min_worlds:
                active = self._race(active, per, log_weights)
                if len(active) == 1:
                    break

        if not log_weights:
            counts.declined_other += 1
            return None
        top_log = max(log_weights)
        weights = [math.exp(value - top_log) for value in log_weights]
        total = sum(weights)
        ess = belief_ess if belief_ess is not None else total * total / sum(weight * weight for weight in weights)
        if ess < config.min_ess:
            counts.declined_ess += 1
            return None
        values = {index: self._weighted_mean(per[index], weights) for index in candidates if per[index]}
        policy_top = candidates[0]
        # The policy's likeliest bin wins a tie, so an indifferent search leaves the policy alone;
        # a candidate dropped as clearly worse is not a contender.
        best = max(active, key=lambda index: (values[index], index == policy_top))
        counts.searched += 1
        counts.ess_total += ess
        counts.changed += best != policy_top
        return Choice(
            action_index=best,
            action=actions[best],
            policy_top=policy_top,
            values=values,
            ess=ess,
            worlds=len(log_weights),
            errors={index: self._paired_error(per[index], per[policy_top], weights) for index in values},
        )

    @staticmethod
    def _paired_error(values: list[float], reference: list[float], weights: list[float]) -> float:
        """Standard error of the weighted mean of `values - reference` over the worlds both have
        (a candidate dropped early has a prefix of them)."""
        count = min(len(values), len(reference))
        if count < 2 or values is reference:
            return 0.0
        used = weights[:count]
        total = sum(used)
        differences = [a - b for a, b in zip(values[:count], reference[:count], strict=True)]
        mean = sum(w * d for w, d in zip(used, differences, strict=True)) / total
        variance = sum(w * (d - mean) ** 2 for w, d in zip(used, differences, strict=True)) / total
        effective = total * total / sum(w * w for w in used)
        return math.sqrt(variance / effective)


class SearchPlayer(Player):
    """A model that thinks before it acts: a `RLAgentPlayer` whose decisions on the streets the
    search covers go through a `Searcher`, and whose other decisions (or the ones the search
    declines) are the policy's own."""

    def __init__(
        self,
        player_id: str,
        name: str,
        base: RLAgentPlayer,
        searcher: Searcher,
        seat_models: dict[int, Callable[[str, str], RLAgentPlayer]] | None = None,
    ) -> None:
        super().__init__(player_id, name)
        self._base = base
        self._searcher = searcher
        self._seat_models = seat_models  # who the search believes sits where (`Searcher.choose`)
        self.choices: list[Choice] = []

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        choice = self._searcher.choose(observation, legal_actions, self._base, seat_models=self._seat_models)
        if choice is None:
            return self._base.act(observation, legal_actions)
        self.choices.append(choice)
        return choice.action


__all__ = [
    "BELIEFS",
    "Choice",
    "SearchConfig",
    "SearchCounts",
    "SearchPlayer",
    "Searcher",
]
