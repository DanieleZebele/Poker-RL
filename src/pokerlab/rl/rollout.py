"""Turning played hands into PPO training data.

There is no Gym-style `step()` in this path on purpose. `Table` drives a hand
synchronously and `RLAgentPlayer` reports every decision through its
`on_decision` hook, so a trajectory falls out of ordinary play with no threads
and no re-implementation of the betting loop.

Pure Python, like the rest of the encoding path: numpy and torch only earn their
place in `rl/ppo.py`, where trajectories become tensors.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.history import HandHistory
from pokerlab.engine.stats import StatsTracker, analyse_hand
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import DealView, DecisionRecord, PolicyFn, RLAgentPlayer
from pokerlab.rl.allin_reward import DEFAULT_ALLIN_RUNOUTS, expected_deltas
from pokerlab.rl.style_log import SIZE_GROUPS, STYLE_WINDOW, size_group
from pokerlab.rl.styles import StyleConfig
from pokerlab.rl.table_mix import TableMix

STYLE_PLAYER = "learner"

# How the training collector seats its tables. A table keeps the same players for
# `DEFAULT_TABLE_HANDS` hands (the statistics window of `engine/stats.py`), so what
# the model reads about an opponent is true of the player in front of it; and
# `DEFAULT_CONCURRENT_TABLES` tables are played in turn, hand by hand, so a batch is
# not the same few opponents repeated. One hand per table is the old behaviour.
DEFAULT_TABLE_HANDS = 200
DEFAULT_CONCURRENT_TABLES = 8


@dataclass(frozen=True)
class CriticFns:
    """What a collector needs because the critic reads more than the policy does
    (`rl/policy.py`): two batched callables, so this module stays free of torch.

    `equity` maps the `DealView`s of many decisions to one row of equities each (a
    number per seat in the order of the views, 0 for a seat out of the hand), and `value`
    maps their policy features and those rows to the critic's state values."""

    equity: Callable[[Sequence[DealView]], list[list[float]]]
    value: Callable[[Sequence[list[float]], Sequence[list[float]]], list[float]]


# How many decisions go through the critic's callables at once.
CRITIC_CHUNK = 4096


@dataclass
class HandTrajectory:
    """One seat's decisions in one hand, plus the terminal reward.

    The episode is a single hand, and every hand starts from freshly drawn
    stacks: `Observation` has no memory of previous hands, so an episode
    spanning several hands would not be Markov with respect to the features.
    """

    seat: int
    player_id: str
    decisions: list[DecisionRecord]
    reward: float  # chip delta over the hand, always in big blinds
    num_players: int = 0  # the table this hand was played at, for per-size reporting
    # The most this seat could win or lose, in big blinds: its own stack or the
    # deepest opponent's, whichever is smaller. What sets the scale of the hand's
    # result, so it is what `value_diagnostics` groups the targets by.
    effective_stack_bb: float = 0.0
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)


def compute_gae(
    trajectory: HandTrajectory, *, gamma: float = 1.0, lam: float = 0.95
) -> tuple[list[float], list[float]]:
    """Generalised advantage estimation over one hand.

    `gamma` defaults to 1.0: hands are finite and the reward is terminal, so
    every decision genuinely contributes to the final chip delta and
    discounting would only add bias. `lam < 1` still trades a little bias for
    variance against an imperfect critic.
    """
    values = [d.value for d in trajectory.decisions]
    rewards = [d.reward for d in trajectory.decisions]
    advantages = [0.0] * len(values)

    next_value = 0.0  # the hand is over after the last decision
    running = 0.0
    for t in reversed(range(len(values))):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
        next_value = values[t]

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


@dataclass(frozen=True)
class Opponent:
    """A non-learner seat filler: a label plus a uniform-signature factory."""

    label: str
    factory: Callable[[str, str, random.Random], Player]


def policy_opponent(label: str, policy_fn: PolicyFn, mix: TableMix) -> Opponent:
    """Wrap a frozen policy (a previously trained model) as an opponent.

    The factory takes an optional style (`rl/styles.py`) as a fourth argument: the same
    weights, with a push on the logits. The model is loaded once however many styles it
    plays with."""

    def factory(player_id: str, name: str, rng: random.Random, style=None) -> Player:
        return RLAgentPlayer(
            player_id,
            name,
            policy_fn=policy_fn,
            big_blind=mix.big_blind,
            starting_stack=mix.starting_stack,
            style=style,
        )

    return Opponent(label=label, factory=factory)


class OpponentPool:
    """Who fills the seats the learner is not sitting in.

    Self-play against nothing but the current policy can chase its own tail:
    the reward is relative, so both sides can drift without either getting
    stronger, and the agent learns exploits that only work against its twin.
    Previously trained models (`extra_opponents`, drawn from the ranked pool --
    see `train.py::registry_opponents`) anchor it to something that does not
    move with it.

    **The run's own past selves are deliberately not in here.** A frozen
    snapshot of the learner is a copy of the network being trained, so it
    drifts with it and anchors nothing -- the opposition it provides is the
    same tail-chasing self-play the fixed pool exists to replace, and every
    seat it took was a seat not facing a real, independently trained model.
    With nothing in the pool (an empty store, the very first run ever),
    `sample` returns `None` and every seat goes to the learner -- plain
    self-play until there is something else to train against.
    """

    def __init__(self, opponents: Sequence[Opponent] = ()) -> None:
        self._fixed = list(opponents)

    def sample(self, rng: random.Random) -> Opponent | None:
        """An opponent for one seat, or None to seat the learner there."""
        return rng.choice(self._fixed) if self._fixed else None

    def __len__(self) -> int:
        return len(self._fixed)


class SeatProxy(Player):
    """Delegates to whoever occupies the seat this hand.

    Swapping occupants this way keeps one `Table` alive across hands, so the
    button keeps rotating; rebuilding the table per hand would reset it to the
    same seat every time.
    """

    def __init__(self, player_id: str, name: str) -> None:
        super().__init__(player_id, name)
        self.inner: Player | None = None

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        assert self.inner is not None, "seat was never assigned an occupant"
        return self.inner.act(observation, legal_actions)


class TableBank:
    """One `Table` per table size, built the first time that size is needed.

    Every size keeps its own button and its own seat proxies, so a run that
    moves between 2 and 9 seats never rebuilds a table (which would reset the
    button) and a size's positions stay uniform over time.

    **Stacks are drawn here, per hand, from a separate rng**: `Table` consumes its
    own `random.Random` only to shuffle, once per hand, so keeping the stacks off
    it means a fixed seed deals the identical card sequence whatever the stacks
    and the betting turn out to be -- which the duplicate-deck measurements
    (`studies/agents/duel_power.py`) rely on.
    """

    def __init__(
        self,
        mix: TableMix,
        rng: random.Random,
        stack_rng: random.Random | None = None,
    ) -> None:
        self.mix = mix
        self._rng = rng
        self._stack_rng = stack_rng if stack_rng is not None else random.Random(rng.random())
        self._tables: dict[tuple[int, int], tuple[Table, list[SeatProxy]]] = {}
        # The hand just played, for a caller that wants more than the stacks.
        self.last_hand: HandHistory | None = None
        # Every seat's hole cards in the hand being played, from the table's own
        # spectator hook: what a critic that sees more than the players do reads.
        self.hole_cards: dict[int, tuple[Card, Card]] = {}

    def _entry(self, num_players: int, slot: int = 0) -> tuple[Table, list[SeatProxy]]:
        entry = self._tables.get((slot, num_players))
        if entry is None:
            proxies = [SeatProxy(f"s{seat}", f"S{seat}") for seat in range(num_players)]
            entry = (
                Table(
                    self.mix.config(num_players),
                    list(proxies),
                    rng=self._rng,
                    on_hand_started=self._hand_started,
                ),
                proxies,
            )
            self._tables[(slot, num_players)] = entry
        return entry

    def _hand_started(self, info: dict) -> None:
        self.hole_cards = info["hole_cards"]

    def seats(self, num_players: int, slot: int = 0) -> list[SeatProxy]:
        """The proxies of the `num_players` table, for the caller to seat.

        `slot` names one of several tables of the same size played at once (each with
        its own button and its own statistics); one slot is all most callers use."""
        return self._entry(num_players, slot)[1]

    def reset_stats(self, num_players: int, slot: int = 0) -> None:
        """Start the statistics of this table afresh: a new set of players is about to
        sit, and what was learned about the last ones describes nobody here."""
        self._entry(num_players, slot)[0].stats_tracker = StatsTracker()

    def play_hand(self, num_players: int, slot: int = 0) -> tuple[list[int], list[int]]:
        """One hand from freshly drawn stacks: `(stacks before, stacks after)`."""
        table, _proxies = self._entry(num_players, slot)
        before = self.mix.draw_stacks(self._stack_rng, num_players)
        table.stacks = list(before)
        self.last_hand = table.play_hand().hand_history
        return before, list(table.stacks)

    def play_session(
        self,
        num_players: int,
        hands: int,
        on_hand: Callable[[HandHistory], None] | None = None,
    ) -> list[int]:
        """`hands` hands at one table size: the chip delta of every seat.

        **A session has its own opponent statistics**: the tracker starts blank and watches
        the session's hands, so a model reads, as it plays, what the players in front of it
        have done -- as in training, which is what makes a rating measure the model it
        trained. The next session, with other players in the seats, starts blank again.

        `on_hand`, if given, is handed each finished hand."""
        self.reset_stats(num_players)
        deltas = [0] * num_players
        for _ in range(hands):
            before, after = self.play_hand(num_players)
            if on_hand is not None and self.last_hand is not None:
                on_hand(self.last_hand)
            for seat in range(num_players):
                deltas[seat] += after[seat] - before[seat]
        return deltas


@dataclass
class _TableSession:
    """One table of the training collector: who sits where until `hands_left` runs out."""

    num_players: int
    occupants: list[Opponent | None]  # None: the learner
    players: list[Player]
    hands_left: int


class SelfPlayCollector:
    """Plays hands and collects the learner's trajectories.

    Every hand draws its own table (`TableMix`): a size, and an independent
    starting stack for every seat. The stacks are redrawn after every hand
    rather than carried over, so the agent trains on a stationary distribution
    instead of drifting into short-stack and busted-seat states; the button of
    each size still rotates, so positions stay uniform.

    With no `opponent_pool`, every seat is the learner -- plain self-play.

    `reward_scale` multiplies the big-blind chip delta to produce the *training*
    reward the critic has to predict. Leaving it at 1.0 means value targets span
    roughly +/- the deepest stack in big blinds -- 100 -- so the squared value loss
    lands around 10,000 and swamps a policy loss of order 0.01.
    `big_blind / starting_stack` (the deepest stack) puts a full-depth swing at
    1.0. `HandTrajectory.reward` stays in big blinds regardless, for reporting.

    **A hand whose betting closed before the river is trained on its expected result**
    (`rl/allin_reward.py`) over `allin_runouts` boards instead of the one that came:
    no decision follows, so it is unbiased and drops the runout's luck, which is most
    of the value target's variance on the preflop. 0 trains on the chips that moved.
    `HandTrajectory.reward` is always the chips that moved.
    """

    def __init__(
        self,
        mix: TableMix,
        policy_fn: PolicyFn,
        *,
        rng: random.Random | None = None,
        opponent_pool: OpponentPool | None = None,
        opponent_probability: float = 0.5,
        reward_scale: float = 1.0,
        table_hands: int = DEFAULT_TABLE_HANDS,
        concurrent_tables: int = DEFAULT_CONCURRENT_TABLES,
        allin_runouts: int = DEFAULT_ALLIN_RUNOUTS,
        styles: StyleConfig | None = None,
        critic: CriticFns,
    ) -> None:
        if table_hands < 1 or concurrent_tables < 1:
            raise ValueError("table_hands and concurrent_tables must be at least 1")
        if allin_runouts < 0:
            raise ValueError("allin_runouts must be 0 (off) or more")
        self._allin_runouts = allin_runouts
        self._styles = styles if styles is not None else StyleConfig()
        # Opponent seats opened so far, and how many of them were given a style.
        self.opponent_seats = 0
        self.styled_seats = 0
        self._critic = critic
        self._mix = mix
        self._reward_scale = reward_scale
        self._table_hands = table_hands
        self._sessions: list[_TableSession | None] = [None] * concurrent_tables
        self._turn = 0
        self._pool = opponent_pool
        self._opponent_probability = opponent_probability
        self._rng = rng if rng is not None else random.Random()
        self._pending: dict[int, list[DecisionRecord]] = {}
        self._style = StatsTracker(window=STYLE_WINDOW)
        # The same, kept apart per group of table sizes, each over its own window.
        self._style_by_group = {
            f"{low}-{high}": StatsTracker(window=STYLE_WINDOW) for low, high in SIZE_GROUPS
        }

        self._bank = TableBank(mix, self._rng)
        # Its own rng, so the boards it imagines never touch the deals.
        self._runout_rng = random.Random(self._rng.random())
        # One learner per seat index, shared by every table size: a learner holds
        # no per-table state, and its seat comes from the observation.
        self._learners = [
            RLAgentPlayer(
                f"s{seat}",
                f"RL{seat}",
                policy_fn=policy_fn,
                big_blind=mix.big_blind,
                starting_stack=mix.starting_stack,
                on_decision=self._record,
                deal_holes=lambda: self._bank.hole_cards,
            )
            for seat in range(mix.max_players)
        ]

    def _record_style(self, learner_seats: set[int]) -> None:
        """Add what the learner's seats did in the hand just played to its style."""
        hand = self._bank.last_hand
        if hand is None:
            return
        dealt = list(hand.starting_stacks)
        counts = analyse_hand(
            hand.actions,
            dealt=dealt,
            button_seat=hand.button_seat,
            board_cards=len(hand.community_cards),
            player_ids={seat: f"s{seat}" for seat in dealt},
        )
        by_size = self._style_by_group[size_group(len(dealt))]
        for seat in learner_seats:
            if seat in hand.starting_stacks:
                self._style.add(STYLE_PLAYER, counts[f"s{seat}"])
                by_size.add(STYLE_PLAYER, counts[f"s{seat}"])

    @property
    def style_hands(self) -> int:
        """Learner-seat hands behind `style_rates` (it fills up to `STYLE_WINDOW`)."""
        return self._style.hands(STYLE_PLAYER)

    @property
    def style_rates(self) -> dict[str, tuple[int, int]]:
        """`(events, opportunities)` per statistic over the learner's recent hands:
        how the *model* plays, every seat it sat in pooled."""
        return self._style.rates(STYLE_PLAYER)

    @property
    def style_by_group(self) -> dict[str, tuple[int, dict[str, tuple[int, int]]]]:
        """`{group: (hands, rates)}` like `style_hands`/`style_rates`, one entry per group
        of table sizes the learner has played at (`style_log.SIZE_GROUPS`)."""
        return {
            group: (tracker.hands(STYLE_PLAYER), tracker.rates(STYLE_PLAYER))
            for group, tracker in self._style_by_group.items()
            if tracker.hands(STYLE_PLAYER)
        }

    def _record(self, record: DecisionRecord) -> None:
        self._pending.setdefault(record.seat, []).append(record)

    def _open_table(self, slot: int) -> _TableSession:
        """A new table for `slot`: a size, who sits at each seat, and blank statistics.

        At least one seat is always the learner, so a hand can never yield no training
        data; every other seat is a pool model with probability `opponent_probability`,
        else another copy of the learner. They stay for `table_hands` hands.
        """
        num_players = self._mix.draw_size(self._rng)
        learner_seat = self._rng.randrange(num_players)
        occupants: list[Opponent | None] = []
        for seat in range(num_players):
            opponent = None
            if (
                seat != learner_seat
                and self._pool is not None
                and self._rng.random() < self._opponent_probability
            ):
                opponent = self._pool.sample(self._rng)
            occupants.append(opponent)
        self._bank.reset_stats(num_players, slot)
        proxies = self._bank.seats(num_players, slot)
        players: list[Player] = []
        for seat, (proxy, opponent) in enumerate(zip(proxies, occupants)):
            if opponent is None:
                players.append(self._learners[seat])
                continue
            # A style is drawn per seat and kept for the table's hands, so the statistics
            # the learner reads fill in over a player who stays the same.
            style = self._styles.draw(self._rng)
            self.opponent_seats += 1
            if style is None:
                players.append(opponent.factory(proxy.player_id, opponent.label, self._rng))
            else:
                self.styled_seats += 1
                players.append(opponent.factory(proxy.player_id, opponent.label, self._rng, style))
        return _TableSession(num_players, occupants, players, self._table_hands)

    def _seat(self, session: _TableSession, proxies: list[SeatProxy]) -> set[int]:
        """Put this table's players in their seats. Returns the seats the learner took."""
        learner_seats: set[int] = set()
        for seat, (proxy, opponent) in enumerate(zip(proxies, session.occupants)):
            proxy.inner = session.players[seat]
            if opponent is None:
                proxy.name = f"RL{seat}"
                learner_seats.add(seat)
            else:
                proxy.name = opponent.label
        return learner_seats

    def collect(
        self, num_hands: int, *, gamma: float = 1.0, lam: float = 0.95
    ) -> list[HandTrajectory]:
        trajectories: list[HandTrajectory] = []
        for _ in range(num_hands):
            # Tables are played in turn, hand by hand; one that has used up its hands
            # is replaced by a new table with new opponents and blank statistics.
            slot = self._turn % len(self._sessions)
            self._turn += 1
            session = self._sessions[slot]
            if session is None or session.hands_left <= 0:
                session = self._sessions[slot] = self._open_table(slot)
            session.hands_left -= 1
            num_players = session.num_players
            proxies = self._bank.seats(num_players, slot)
            learner_seats = self._seat(session, proxies)
            self._pending = {}
            before, after = self._bank.play_hand(num_players, slot)
            self._record_style(learner_seats)
            expected = (
                expected_deltas(self._bank.last_hand, self._runout_rng, self._allin_runouts)
                if self._pending and self._bank.last_hand is not None
                else None
            )

            for seat, decisions in self._pending.items():
                reward = (after[seat] - before[seat]) / self._mix.big_blind
                trained = expected[seat] / self._mix.big_blind if expected is not None else reward
                decisions[-1].reward = trained * self._reward_scale
                others = [stack for other, stack in enumerate(before) if other != seat]
                trajectory = HandTrajectory(
                    seat=seat,
                    player_id=proxies[seat].player_id,
                    decisions=decisions,
                    reward=reward,
                    num_players=num_players,
                    effective_stack_bb=min(before[seat], max(others)) / self._mix.big_blind,
                )
                trajectories.append(trajectory)
        self._fill_critic(trajectories)
        for trajectory in trajectories:
            trajectory.advantages, trajectory.returns = compute_gae(trajectory, gamma=gamma, lam=lam)
        return trajectories

    def _fill_critic(self, trajectories: list[HandTrajectory]) -> None:
        """Give every decision what the critic reads and the value it makes of it.

        Done for all the hands together, after they are played, so the callables run on
        large batches instead of a few decisions at a time; the policy's own value, which
        the model does not compute while playing, is replaced by the critic's."""
        decisions = [decision for trajectory in trajectories for decision in trajectory.decisions]
        for start in range(0, len(decisions), CRITIC_CHUNK):
            chunk = decisions[start : start + CRITIC_CHUNK]
            views = []
            for decision in chunk:
                if decision.critic_view is None:
                    raise ValueError("a decision reached the critic without the cards it needs")
                views.append(decision.critic_view)
            extras = self._critic.equity(views)
            values = self._critic.value([decision.features for decision in chunk], extras)
            for decision, extra, value in zip(chunk, extras, values, strict=True):
                decision.critic_extra = extra
                decision.value = value
