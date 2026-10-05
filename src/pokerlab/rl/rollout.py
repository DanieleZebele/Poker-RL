"""Turning played hands into PPO training data.

There is no Gym-style `step()` in this path on purpose. `Table` drives a hand
synchronously and `RLAgentPlayer` reports every decision through its
`on_decision` hook, so a trajectory falls out of ordinary play with no threads
and no re-implementation of the betting loop. `rl/env.py`'s `PokerEnv` exists
for debugging and Gym compatibility, not for training throughput.

Pure Python, like the rest of the encoding path: numpy and torch only earn their
place in `rl/ppo.py`, where trajectories become tensors.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.history import HandHistory
from pokerlab.engine.stats import StatsTracker, analyse_hand
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import DecisionRecord, PolicyFn, RLAgentPlayer
from pokerlab.rl.table_mix import TableMix

# How the learner plays, over its last `STYLE_WINDOW` seat-hands: every seat it
# took in a hand adds one, so a table of several copies of the learner adds
# several. ~2 per hand at the default mixture, so about nine thousand hands --
# long enough that the rare statistics (3-bet, fold to c-bet) have a few hundred
# opportunities, short enough to follow the policy as it changes.
STYLE_WINDOW = 20_000
STYLE_PLAYER = "learner"


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
    """Wrap a frozen policy (a previously trained model) as an opponent."""

    def factory(player_id: str, name: str, rng: random.Random) -> Player:
        return RLAgentPlayer(
            player_id,
            name,
            policy_fn=policy_fn,
            big_blind=mix.big_blind,
            starting_stack=mix.starting_stack,
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
    (`duel_power`) rely on.
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
        self._tables: dict[int, tuple[Table, list[SeatProxy]]] = {}
        # The hand just played, for a caller that wants more than the stacks.
        self.last_hand: HandHistory | None = None

    def _entry(self, num_players: int) -> tuple[Table, list[SeatProxy]]:
        entry = self._tables.get(num_players)
        if entry is None:
            proxies = [SeatProxy(f"s{seat}", f"S{seat}") for seat in range(num_players)]
            entry = (Table(self.mix.config(num_players), list(proxies), rng=self._rng), proxies)
            self._tables[num_players] = entry
        return entry

    def seats(self, num_players: int) -> list[SeatProxy]:
        """The proxies of the `num_players` table, for the caller to seat."""
        return self._entry(num_players)[1]

    def play_hand(self, num_players: int) -> tuple[list[int], list[int]]:
        """One hand from freshly drawn stacks: `(stacks before, stacks after)`."""
        table, _proxies = self._entry(num_players)
        before = self.mix.draw_stacks(self._stack_rng, num_players)
        table.stacks = list(before)
        self.last_hand = table.play_hand().hand_history
        return before, list(table.stacks)

    def play_session(self, num_players: int, hands: int) -> list[int]:
        """`hands` hands at one table size: the chip delta of every seat."""
        deltas = [0] * num_players
        for _ in range(hands):
            before, after = self.play_hand(num_players)
            for seat in range(num_players):
                deltas[seat] += after[seat] - before[seat]
        return deltas


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
    ) -> None:
        self._mix = mix
        self._reward_scale = reward_scale
        self._pool = opponent_pool
        self._opponent_probability = opponent_probability
        self._rng = rng if rng is not None else random.Random()
        self._pending: dict[int, list[DecisionRecord]] = {}
        self._style = StatsTracker(window=STYLE_WINDOW)

        self._bank = TableBank(mix, self._rng)
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
        for seat in learner_seats:
            if seat in hand.starting_stacks:
                self._style.add(STYLE_PLAYER, counts[f"s{seat}"])

    @property
    def style_hands(self) -> int:
        """Learner-seat hands behind `style_rates` (it fills up to `STYLE_WINDOW`)."""
        return self._style.hands(STYLE_PLAYER)

    @property
    def style_rates(self) -> dict[str, tuple[int, int]]:
        """`(events, opportunities)` per statistic over the learner's recent hands:
        how the *model* plays, every seat it sat in pooled."""
        return self._style.rates(STYLE_PLAYER)

    def _record(self, record: DecisionRecord) -> None:
        self._pending.setdefault(record.seat, []).append(record)

    def _seat_occupants(self, proxies: list[SeatProxy]) -> set[int]:
        """Assign this hand's occupants, keeping at least one learner seat.

        Returns the seats the learner took.
        """
        learner_seats: set[int] = set()
        learner_seat = self._rng.randrange(len(proxies))
        for seat, proxy in enumerate(proxies):
            opponent = None
            if (
                seat != learner_seat
                and self._pool is not None
                and self._rng.random() < self._opponent_probability
            ):
                opponent = self._pool.sample(self._rng)
            if opponent is None:
                proxy.inner = self._learners[seat]
                proxy.name = f"RL{seat}"
                learner_seats.add(seat)
            else:
                proxy.inner = opponent.factory(proxy.player_id, opponent.label, self._rng)
                proxy.name = opponent.label
        return learner_seats

    def collect(
        self, num_hands: int, *, gamma: float = 1.0, lam: float = 0.95
    ) -> list[HandTrajectory]:
        trajectories: list[HandTrajectory] = []
        for _ in range(num_hands):
            num_players = self._mix.draw_size(self._rng)
            proxies = self._bank.seats(num_players)
            learner_seats = self._seat_occupants(proxies)
            self._pending = {}
            before, after = self._bank.play_hand(num_players)
            self._record_style(learner_seats)

            for seat, decisions in self._pending.items():
                reward = (after[seat] - before[seat]) / self._mix.big_blind
                decisions[-1].reward = reward * self._reward_scale
                others = [stack for other, stack in enumerate(before) if other != seat]
                trajectory = HandTrajectory(
                    seat=seat,
                    player_id=proxies[seat].player_id,
                    decisions=decisions,
                    reward=reward,
                    num_players=num_players,
                    effective_stack_bb=min(before[seat], max(others)) / self._mix.big_blind,
                )
                trajectory.advantages, trajectory.returns = compute_gae(
                    trajectory, gamma=gamma, lam=lam
                )
                trajectories.append(trajectory)
        return trajectories
