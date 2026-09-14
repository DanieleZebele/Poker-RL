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
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import DecisionRecord, PolicyFn, RLAgentPlayer
from pokerlab.players.scripted import get_bot_profile


@dataclass
class HandTrajectory:
    """One seat's decisions in one hand, plus the terminal reward.

    The episode is a single hand: stacks persist in `Table`, but `Observation`
    has no memory of previous hands, so an episode spanning several hands would
    not be Markov with respect to the features.
    """

    seat: int
    player_id: str
    decisions: list[DecisionRecord]
    reward: float  # chip delta over the hand, always in big blinds
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
    """A non-learner seat filler, in `BOT_CATALOG`'s factory shape."""

    label: str
    factory: Callable[[str, str, random.Random], Player]


def scripted_opponent(key: str) -> Opponent:
    profile = get_bot_profile(key)
    return Opponent(label=profile.label, factory=profile.factory)


def policy_opponent(label: str, policy_fn: PolicyFn, config: GameConfig) -> Opponent:
    """Wrap a frozen policy (typically a past snapshot) as an opponent."""

    def factory(player_id: str, name: str, rng: random.Random) -> Player:
        return RLAgentPlayer(
            player_id,
            name,
            policy_fn=policy_fn,
            big_blind=config.big_blind,
            starting_stack=config.starting_stack,
        )

    return Opponent(label=label, factory=factory)


class OpponentPool:
    """Who fills the seats the learner is not sitting in.

    Self-play against nothing but the current policy can chase its own tail:
    the reward is relative, so both sides can drift without either getting
    stronger, and the agent learns exploits that only work against its twin.
    Past snapshots and the scripted catalog anchor it to something that does
    not move with it.
    """

    def __init__(self, opponents: Sequence[Opponent] = (), *, max_snapshots: int = 5) -> None:
        self._fixed = list(opponents)
        self._snapshots: deque[Opponent] = deque(maxlen=max_snapshots)

    def add_snapshot(self, opponent: Opponent) -> None:
        self._snapshots.append(opponent)

    def sample(self, rng: random.Random) -> Opponent | None:
        """An opponent for one seat, or None to seat the learner there."""
        candidates = [*self._fixed, *self._snapshots]
        return rng.choice(candidates) if candidates else None

    def __len__(self) -> int:
        return len(self._fixed) + len(self._snapshots)


class _SeatProxy(Player):
    """Delegates to whoever occupies the seat this hand.

    Swapping occupants this way keeps one `Table` alive across hands, so the
    button keeps rotating and stacks persist; rebuilding the table per hand
    would reset both.
    """

    def __init__(self, player_id: str, name: str) -> None:
        super().__init__(player_id, name)
        self.inner: Player | None = None

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        assert self.inner is not None, "seat was never assigned an occupant"
        return self.inner.act(observation, legal_actions)


class SelfPlayCollector:
    """Plays hands and collects the learner's trajectories.

    Stacks are reset between hands (`rebuy=True`) so the agent trains on a
    stationary stack distribution instead of drifting into short-stack and
    busted-seat states; the button still rotates, so positions stay uniform.

    With no `opponent_pool`, every seat is the learner -- plain self-play.

    `reward_scale` multiplies the big-blind chip delta to produce the *training*
    reward the critic has to predict. Leaving it at 1.0 means value targets span
    roughly +/- (starting_stack / big_blind) -- 100 for a 200-chip stack at 2/1 --
    so the squared value loss lands around 10,000 and swamps a policy loss of
    order 0.01. `big_blind / starting_stack` puts a stack-sized swing at 1.0.
    `HandTrajectory.reward` stays in big blinds regardless, for reporting.
    """

    def __init__(
        self,
        config: GameConfig,
        policy_fn: PolicyFn,
        *,
        rng: random.Random | None = None,
        rebuy: bool = True,
        opponent_pool: OpponentPool | None = None,
        opponent_probability: float = 0.5,
        reward_scale: float = 1.0,
    ) -> None:
        self._config = config
        self._rebuy = rebuy
        self._reward_scale = reward_scale
        self._pool = opponent_pool
        self._opponent_probability = opponent_probability
        self._rng = rng if rng is not None else random.Random()
        self._pending: dict[int, list[DecisionRecord]] = {}

        self._proxies = [_SeatProxy(f"s{seat}", f"S{seat}") for seat in range(config.num_players)]
        self._learners = [
            RLAgentPlayer(
                f"s{seat}",
                f"RL{seat}",
                policy_fn=policy_fn,
                big_blind=config.big_blind,
                starting_stack=config.starting_stack,
                on_decision=self._record,
            )
            for seat in range(config.num_players)
        ]
        self._table = Table(config, list(self._proxies), rng=self._rng)

    def _record(self, record: DecisionRecord) -> None:
        self._pending.setdefault(record.seat, []).append(record)

    def _seat_occupants(self) -> None:
        """Assign this hand's occupants, keeping at least one learner seat."""
        learner_seat = self._rng.randrange(self._config.num_players)
        for seat, proxy in enumerate(self._proxies):
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
            else:
                proxy.inner = opponent.factory(proxy.player_id, opponent.label, self._rng)
                proxy.name = opponent.label

    def collect(
        self, num_hands: int, *, gamma: float = 1.0, lam: float = 0.95
    ) -> list[HandTrajectory]:
        trajectories: list[HandTrajectory] = []
        for _ in range(num_hands):
            if sum(1 for stack in self._table.stacks if stack > 0) < 2:
                break
            self._seat_occupants()
            stacks_before = list(self._table.stacks)
            self._pending = {}
            self._table.play_hand()

            for seat, decisions in self._pending.items():
                reward = (self._table.stacks[seat] - stacks_before[seat]) / self._config.big_blind
                decisions[-1].reward = reward * self._reward_scale
                trajectory = HandTrajectory(
                    seat=seat,
                    player_id=self._proxies[seat].player_id,
                    decisions=decisions,
                    reward=reward,
                )
                trajectory.advantages, trajectory.returns = compute_gae(
                    trajectory, gamma=gamma, lam=lam
                )
                trajectories.append(trajectory)

            if self._rebuy:
                self._table.stacks = [self._config.starting_stack] * self._config.num_players
        return trajectories
