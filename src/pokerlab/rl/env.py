"""A Gym/PettingZoo-shaped wrapper around `pokerlab.engine.table.Table`.

Why this needed no engine change: `pokerlab.players.base.Player` already talks
to the engine purely through `act(observation, legal_actions) -> Action`, using
plain, JSON-friendly dataclasses. `PokerEnv` sits on the other side of that same
boundary -- it encodes an Observation into a feature vector, turns legal_actions
into an action mask, and shapes a reward from chip deltas.

**This is not the training path.** `Table` drives a hand synchronously, so
handing control back to a caller between decisions costs a background thread and
two context switches per action. `rl/rollout.py` collects PPO trajectories
straight through `Player.act()` instead -- no threads, no duplicated betting
logic -- and is what self-play should use. `TablePokerEnv` exists for debugging,
evaluation, and interoperating with code that expects the Gym shape.

Poker is sequential, so this follows the PettingZoo agent-environment-cycle
shape: exactly one agent is asked to act at a time, and an episode is one hand.
"""

from __future__ import annotations

import queue
import random
import threading
from abc import ABC, abstractmethod
from typing import Any, Self

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.rl.action_space import action_index_to_action, legal_action_mask
from pokerlab.rl.features import encode_observation


class PokerEnv(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Observation]: ...

    @abstractmethod
    def step(
        self, actions: dict[str, Action]
    ) -> tuple[dict[str, Observation], dict[str, float], dict[str, bool], dict[str, dict[str, Any]]]:
        ...

    @abstractmethod
    def legal_action_mask(self, agent_id: str) -> list[bool]: ...

    @abstractmethod
    def encode_observation(self, observation: Observation) -> list[float]: ...


class _Aborted(Exception):
    """Raised inside the worker thread to unwind a hand abandoned by close()."""


class _QueuePlayer(Player):
    def __init__(self, player_id: str, name: str, env: TablePokerEnv) -> None:
        super().__init__(player_id, name)
        self._env = env

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self._env._requests.put(("act", (observation, legal_actions)))
        kind, payload = self._env._decisions.get()
        if kind == "abort":
            raise _Aborted
        return payload


class TablePokerEnv(PokerEnv):
    """One hand per episode, every seat exposed as its own agent."""

    def __init__(self, config: GameConfig, rng: random.Random | None = None) -> None:
        self._config = config
        self._rng = rng if rng is not None else random.Random()
        self._requests: queue.Queue = queue.Queue()
        self._decisions: queue.Queue = queue.Queue()
        self._players = [
            _QueuePlayer(f"p{seat}", f"P{seat}", self) for seat in range(config.num_players)
        ]
        self._table = Table(config, self._players, rng=self._rng)
        self._agent_ids = [player.player_id for player in self._players]
        self._thread: threading.Thread | None = None
        self._pending: tuple[Observation, list[LegalAction]] | None = None
        self._stacks_before: list[int] = []
        self._done = True

    def _agent_id(self, seat: int) -> str:
        return self._agent_ids[seat]

    def _run_hand(self) -> None:
        try:
            self._table.play_hand()
        except _Aborted:
            return
        # Deliberately blind: a worker thread that dies quietly would park the
        # caller on an empty queue forever, so every failure is forwarded.
        except Exception as exc:  # noqa: BLE001
            self._requests.put(("error", exc))
            return
        self._requests.put(("done", None))

    def _pump(self) -> Observation | None:
        """Block until the worker either asks for a decision or finishes."""
        kind, payload = self._requests.get()
        if kind == "error":
            self._done = True
            self._pending = None
            raise payload
        if kind == "done":
            self._done = True
            self._pending = None
            return None
        self._pending = payload
        return payload[0]

    def _terminal_rewards(self) -> dict[str, float]:
        return {
            self._agent_id(seat): (stack - self._stacks_before[seat]) / self._config.big_blind
            for seat, stack in enumerate(self._table.stacks)
        }

    def reset(self, seed: int | None = None) -> dict[str, Observation]:
        self.close()
        if seed is not None:
            self._rng.seed(seed)
        self._table.stacks = [self._config.starting_stack] * self._config.num_players
        self._stacks_before = list(self._table.stacks)
        self._done = False
        self._thread = threading.Thread(target=self._run_hand, daemon=True)
        self._thread.start()

        observation = self._pump()
        return {} if observation is None else {self._agent_id(observation.my_seat): observation}

    def step(
        self, actions: dict[str, Action]
    ) -> tuple[dict[str, Observation], dict[str, float], dict[str, bool], dict[str, dict[str, Any]]]:
        if self._pending is None:
            raise RuntimeError("no agent is waiting to act; call reset() first")
        agent_id = self._agent_id(self._pending[0].my_seat)
        if agent_id not in actions:
            raise KeyError(f"expected an action for {agent_id}, got {sorted(actions)}")

        self._decisions.put(("action", actions[agent_id]))
        observation = self._pump()

        if observation is None:
            rewards = self._terminal_rewards()
            dones = dict.fromkeys(self._agent_ids, True)
            return {}, rewards, dones, {agent: {} for agent in self._agent_ids}

        next_agent = self._agent_id(observation.my_seat)
        return (
            {next_agent: observation},
            {next_agent: 0.0},  # the reward is terminal-only
            {next_agent: False},
            {next_agent: {}},
        )

    def _pending_for(self, agent_id: str) -> tuple[Observation, list[LegalAction]]:
        if self._pending is None:
            raise RuntimeError("no agent is waiting to act")
        observation, legal_actions = self._pending
        if self._agent_id(observation.my_seat) != agent_id:
            raise KeyError(f"{agent_id} is not the agent to act")
        return observation, legal_actions

    def legal_action_mask(self, agent_id: str) -> list[bool]:
        return legal_action_mask(*self._pending_for(agent_id))

    def action_from_index(self, agent_id: str, action_index: int) -> Action:
        """Map a sampled action bin back to the `Action` `step()` expects."""
        observation, legal_actions = self._pending_for(agent_id)
        return action_index_to_action(action_index, observation, legal_actions)

    def encode_observation(self, observation: Observation) -> list[float]:
        """Encode using the pending agent's mask, the only one the env knows."""
        if self._pending is None:
            raise RuntimeError("no agent is waiting to act")
        return encode_observation(
            observation,
            big_blind=self._config.big_blind,
            starting_stack=self._config.starting_stack,
            legal_mask=legal_action_mask(*self._pending),
        )

    def close(self, timeout: float = 5.0) -> None:
        """Abandon any hand in flight and join the worker.

        Mandatory between episodes: the worker blocks inside `Player.act()`, so
        without the abort sentinel every reset() would leak a parked thread.
        """
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            self._decisions.put(("abort", None))
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("the table worker thread did not stop")

        # Safe to drain only now: the worker is dead and cannot put anything else.
        for pipe in (self._requests, self._decisions):
            while not pipe.empty():
                pipe.get_nowait()
        self._pending = None
        self._done = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
