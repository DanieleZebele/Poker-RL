"""A Player driven by a learned policy.

Deliberately torch-free: it takes a callable, not a model. That keeps the core
install working without the `rl` extra, lets `players/__init__.py` import it
eagerly like every other player, and means the same class serves a real network,
a uniform-random baseline in tests, and a replayed checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.players.base import Observation, Player
from pokerlab.rl.action_space import action_index_to_action, legal_action_mask
from pokerlab.rl.features import encode_observation


@dataclass(frozen=True)
class PolicyDecision:
    """What a policy returns for one decision. `log_prob` and `value` are what
    PPO needs later and are ignored when simply playing."""

    action_index: int
    log_prob: float = 0.0
    value: float = 0.0


@dataclass
class DecisionRecord:
    """One `(state, action)` pair as the policy saw it, ready to become a
    training sample once the hand ends and supplies the reward."""

    player_id: str
    seat: int
    features: list[float]
    legal_mask: list[bool]
    action_index: int
    log_prob: float
    value: float
    # The *training* reward, in whatever unit the critic predicts (see
    # SelfPlayCollector's reward_scale) -- not necessarily big blinds. Zero on
    # every decision but the last of a hand: the reward is terminal.
    reward: float = field(default=0.0)


PolicyFn = Callable[[list[float], list[bool]], PolicyDecision]


class RLAgentPlayer(Player):
    """Wraps a policy's forward pass the way `ScriptedBot` wraps a rule.

    `big_blind` and `starting_stack` are the feature normalisation constants;
    they come from the table's `GameConfig`, since `Observation` carries no
    table configuration of its own.

    `on_decision` is the trajectory-collection hook. It is why training needs no
    Gym-style `step()`: `Table` drives the hand synchronously and every decision
    is reported here as it happens, with the reward filled in at hand end.
    """

    def __init__(
        self,
        player_id: str,
        name: str,
        *,
        policy_fn: PolicyFn,
        big_blind: int,
        starting_stack: int,
        on_decision: Callable[[DecisionRecord], None] | None = None,
    ) -> None:
        super().__init__(player_id, name)
        self._policy_fn = policy_fn
        self._big_blind = big_blind
        self._starting_stack = starting_stack
        self._on_decision = on_decision

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        mask = legal_action_mask(observation, legal_actions)
        features = encode_observation(
            observation,
            big_blind=self._big_blind,
            starting_stack=self._starting_stack,
            legal_mask=mask,
        )
        decision = self._policy_fn(features, mask)
        if not mask[decision.action_index]:
            raise ValueError(
                f"policy chose masked action bin {decision.action_index}; mask was {mask}"
            )
        if self._on_decision is not None:
            self._on_decision(
                DecisionRecord(
                    player_id=self.player_id,
                    seat=observation.my_seat,
                    features=features,
                    legal_mask=mask,
                    action_index=decision.action_index,
                    log_prob=decision.log_prob,
                    value=decision.value,
                )
            )
        return action_index_to_action(decision.action_index, observation, legal_actions)
