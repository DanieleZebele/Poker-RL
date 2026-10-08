"""A Player driven by a learned policy.

Deliberately torch-free: it takes a callable, not a model. That keeps the core
install working without the `rl` extra, lets `players/__init__.py` import it
eagerly like every other player, and means the same class serves a real network,
a uniform-random baseline in tests, and a replayed checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.state import PlayerStatus
from pokerlab.players.base import Observation, Player
from pokerlab.rl.action_space import action_index_to_action, legal_action_mask
from pokerlab.rl.features import card_index, chips_at_stake, encode_observation


@dataclass(frozen=True)
class PolicyDecision:
    """What a policy returns for one decision. `log_prob` and `value` are what
    PPO needs later and are ignored when simply playing."""

    action_index: int
    log_prob: float = 0.0
    value: float = 0.0


@dataclass(frozen=True)
class DealView:
    """What a critic may know about one decision and the player itself may not.

    `holes` has one entry per seat at the table, in the order the observation's features
    use (slot 0 is the deciding player, then the seats clockwise): the two cards of a seat
    still in the hand as indices 0..51 (`rl.features.card_index`), or None for a seat that
    has folded. `board` is the visible community cards. Collected only while training, from
    the table itself; a published model never sees one."""

    holes: tuple[tuple[int, int] | None, ...]
    board: tuple[int, ...]


def deal_view(observation: Observation, hole_cards: dict[int, tuple[Card, Card]]) -> DealView:
    """The `DealView` of a decision, given every seat's hole cards."""
    seats = list(observation.seats)
    mine = next(i for i, seat in enumerate(seats) if seat.seat == observation.my_seat)
    rotated = seats[mine:] + seats[:mine]
    holes = tuple(
        None
        if seat.status is PlayerStatus.FOLDED
        else (card_index(hole_cards[seat.seat][0]), card_index(hole_cards[seat.seat][1]))
        for seat in rotated
    )
    return DealView(holes, tuple(card_index(card) for card in observation.community_cards))


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
    # What this hand can still move for this seat, in big blinds (`chips_at_stake`): what
    # the critic's loss weighs the decision by (`rl/ppo.py`), since the spread of the
    # value target grows with it.
    stake_bb: float
    # The *training* reward, in whatever unit the critic predicts (see
    # SelfPlayCollector's reward_scale) -- not necessarily big blinds. Zero on
    # every decision but the last of a hand: the reward is terminal.
    reward: float = field(default=0.0)
    # Only in training, where the critic reads more than the policy does (`rl/policy.py`):
    # what the table knew at this decision, and then the numbers the critic is given in
    # its place, which the collector fills in from it once the hands are played.
    critic_view: DealView | None = None
    critic_extra: list[float] | None = None


# `(features, mask)`, plus `(bias, temperature)` for a player with a style (`rl/styles.py`):
# a policy that never seats a styled opponent need only take the first two.
PolicyFn = Callable[..., PolicyDecision]


class RLAgentPlayer(Player):
    """Wraps a policy's forward pass as a `Player`'s decision function.

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
        deal_holes: Callable[[], dict[int, tuple[Card, Card]]] | None = None,
        style: Callable[[Observation], tuple[list[float], float]] | None = None,
    ) -> None:
        super().__init__(player_id, name)
        self._policy_fn = policy_fn
        self._big_blind = big_blind
        self._starting_stack = starting_stack
        self._on_decision = on_decision
        # Where to read every seat's hole cards from, when the critic is given more than
        # the policy sees: a training collector supplies it, a playing seat never does.
        self._deal_holes = deal_holes
        # An opponent's style (`rl/styles.py`): from the state, a push on the logits and a
        # temperature, handed to the policy. A learner never has one.
        self._style = style

    def _normalisers(self, observation: Observation) -> tuple[int, float]:
        """The big blind and the starting stack the features are divided by in this hand.

        They are the ones the player was built with, unless the blinds have gone up: then the
        stack constant follows the blind, so the model keeps reading stacks and bets in big
        blinds -- the unit it was trained in -- whatever the blind is at the moment."""
        big_blind = observation.big_blind
        if not big_blind or big_blind == self._big_blind:
            return self._big_blind, self._starting_stack
        return big_blind, self._starting_stack * big_blind / self._big_blind

    def action_probabilities(
        self, observation: Observation, legal_actions: list[LegalAction]
    ) -> list[float] | None:
        """The policy's probability of each action bin in this decision -- what `act` samples
        from, for a spectator that wants to see it. None if the policy cannot say (a plain
        callable with no `distribution`). It consumes no randomness, so asking changes nothing
        about how the game goes."""
        distribution = getattr(self._policy_fn, "distribution", None)
        if distribution is None:
            return None
        mask = legal_action_mask(observation, legal_actions)
        big_blind, starting_stack = self._normalisers(observation)
        features = encode_observation(
            observation, big_blind=big_blind, starting_stack=starting_stack, legal_mask=mask
        )
        if self._style is None:
            return distribution(features, mask)
        bias, temperature = self._style(observation)
        return distribution(features, mask, bias, temperature)

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        mask = legal_action_mask(observation, legal_actions)
        big_blind, starting_stack = self._normalisers(observation)
        features = encode_observation(
            observation,
            big_blind=big_blind,
            starting_stack=starting_stack,
            legal_mask=mask,
        )
        if self._style is None:
            decision = self._policy_fn(features, mask)
        else:
            bias, temperature = self._style(observation)
            decision = self._policy_fn(features, mask, bias, temperature)
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
                    stake_bb=chips_at_stake(observation) / big_blind,
                    critic_view=(
                        deal_view(observation, self._deal_holes())
                        if self._deal_holes is not None
                        else None
                    ),
                )
            )
        return action_index_to_action(decision.action_index, observation, legal_actions)
