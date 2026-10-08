"""Minimal test-only players: cheap, deterministic opponents for fuzzing
engine correctness (chip conservation, action legality).

Nothing here is product-facing -- it exists purely so engine-level tests have
a cheap opponent to play against.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.players.base import Observation, Player
from pokerlab.rl.table_mix import SIZES, TableMix

Strategy = Callable[[Observation, list[LegalAction]], Action]


class _StrategyPlayer(Player):
    def __init__(self, player_id: str, name: str, strategy: Strategy) -> None:
        super().__init__(player_id, name)
        self._strategy = strategy

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        return self._strategy(observation, legal_actions)


def make_always_call_bot(player_id: str, name: str = "AlwaysCallBot") -> Player:
    """Never voluntarily folds: checks or calls whenever possible, and goes
    all-in rather than fold if it can't afford a full call."""

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        types = {la.action_type for la in legal_actions}
        if ActionType.CHECK in types:
            return Action(ActionType.CHECK)
        if ActionType.CALL in types:
            return Action(ActionType.CALL)
        if ActionType.ALL_IN in types:
            return Action(ActionType.ALL_IN)
        return Action(ActionType.FOLD)

    return _StrategyPlayer(player_id, name, strategy)


def make_random_legal_bot(
    player_id: str, name: str = "RandomBot", rng: random.Random | None = None
) -> Player:
    """Picks uniformly among whatever is currently legal; useful for
    fuzz-testing the engine (see test_full_hand_flow.py)."""
    rng = rng if rng is not None else random.Random()

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        choice = rng.choice(legal_actions)
        if choice.action_type in (ActionType.BET, ActionType.RAISE):
            assert choice.min_amount is not None and choice.max_amount is not None
            amount = rng.randint(choice.min_amount, choice.max_amount)
            return Action(choice.action_type, amount=amount)
        return Action(choice.action_type)

    return _StrategyPlayer(player_id, name, strategy)


def fixed_mix(
    num_players: int, *, stack_bb: float = 50.0, small_blind: int = 1, big_blind: int = 2
) -> TableMix:
    """A `TableMix` that always draws the same table: one size, one stack.

    The defaults are the old fixed test table (100 chips at 1/2), so a test that
    wants the mixture's variety builds a `TableMix` itself.
    """
    weights = tuple(1.0 if size == num_players else 0.0 for size in SIZES)
    return TableMix(
        weights=weights,
        stack_min_bb=stack_bb,
        stack_max_bb=stack_bb,
        small_blind=small_blind,
        big_blind=big_blind,
    )


# The smallest equity encoder the tests build every network on: a model cannot exist without
# one. Torch is imported inside the helpers, so the torch-free tests can still import this file.
TINY_EQUITY = {"hidden": 8, "latent": 4, "blocks": 1}


def tiny_model(**shape):
    """A `PokerActorCritic` of the given shape on a randomly initialised `TINY_EQUITY` encoder."""
    from pokerlab.rl.policy import PokerActorCritic

    return PokerActorCritic(**shape, equity=TINY_EQUITY)


def tiny_critic(model):
    """The critic callables of `model`: an equity network whose encoder is the model's own."""
    from pokerlab.rl.equity_net import EquityNet
    from pokerlab.rl.policy import make_critic_fns

    net = EquityNet(**TINY_EQUITY, arch="sets")
    net.phi.load_state_dict(model.equity_encoder.phi.state_dict())
    return make_critic_fns(model, net.requires_grad_(False).eval())


def tiny_parts(**shape) -> dict:
    """`model=` and `critic=` for a `SelfPlayTrainer`."""
    model = tiny_model(**shape)
    return {"model": model, "critic": tiny_critic(model)}


def fake_critic():
    """A critic with no network behind it, for tests of the collector that need no torch: every
    seat's equity is a quarter and a decision's value is their sum."""
    from pokerlab.players.rl_agent import DealView  # noqa: F401  (documents what the views are)
    from pokerlab.rl.rollout import CriticFns

    return CriticFns(
        equity=lambda views: [[0.25] * 9 for _ in views],
        value=lambda features, extras: [float(sum(extra)) for extra in extras],
    )


def fake_collector(*args, **kwargs):
    """A `SelfPlayCollector` with `fake_critic()` unless the test brings its own critic."""
    from pokerlab.rl.rollout import SelfPlayCollector

    kwargs.setdefault("critic", fake_critic())
    return SelfPlayCollector(*args, **kwargs)
