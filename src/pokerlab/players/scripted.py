from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import PlayerStatus
from pokerlab.evaluator.evaluator import HandCategory, evaluate
from pokerlab.players.base import Observation, Player

Strategy = Callable[[Observation, list[LegalAction]], Action]


class ScriptedBot(Player):
    """A Player whose decisions come from a plain function. This is exactly
    the shape a future RLAgentPlayer will also have -- it just wraps a
    model's forward pass instead of a hand-coded rule -- so the engine
    never needs to change to support it."""

    def __init__(self, player_id: str, name: str, strategy: Strategy) -> None:
        super().__init__(player_id, name)
        self._strategy = strategy

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        return self._strategy(observation, legal_actions)


def _legal_types(legal_actions: list[LegalAction]) -> set[ActionType]:
    return {la.action_type for la in legal_actions}


def _find(legal_actions: list[LegalAction], action_type: ActionType) -> LegalAction:
    return next(la for la in legal_actions if la.action_type == action_type)


def make_always_call_bot(player_id: str, name: str = "AlwaysCallBot") -> ScriptedBot:
    """Never voluntarily folds: checks or calls whenever possible, and goes
    all-in rather than fold if it can't afford a full call."""

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        types = _legal_types(legal_actions)
        if ActionType.CHECK in types:
            return Action(ActionType.CHECK)
        if ActionType.CALL in types:
            return Action(ActionType.CALL)
        if ActionType.ALL_IN in types:
            return Action(ActionType.ALL_IN)
        return Action(ActionType.FOLD)

    return ScriptedBot(player_id, name, strategy)


def make_random_legal_bot(player_id: str, name: str = "RandomBot", rng: random.Random | None = None) -> ScriptedBot:
    """Picks uniformly among whatever is currently legal; useful as a chaotic
    opponent and for fuzz-testing the engine (see test_full_hand_flow.py)."""
    rng = rng if rng is not None else random.Random()

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        choice = rng.choice(legal_actions)
        if choice.action_type in (ActionType.BET, ActionType.RAISE):
            assert choice.min_amount is not None and choice.max_amount is not None
            amount = rng.randint(choice.min_amount, choice.max_amount)
            return Action(choice.action_type, amount=amount)
        return Action(choice.action_type)

    return ScriptedBot(player_id, name, strategy)


_CATEGORY_STRENGTH = {
    HandCategory.HIGH_CARD: 0.15,
    HandCategory.PAIR: 0.35,
    HandCategory.TWO_PAIR: 0.55,
    HandCategory.THREE_OF_A_KIND: 0.70,
    HandCategory.STRAIGHT: 0.80,
    HandCategory.FLUSH: 0.85,
    HandCategory.FULL_HOUSE: 0.90,
    HandCategory.FOUR_OF_A_KIND: 0.97,
    HandCategory.STRAIGHT_FLUSH: 1.0,
}


def _preflop_strength(hole: tuple[Card, Card]) -> float:
    """Crude preflop hand-strength heuristic (high ranks / pairs / suited /
    connected score higher). Not a real preflop chart -- a light heuristic
    to give the bot *some* card-awareness, nothing more."""
    r1, r2 = sorted((hole[0].rank.value, hole[1].rank.value), reverse=True)
    score = (r1 + r2) / 28.0
    if r1 == r2:
        score += 0.25
    if hole[0].suit == hole[1].suit:
        score += 0.05
    if r1 != r2 and (r1 - r2) <= 1:
        score += 0.05
    return min(score, 1.0)


def _hand_strength(observation: Observation) -> float:
    if not observation.community_cards:
        return _preflop_strength(observation.hole_cards)
    rank = evaluate((*observation.hole_cards, *observation.community_cards))
    return _CATEGORY_STRENGTH[rank.category]


# Field-size adjustment: these presets are tuned assuming a "typical" pot
# has this many live opponents. More live opponents than that -> your hand
# has to beat more people, so play tighter and bluff less (a bluff only
# works if *everyone* folds, which gets less likely with each extra caller).
# Fewer -> loosen up and bluff more, since heads-up almost any hand and any
# bluff has real value.
_BASELINE_OPPONENTS = 3
_TIGHTNESS_PER_OPPONENT = 0.04
_BLUFF_DECAY_PER_OPPONENT = 0.12


def _active_opponent_count(observation: Observation) -> int:
    """How many other seats are still live (not folded) in this hand right
    now -- shrinks as players fold over the course of a hand, so a bot
    naturally loosens up on a later street once most of the field is gone."""
    return sum(
        1
        for seat in observation.seats
        if seat.seat != observation.my_seat and seat.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
    )


def _field_adjusted_tightness(tightness: float, opponents: int) -> float:
    delta = (opponents - _BASELINE_OPPONENTS) * _TIGHTNESS_PER_OPPONENT
    return min(1.0, max(0.0, tightness + delta))


def _field_adjusted_bluff_frequency(bluff_frequency: float, opponents: int) -> float:
    factor = (1.0 - _BLUFF_DECAY_PER_OPPONENT) ** (opponents - _BASELINE_OPPONENTS)
    return min(1.0, max(0.0, bluff_frequency * factor))


def make_heuristic_bot(
    player_id: str,
    name: str,
    *,
    tightness: float,
    aggression: float,
    bluff_frequency: float,
    size_variance: float,
    rng: random.Random | None = None,
) -> ScriptedBot:
    """General-purpose rule-based strategy, parametrized along four axes so
    a handful of distinct "personalities" (see BOT_CATALOG below) can share
    one implementation instead of duplicating hand-strength logic:

    - tightness (0-1): minimum hand strength to voluntarily play at all
      (higher = folds more hands).
    - aggression (0-1): how much of that playable range is also played as a
      bet/raise rather than a check/call (higher = raises more often, and
      raises bigger relative to the pot).
    - bluff_frequency (0-1): independent per-decision chance of betting or
      raising anyway despite a weak hand -- this is what makes the bot
      occasionally "bluff" rather than being a pure function of hand strength.
    - size_variance (0-1): random jitter applied to bet/raise sizing, so the
      exact same hand strength doesn't always produce the exact same bet.

    `tightness` and `bluff_frequency` are also adjusted per-decision by how
    many opponents are still live in the hand (see _field_adjusted_*): the
    values passed in here are treated as the setting for a "typical" pot
    with a few opponents, tightening/loosening automatically for bigger or
    smaller fields (see module-level docstring above _BASELINE_OPPONENTS).

    None of these settings make for a strong strategy in an absolute sense
    -- they exist to give varied, partly-unpredictable opponents for manual
    testing and (later) RL self-play, not to approximate optimal play.
    """
    rng = rng if rng is not None else random.Random()

    def strategy(observation: Observation, legal_actions: list[LegalAction]) -> Action:
        types = _legal_types(legal_actions)
        strength = _hand_strength(observation)
        opponents = _active_opponent_count(observation)
        effective_tightness = _field_adjusted_tightness(tightness, opponents)
        effective_bluff_frequency = _field_adjusted_bluff_frequency(bluff_frequency, opponents)
        # How much of the playable range (>= effective_tightness) is played
        # aggressively: aggression=1 raises with anything playable;
        # aggression=0 never raises from hand strength alone (only via a bluff roll).
        raise_threshold = effective_tightness + (1 - effective_tightness) * (1 - aggression)

        bluffing = rng.random() < effective_bluff_frequency
        wants_to_play = bluffing or strength >= effective_tightness
        wants_to_raise = bluffing or strength >= raise_threshold

        def sized(la: LegalAction) -> int:
            assert la.min_amount is not None and la.max_amount is not None
            target = la.min_amount + observation.pot_size * (0.3 + aggression)
            jitter = 1.0 + rng.uniform(-size_variance, size_variance)
            return max(la.min_amount, min(la.max_amount, round(target * jitter)))

        if wants_to_raise and ActionType.RAISE in types:
            return Action(ActionType.RAISE, amount=sized(_find(legal_actions, ActionType.RAISE)))
        if wants_to_raise and ActionType.BET in types:
            return Action(ActionType.BET, amount=sized(_find(legal_actions, ActionType.BET)))
        if ActionType.CHECK in types:
            return Action(ActionType.CHECK)
        if wants_to_play and ActionType.CALL in types:
            return Action(ActionType.CALL)
        if ActionType.FOLD in types:
            return Action(ActionType.FOLD)
        return Action(ActionType.ALL_IN)

    return ScriptedBot(player_id, name, strategy)


def make_loose_aggressive_bot(player_id: str, name: str = "ManiacBot", rng: random.Random | None = None) -> ScriptedBot:
    """"Maniac" (LAG): plays almost any hand and raises often, with frequent
    bluffs and wildly varying bet sizes. Unpredictable and hard to read, but
    fundamentally a losing style long-run -- it gives away too much value."""
    return make_heuristic_bot(
        player_id, name, tightness=0.10, aggression=0.85, bluff_frequency=0.35, size_variance=0.6, rng=rng
    )


def make_tight_passive_bot(player_id: str, name: str = "RockBot", rng: random.Random | None = None) -> ScriptedBot:
    """"Rock" (tight-passive): only plays strong hands, and even then mostly
    just calls rather than raises. Disciplined but predictable -- once you
    notice it only bets big with big hands, it's easy to play against."""
    return make_heuristic_bot(
        player_id, name, tightness=0.55, aggression=0.15, bluff_frequency=0.03, size_variance=0.15, rng=rng
    )


def make_tight_aggressive_bot(player_id: str, name: str = "TAGBot", rng: random.Random | None = None) -> ScriptedBot:
    """"Shark" (TAG): plays a disciplined-but-reasonably-wide range and
    backs it with aggression, with just enough bluffing and size variance to
    avoid being an open book. The strongest of the preset heuristics here --
    still just a hand-coded rule of thumb, and exactly the kind of opponent
    a future RL agent should eventually be able to outclass."""
    return make_heuristic_bot(
        player_id, name, tightness=0.45, aggression=0.7, bluff_frequency=0.12, size_variance=0.35, rng=rng
    )


BotFactory = Callable[[str, str, random.Random], Player]


@dataclass(frozen=True)
class BotProfile:
    """One entry in the bot catalog: metadata plus a uniform-signature
    factory, so a caller (e.g. the CLI) can list and pick bots by name
    without knowing each preset function's own signature."""

    key: str
    label: str
    difficulty: int  # 1 = weakest/most predictable ... 5 = strongest
    description: str
    factory: BotFactory


BOT_CATALOG: list[BotProfile] = [
    BotProfile(
        key="random",
        label="Random",
        difficulty=1,
        description="Picks uniformly among legal actions, completely ignoring its cards.",
        factory=lambda pid, name, rng: make_random_legal_bot(pid, name, rng=rng),
    ),
    BotProfile(
        key="calling_station",
        label="CallingStation",
        difficulty=2,
        description="Loose-passive: checks or calls almost everything, never raises, never folds voluntarily.",
        factory=lambda pid, name, rng: make_always_call_bot(pid, name),
    ),
    BotProfile(
        key="maniac",
        label="Maniac",
        difficulty=3,
        description="Loose-aggressive: plays most hands, raises often, bluffs a lot, wildly variable bet sizes.",
        factory=lambda pid, name, rng: make_loose_aggressive_bot(pid, name, rng=rng),
    ),
    BotProfile(
        key="rock",
        label="Rock",
        difficulty=4,
        description="Tight-passive: only plays strong hands, rarely raises even then, very predictable.",
        factory=lambda pid, name, rng: make_tight_passive_bot(pid, name, rng=rng),
    ),
    BotProfile(
        key="shark",
        label="Shark",
        difficulty=5,
        description="Tight-aggressive: disciplined hand selection backed by aggression, with some bluffing "
        "and size variance for balance. The strongest preset here.",
        factory=lambda pid, name, rng: make_tight_aggressive_bot(pid, name, rng=rng),
    ),
]

_BOT_CATALOG_BY_KEY = {profile.key: profile for profile in BOT_CATALOG}


def list_bot_profiles() -> list[BotProfile]:
    """BOT_CATALOG sorted from most stupid/predictable to strongest."""
    return sorted(BOT_CATALOG, key=lambda p: p.difficulty)


def get_bot_profile(key: str) -> BotProfile:
    try:
        return _BOT_CATALOG_BY_KEY[key]
    except KeyError:
        available = ", ".join(p.key for p in list_bot_profiles())
        raise KeyError(f"unknown bot key {key!r}; available: {available}") from None
