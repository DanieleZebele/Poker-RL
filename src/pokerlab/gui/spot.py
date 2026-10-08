"""Ask the trained models what they would do in a spot you describe.

The point of this mode is to use a population of trained networks as an advisor:
you build a situation -- how many players, what stacks, your cards, the board,
and the actions taken so far -- and every loaded model tells you what it would
do, with the probability it puts on each option and what it thinks the spot is
worth.

Two decisions shape everything here.

**The models are shown an `Observation` the engine built, never one assembled by
hand.** `rl/features.py` does not read an `Observation` field by field: it
rebuilds each player's per-hand total from the action log (`committed_by_seat`,
which works only because the engine records `current_bet` *after* the action), it
ranks position by how many seats act after you, and it aggregates the history per
street. A hand-written `Observation` that looked plausible would produce features
that are subtly wrong, and the advice would be confidently meaningless. So a spot
is described as a *script*, and the script is replayed through a real `Table`.

**A spot is replayed from scratch on every change, and that removes a whole class
of bug.** `Table.play_hand()` is synchronous, so stepping through a hand
interactively would otherwise need the thread-and-queue dance `players/gui.py`
does. Instead the spot keeps the list of actions chosen so far and re-runs the
hand from a fresh table whenever it needs the next decision point. A scripted
hand is microseconds -- the cost is the models' forward passes, and those happen
once, when you ask for advice -- so there is no state to corrupt, nothing to
unwind, and no partially played hand to reason about. It also makes every script
legal by construction: the engine says who acts and what they may do, and the
caller can only pick from that.

**The cards are placed without touching the engine.** `Table` consumes its
`random.Random` only to shuffle the deck, once per hand, and `play_hand` then
deals two cards to each seated player in seat order followed by the flop, turn
and river with no burn. So an rng whose `shuffle` *arranges* the deck instead of
randomising it puts any card anywhere, and the engine deals normally from it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pokerlab.cards.card import Card, Rank, Suit
from pokerlab.engine.actions import Action, ActionType, IllegalActionError, LegalAction
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import ActionRecord, Street
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of import
    from pokerlab.rl.policy import PokerActorCritic

# How many of the best-rated models the mode loads. Five is what the request
# asked for and it is also about the most that fits on screen with a probability
# breakdown each.
DEFAULT_ADVISORS = 5

_RANK_BY_SYMBOL = {
    "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
    "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "T": Rank.TEN, "J": Rank.JACK,
    "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
}
_SUIT_BY_SYMBOL = {"s": Suit.SPADES, "h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS}


def bin_labels() -> list[str]:
    """A name per action bin, built from `action_space`'s own constants so adding
    a bin there cannot leave this list short."""
    from pokerlab.rl.action_space import (
        ACTION_DIM,
        ALL_IN_BIN,
        CHECK_CALL_BIN,
        FIRST_FRACTION_BIN,
        FOLD_BIN,
        POT_FRACTIONS,
        RAISE_MIN_BIN,
    )

    labels = [f"bin {index}" for index in range(ACTION_DIM)]
    labels[FOLD_BIN] = "fold"
    labels[CHECK_CALL_BIN] = "check/call"
    labels[RAISE_MIN_BIN] = "min-raise"
    labels[ALL_IN_BIN] = "all-in"
    for offset, fraction in enumerate(POT_FRACTIONS):
        labels[FIRST_FRACTION_BIN + offset] = f"{fraction:.0%} pot"
    return labels


def parse_card(text: str) -> Card:
    """`"Ah"` -> the ace of hearts. Rank upper, suit lower, as the GUI prints them."""
    cleaned = text.strip()
    if len(cleaned) != 2:
        raise ValueError(f"una carta si scrive con due caratteri, non {text!r}")
    rank, suit = cleaned[0].upper(), cleaned[1].lower()
    if rank not in _RANK_BY_SYMBOL or suit not in _SUIT_BY_SYMBOL:
        raise ValueError(f"carta non riconosciuta: {text!r}")
    return Card(_RANK_BY_SYMBOL[rank], _SUIT_BY_SYMBOL[suit])


def format_card(card: Card) -> str:
    symbol = {v: k for k, v in _RANK_BY_SYMBOL.items()}[card.rank]
    suit = {v: k for k, v in _SUIT_BY_SYMBOL.items()}[card.suit]
    return f"{symbol}{suit}"


class ArrangingRandom(random.Random):
    """A `random.Random` whose `shuffle` deals the cards you asked for.

    `Deck.shuffle()` calls `rng.shuffle(cards)` on a freshly ordered 52-card
    list, and that is the only thing a hand uses its rng for. Overriding it is
    therefore enough to control the whole deal without any engine change, and it
    keeps the engine dealing normally rather than being handed a rigged deck.

    `placements` maps a position in the deal order to a card. Positions the caller
    left out are filled with whatever is left, shuffled by the underlying
    generator so two unspecified opponents do not always hold the same hand.
    """

    def __init__(self, placements: dict[int, Card], seed: int | None = None) -> None:
        super().__init__(seed)
        self._placements = dict(placements)

    def shuffle(self, x, random=None) -> None:
        wanted = set(self._placements.values())
        if len(wanted) != len(self._placements):
            raise ValueError("la stessa carta e' stata assegnata a due posizioni")
        rest = [card for card in x if card not in wanted]
        super().shuffle(rest)
        spare = iter(rest)
        arranged = [self._placements.get(index) or next(spare) for index in range(len(x))]
        x[:] = arranged


def deal_positions(num_players: int) -> dict[str, list[int]]:
    """Where each thing sits in the deal order `play_hand` consumes.

    Two cards per seated player in seat order, then the flop, the turn and the
    river with no burn card. Pinned by a test that plays a hand and checks the
    cards came out where this says they would -- if the engine's dealing ever
    changes, that is the test that fails.
    """
    hole = {seat: [2 * seat, 2 * seat + 1] for seat in range(num_players)}
    board = 2 * num_players
    return {
        **{f"seat{seat}": slots for seat, slots in hole.items()},
        "flop": [board, board + 1, board + 2],
        "turn": [board + 3],
        "river": [board + 4],
    }


@dataclass
class Spot:
    """A situation to ask about: the table, your cards, and what has happened.

    `my_seat` is which chair is yours. The button is always the lowest seat --
    `Table._advance_button` puts it there on a table's first hand -- so choosing
    your seat *is* choosing your position, and `position_names` turns that into
    the labels a player thinks in. Nothing is lost by fixing it: the features
    index seats relative to you anyway.
    """

    num_players: int = 6
    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2
    my_seat: int = 0
    hole_cards: tuple[Card, Card] | None = None
    board: tuple[Card, ...] = ()
    # The actions chosen so far, in the order the engine asked for them. The
    # engine decides whose turn it is, so a script is a flat list rather than a
    # list of (seat, action) pairs -- and it cannot describe an illegal sequence.
    script: list[Action] = field(default_factory=list)
    # Per-seat stacks, when they are not all `starting_stack`.
    stacks: dict[int, int] = field(default_factory=dict)
    seed: int = 0
    # What the opponents' statistics say (`engine/stats.py::StatsTracker.vector`), by seat,
    # for the seats the tracker has seen. They go into the `Observation`s the models read,
    # as they do at every table the models were trained and rated on; a seat missing here
    # reads as "unknown".
    seat_stats: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def game(self) -> GameConfig:
        return GameConfig(
            num_players=self.num_players,
            starting_stack=self.starting_stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

    def validate(self) -> None:
        """A clear error rather than a `KeyError` from deep inside the deal.

        The seat going out of range is not hypothetical: it is what happens when
        the player count is lowered with a high seat already chosen, so the UI
        clamps it -- and this is the backstop for every other caller.
        """
        if not 2 <= self.num_players <= 9:
            raise ValueError(f"un tavolo ha da 2 a 9 posti, non {self.num_players}")
        if not 0 <= self.my_seat < self.num_players:
            raise ValueError(
                f"il posto {self.my_seat} non esiste a un tavolo da {self.num_players}"
            )
        if len(self.board) > 5:
            raise ValueError(f"il board ha al massimo 5 carte, non {len(self.board)}")
        chosen = list(self.hole_cards or ()) + list(self.board)
        if len(set(chosen)) != len(chosen):
            raise ValueError("la stessa carta e' stata usata due volte")

    def placements(self) -> dict[int, Card]:
        self.validate()
        slots = deal_positions(self.num_players)
        placed: dict[int, Card] = {}
        if self.hole_cards is not None:
            for slot, card in zip(slots[f"seat{self.my_seat}"], self.hole_cards, strict=True):
                placed[slot] = card
        board_slots = slots["flop"] + slots["turn"] + slots["river"]
        for slot, card in zip(board_slots, self.board, strict=False):
            placed[slot] = card
        return placed

    def table_stacks(self) -> list[int]:
        return [self.stacks.get(seat, self.starting_stack) for seat in range(self.num_players)]


def position_names(num_players: int) -> list[str]:
    """A position label per seat, with the button at the lowest seat.

    Postflop action opens on the small blind and closes on the button, which is
    the same ordering `features.py` ranks positional value by -- so these labels
    and what the models see agree by construction.
    """
    if num_players == 2:
        return ["BTN/SB", "BB"]
    remaining = num_players - 3
    # Named from both ends, which is how the positions are actually named: the
    # seat right after the big blind is under the gun, the seat right before the
    # button is the cutoff, and the middle fills in between. So 6-max reads
    # UTG/HJ/CO and 9-max UTG/UTG+1/UTG+2/LJ/HJ/CO.
    back = ["CO", "HJ", "LJ"]
    tail: list[str] = []
    for index in range(remaining):
        from_back = remaining - 1 - index
        if index == 0:
            tail.append("UTG")
        elif from_back < len(back):
            tail.append(back[from_back])
        else:
            tail.append(f"UTG+{index}")
    return ["BTN", "SB", "BB"] + tail


@dataclass
class SpotState:
    """Where a replayed script ended up."""

    # None once the hand is over -- the script described a completed hand.
    to_act: int | None
    observation: Observation | None
    legal_actions: list[LegalAction]
    street: Street
    pot: int
    board: tuple[Card, ...]
    stacks: dict[int, int]
    bets: dict[int, int]
    # Every decision the script made, with who made it, for the log the UI shows.
    taken: list[tuple[int, Action]] = field(default_factory=list)
    finished: bool = False
    # Set when part of the script did not apply, for either of the two reasons a
    # stale script goes wrong after the table is edited under it: the engine
    # *refused* an action that used to be legal, or the hand *ended* before the
    # script ran out (lower the player count and two folds can win the pot).
    # Either way the state returned is the one the applied prefix reaches, and
    # this says where the rest was dropped -- so the caller reports it instead of
    # crashing or silently ignoring actions.
    invalid_from: int | None = None
    # The engine's own records of the hand so far: the blinds, then every scripted action.
    # What a `StatsTracker` is fed when the hand is over (`analyse_hand` reads these).
    records: list[ActionRecord] = field(default_factory=list)


class _FixedStats:
    """Stands in for a `StatsTracker` in a replay: hands the statistics the spot was given to
    the `Table`, which puts them in the `Observation`s, and records nothing. The replayed
    hand is a rebuilt one, run again at every edit, so letting the real tracker watch it
    would count it over and over."""

    def __init__(self, by_seat: dict[int, tuple[float, ...]]) -> None:
        self._by_seat = dict(by_seat)

    def vectors(self, player_ids) -> dict[int, tuple[float, ...]]:
        return {seat: vector for seat, vector in self._by_seat.items() if seat in player_ids}

    def record_hand(self, *args, **kwargs) -> None:
        return None


def _table(spot: Spot, replayer: "_Replayer") -> Table:
    """One shared Replayer in every seat: it is the engine that decides whose turn it is, so
    the script is consumed in the engine's own order and a caller cannot describe a sequence
    out of turn."""
    game = spot.game()
    table = Table(
        game,
        [replayer] * game.num_players,
        rng=ArrangingRandom(spot.placements(), seed=spot.seed),
        stats_tracker=_FixedStats(spot.seat_stats) if spot.seat_stats else None,
    )
    table.stacks = spot.table_stacks()
    return table


def _scripted_records(actions: list[ActionRecord], consumed: int) -> list[ActionRecord]:
    """The blinds and the first `consumed` decisions: what the script made happen, without
    the passive actions the replayer plays after it to let the hand finish."""
    blinds = 0
    for record in actions:
        if record.action_type is not ActionType.POST_BLIND:
            break
        blinds += 1
    return list(actions[: blinds + consumed])


class _Replayer(Player):
    """Replays a script, then stops at the first decision it has no answer for.

    When the script runs out this captures the `Observation` and the legal
    actions -- the spot the caller is asking about -- and from then on returns
    the most passive legal action so the hand can finish without raising. What
    happens after the captured decision is irrelevant: only the captured state is
    read.
    """

    def __init__(self, player_id: str, name: str, script: list[Action]) -> None:
        super().__init__(player_id, name)
        self._script = list(script)
        self._index = 0
        self.captured: tuple[Observation, list[LegalAction]] | None = None
        self.taken: list[tuple[int, Action]] = []
        # The street of the last decision seen at all, scripted ones included, so
        # a hand that ended can still say how far it got.
        self.last_street: Street = Street.PREFLOP

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self.last_street = observation.street
        if self._index < len(self._script):
            action = self._script[self._index]
            self._index += 1
            self.taken.append((observation.my_seat, action))
            return action
        if self.captured is None:
            self.captured = (observation, list(legal_actions))
        types = {legal.action_type for legal in legal_actions}
        if ActionType.CHECK in types:
            return Action(ActionType.CHECK)
        if ActionType.FOLD in types:
            return Action(ActionType.FOLD)
        return Action(legal_actions[0].action_type)

    @property
    def consumed(self) -> int:
        return self._index


def replay(spot: Spot) -> SpotState:
    """Run the script through a real `Table` and report where it stopped.

    Called after every edit in the UI, which is affordable because a scripted
    hand costs microseconds: the expensive part of this mode is the models'
    forward passes, and those happen only when advice is asked for.

    A script that has stopped being legal -- because the table was edited under
    it -- is truncated at the first action the engine refuses, and the result
    carries `invalid_from`. It is not an error: the engine refusing is the only
    reliable way to find out, and the caller wants the spot the legal prefix
    reaches plus a note, not an exception.
    """
    while True:
        try:
            return _replay_once(spot)
        except IllegalActionError:
            trimmed = _legal_prefix(spot)
            if trimmed is None:
                raise
            spot, invalid_from = trimmed
            state = _replay_once(spot)
            state.invalid_from = invalid_from
            return state


def _legal_prefix(spot: Spot) -> tuple[Spot, int] | None:
    """`spot` with the refused action and everything after it dropped."""
    replayer = _Replayer("spot", "spot", spot.script)
    table = _table(spot, replayer)
    try:
        table.play_hand()
    except IllegalActionError:
        pass
    cut = max(0, replayer.consumed - 1)
    if cut >= len(spot.script):
        return None
    return replace(spot, script=list(spot.script[:cut])), cut


def _replay_once(spot: Spot) -> SpotState:
    replayer = _Replayer("spot", "spot", spot.script)
    table = _table(spot, replayer)
    result = table.play_hand()
    records = _scripted_records(result.hand_history.actions, replayer.consumed)

    if replayer.captured is None:
        # The hand ended. Anything the script had left is reported as dropped:
        # silently ignoring it would let the UI show actions that never happened.
        leftover = replayer.consumed if replayer.consumed < len(spot.script) else None
        return SpotState(
            to_act=None,
            observation=None,
            legal_actions=[],
            street=replayer.last_street,
            pot=0,
            board=(),
            stacks=dict(enumerate(table.stacks)),
            bets={},
            taken=replayer.taken,
            finished=True,
            invalid_from=leftover,
            records=records,
        )

    observation, legal_actions = replayer.captured
    return SpotState(
        to_act=observation.my_seat,
        observation=observation,
        legal_actions=legal_actions,
        street=observation.street,
        pot=observation.pot_size,
        board=observation.community_cards,
        stacks={info.seat: info.stack for info in observation.seats},
        bets={info.seat: info.current_bet for info in observation.seats},
        taken=replayer.taken,
        finished=False,
        records=records,
    )


@dataclass(frozen=True)
class BinAdvice:
    """One action bin as a model sees it."""

    index: int
    label: str
    action: Action | None
    probability: float
    legal: bool


@dataclass(frozen=True)
class ModelAdvice:
    """What one model would do here."""

    label: str
    rating: float
    # The bin it puts the most probability on, and that probability.
    best: BinAdvice
    bins: list[BinAdvice]
    # The critic's estimate for the spot, in the units the model was trained in
    # (`reward_scale`, so roughly stacks rather than big blinds -- read it as a
    # sign and a magnitude, not as chips).
    value: float


def bb_number(chips: int, big_blind: int) -> str:
    """`37` chips at a big blind of 2 -> "18,5": big blinds, a comma for the
    decimals (as the poker client writes them), no trailing zeros."""
    value = chips / big_blind
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def format_bb(chips: int, big_blind: int) -> str:
    return f"{bb_number(chips, big_blind)} BB"


def parse_bb(text: str, big_blind: int) -> int:
    """ "18,5" (or "18.5") big blinds -> chips at that big blind, rounded to
    the nearest chip. ValueError if it is not a number."""
    value = float(text.strip().replace(",", "."))
    if value < 0:
        raise ValueError(f"importo negativo: {text!r}")
    return round(value * big_blind)


def describe_action(
    action: Action | None, observation: Observation | None = None, big_blind: int | None = None
) -> str:
    """A short label for an action bin; amounts in big blinds when `big_blind`
    is given, in chips otherwise.

    CALL and ALL_IN carry no meaningful `Action.amount` -- the engine fills that
    in itself -- so the amount is read off the observation instead, exactly as
    `gui/app.py::_describe_action` does for the hand log.
    """
    if action is None:
        return "-"

    def amount(chips: int) -> str:
        return format_bb(chips, big_blind) if big_blind else str(chips)

    kind = action.action_type
    if kind is ActionType.FOLD:
        return "fold"
    if kind is ActionType.CHECK:
        return "check"
    if kind is ActionType.CALL:
        if observation is not None:
            owed = observation.current_bet_to_match - observation.my_current_bet
            return f"call {amount(owed)}"
        return "call"
    if kind is ActionType.ALL_IN:
        if observation is not None:
            return f"all-in {amount(observation.my_stack + observation.my_current_bet)}"
        return "all-in"
    verb = "bet" if kind is ActionType.BET else "raise a"
    return f"{verb} {amount(action.amount)}"


def advise(
    observation: Observation,
    legal_actions: list[LegalAction],
    models: list[tuple[str, float, PokerActorCritic]],
    *,
    big_blind: int,
    starting_stack: int,
    device: str = "cpu",
) -> list[ModelAdvice]:
    """Each model's full opinion on one decision.

    Deliberately not routed through `RLAgentPlayer`: that samples one action and
    throws the distribution away, and the distribution is the interesting part --
    "fold 97%" and "fold 34%" are different advice even when both would fold. The
    critic's value comes back too, since it is free in the same forward pass.

    torch is imported inside the function, like every other torch user the GUI
    has, so opening the app still costs nothing until a model is actually loaded.
    """
    import torch

    from pokerlab.rl.action_space import action_index_to_action, legal_action_mask
    from pokerlab.rl.features import encode_observation

    labels = bin_labels()

    mask = legal_action_mask(observation, legal_actions)
    features = encode_observation(
        observation, big_blind=big_blind, starting_stack=starting_stack, legal_mask=mask
    )
    actions = [
        action_index_to_action(index, observation, legal_actions) if legal else None
        for index, legal in enumerate(mask)
    ]

    advice: list[ModelAdvice] = []
    for label, rating, model in models:
        model.to(device)
        with torch.no_grad():
            logits, value = model(
                torch.tensor([features], dtype=torch.float32, device=device),
                torch.tensor([mask], dtype=torch.bool, device=device),
            )
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()
        bins = [
            BinAdvice(
                index=index,
                label=labels[index],
                action=actions[index],
                probability=probabilities[index],
                legal=mask[index],
            )
            for index in range(len(mask))
        ]
        legal_bins = [b for b in bins if b.legal]
        best = max(legal_bins, key=lambda b: b.probability)
        advice.append(
            ModelAdvice(
                label=label,
                rating=rating,
                best=best,
                bins=sorted(legal_bins, key=lambda b: -b.probability),
                value=float(value.item()),
            )
        )
    return advice


def load_advisors(
    paths: list[tuple[str, float, str]], *, device: str = "cpu"
) -> tuple[list[tuple[str, float, Any]], list[tuple[str, str]]]:
    """Rebuild the given checkpoints, returning the loaded ones and the failures.

    A checkpoint that cannot be read is reported rather than raised: the store
    holds thousands of files written by several machines, and one bad one must
    not take the mode down.
    """
    from pokerlab.rl.ppo import build_model_from_checkpoint

    loaded: list[tuple[str, float, Any]] = []
    failed: list[tuple[str, str]] = []
    for label, rating, path in paths:
        try:
            model, _checkpoint = build_model_from_checkpoint(path, device=device)
        except Exception as exc:  # noqa: BLE001 - a user-owned directory
            failed.append((label, str(exc)))
            continue
        loaded.append((label, rating, model))
    return loaded, failed
