"""Read the hand and the board off the poker client, for the spot screen.

`ScreenReader.read()` captures the two zones saved in `vision_data/regions.json`
and reads them with `vision.recognize`, built from the labelled crops in
`vision_data/crops/`. The spot screen calls it every half second; nothing
here touches Tk, so the whole decision of *what* gets applied is testable.

A reading is deliberately conservative: a zone whose suit could not be read
(`?`), or a card read in both the hand and the board, is reported as unreadable
rather than half-applied -- the screen then keeps what it had.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pokerlab.vision.labels import VALID_COUNTS
from pokerlab.vision.regions import (
    BOARD,
    DEALER_ZONE_MAX_SIDE,
    HOLE_CARDS,
    POT,
    TURN_TIMER,
    bet_region_name,
    dealer_region_name,
    load_regions,
    player_region_name,
    stack_region_name,
)

# The table size whose per-seat zones are read unless told otherwise; the spot
# screen chooses 6 or 8 (`regions.DEALER_TABLE_SIZES`).
DEALER_PLAYERS = 6


@dataclass
class ScreenReading:
    """What is on screen now. `None` for a zone means "not known" (not set,
    capture failed, or unreadable) -- not the same as `[]`, "no card there"."""

    hole: list[str] | None = None
    board: list[str] | None = None
    # The client seat holding the dealer button (0 = you, clockwise on screen),
    # or None when no zone shows it or the dealer zones are not set.
    dealer: int | None = None
    # The state of each client seat whose player zone is set (`labels.SEAT_STATES`),
    # or None when no player zone is set.
    seats: dict[int, str] | None = None
    # Chips in front of each client seat, in big blinds (0.0 when the zone shows
    # none); a seat whose amount could not be read is missing. None when no bet
    # zone is set. `pot` is what the client shows as the pot: the earlier
    # streets only, the current bets are not in it.
    bets: dict[int, float] | None = None
    pot: float | None = None
    # Your countdown bar is showing: it is your turn. None when the zone is not
    # set or there are not yet examples of both kinds to tell them apart.
    my_turn: bool | None = None
    # Each client seat's stack (chips behind, not counting what is in front of
    # it), in big blinds; a seat whose stack could not be read is missing.
    # None when no stack zone is set.
    stacks: dict[int, float] | None = None
    problems: list[str] = field(default_factory=list)


class VisionNotAvailable(Exception):
    """The vision extra is missing: reading the screen cannot work at all."""


class ScreenReader:
    def __init__(self, crops_dir: Path | None = None, players: int = DEALER_PLAYERS) -> None:
        """`players`: the table size whose seat zones are read (`dealer_<players>_*`,
        `player_<players>_*`, ...); it can be changed between two readings."""
        self.players = players
        try:
            from pokerlab.vision import recognize
        except ImportError as exc:
            raise VisionNotAvailable(f"serve l'extra vision: {exc}") from exc
        self._recognize = recognize
        self.crops_dir = Path(crops_dir) if crops_dir is not None else recognize.CROPS_DIR
        self._recognizer = None
        self._crop_count = -1
        self._sit_out_templates: list = []
        self._empty_backgrounds: dict = {}  # zone -> that zone's empty-seat thumbnails
        self._seat_crop_count = -1
        self._amount_reader = None
        self._amount_crop_count = -1
        self._turn_reader = None
        self._turn_crop_count = -1

    def _get_recognizer(self):
        """The recogniser, rebuilt whenever the number of saved crops changes,
        so an example collected meanwhile counts on the next reading."""
        count = len(list(self.crops_dir.glob("*.png")))
        if self._recognizer is None or count != self._crop_count:
            self._crop_count = count
            try:
                self._recognizer = self._recognize.build_recognizer(self._recognize.load_dataset(self.crops_dir))
            except ValueError:
                self._recognizer = None
        return self._recognizer

    def read(self) -> ScreenReading:
        from pokerlab.vision.capture import VisionUnavailable, grab_regions

        reading = ScreenReading()
        regions = load_regions()  # re-read each time: zones set meanwhile count at once
        if not regions.regions:
            reading.problems.append("zone non impostate (Collect vision data)")
            return reading
        for seat in range(self.players):
            region = regions.get(dealer_region_name(self.players, seat))
            if region is not None and max(region.width, region.height) > DEALER_ZONE_MAX_SIDE:
                # A player box drawn as a dealer zone: its stack chip is gold too.
                regions.clear(dealer_region_name(self.players, seat))
                reading.problems.append(f"zona gettone del posto {seat} troppo grande, ignorata: rifalla")
        # Every zone in one grab (`grab_regions`): one capture per zone cost 233 ms.
        wanted = {name: region for name, region in regions.regions.items() if name in self._zones_read()}
        try:
            frames = dict(zip(wanted, grab_regions(list(wanted.values())), strict=True))
        except VisionUnavailable as exc:
            raise VisionNotAvailable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - a screen grab can fail in many ways
            reading.problems.append(f"cattura non riuscita: {exc}")
            return reading
        self._read_cards(frames, reading)
        self._read_dealer(frames, reading)
        self._read_seats(frames, reading)
        self._read_amounts(frames, reading)
        self._read_turn(frames, reading)
        return reading

    def _zones_read(self) -> set[str]:
        names = {HOLE_CARDS, BOARD}
        for seat in range(self.players):
            names.add(dealer_region_name(self.players, seat))
            names.add(player_region_name(self.players, seat))
            names.add(bet_region_name(self.players, seat))
            names.add(stack_region_name(self.players, seat))
        names.add(POT)
        names.add(TURN_TIMER)
        return names

    def _read_cards(self, frames: dict, reading: ScreenReading) -> None:
        if HOLE_CARDS not in frames and BOARD not in frames:
            return
        recognizer = self._get_recognizer()
        if recognizer is None:
            # The cards need examples; the dealer and seat rules do not.
            reading.problems.append("nessun ritaglio di carte etichettato (Collect vision data)")
            return
        for zone in (HOLE_CARDS, BOARD):
            frame = frames.get(zone)
            if frame is None:
                continue
            cards = [m.card for m in recognizer.recognize(frame, zone)]
            where = "mano" if zone == HOLE_CARDS else "board"
            if any(card.endswith("?") for card in cards):
                reading.problems.append(f"seme non riconosciuto in {where}")
                continue
            if len(cards) not in VALID_COUNTS[zone]:
                # A board of 1-2 cards does not exist: it is the flop half dealt.
                reading.problems.append(f"{where} a metà ({len(cards)} carte), attendo")
                continue
            if zone == HOLE_CARDS:
                reading.hole = cards
            else:
                reading.board = cards
        if reading.hole and reading.board and set(reading.hole) & set(reading.board):
            reading.problems.append("la stessa carta letta in mano e sul board")
            reading.hole = reading.board = None

    def _read_dealer(self, frames: dict, reading: ScreenReading) -> None:
        """Which seat shows the button; nothing if no dealer zone is set. Two
        zones showing one is a reading not to trust, so it is not applied."""
        from pokerlab.vision.dealer import find_dealer

        seats = {
            seat: frames[name] for seat in range(self.players)
            if (name := dealer_region_name(self.players, seat)) in frames
        }
        if not seats:
            return
        found = find_dealer(seats)
        if found.ambiguous:
            reading.problems.append("dealer in più di un posto, attendo")
            return
        reading.dealer = found.seat

    def _get_sit_out_templates(self) -> list:
        """The "SIT OUT" lettering of the labelled seat crops, reloaded when
        their number changes; the empty-seat backgrounds are reloaded with it."""
        from pokerlab.vision.labels import PLAYERS_DIR
        from pokerlab.vision.seats import empty_backgrounds, load_seats, sit_out_templates

        count = len(list(PLAYERS_DIR.glob("*.png")))
        if count != self._seat_crop_count:
            self._seat_crop_count = count
            labelled = load_seats(PLAYERS_DIR)
            self._sit_out_templates = sit_out_templates(labelled)
            self._empty_backgrounds = empty_backgrounds(labelled)
        return self._sit_out_templates

    def _read_seats(self, frames: dict, reading: ScreenReading) -> None:
        """The state of every seat whose player zone is set."""
        from pokerlab.vision.seats import read_seat

        seats = {
            seat: (name, frames[name]) for seat in range(self.players)
            if (name := player_region_name(self.players, seat)) in frames
        }
        if not seats:
            return
        templates = self._get_sit_out_templates()
        reading.seats = {
            seat: read_seat(frame, seat, templates, self._empty_backgrounds.get(name)).state
            for seat, (name, frame) in seats.items()
        }


    def _get_amount_reader(self):
        """The bet/pot reader, rebuilt when the number of labelled amount crops
        changes (the card ranks are its fallback for unseen digits)."""
        from pokerlab.vision.amounts import build_reader, load_amounts
        from pokerlab.vision.labels import AMOUNTS_DIR, STACKS_DIR

        # Bets and stacks are written in the same digits, so both lend examples.
        count = len(list(AMOUNTS_DIR.glob("*.png"))) + len(list(STACKS_DIR.glob("*.png")))
        if self._amount_reader is None or count != self._amount_crop_count:
            self._amount_crop_count = count
            self._amount_reader = build_reader(load_amounts(AMOUNTS_DIR) + load_amounts(STACKS_DIR))
        return self._amount_reader

    def _read_amounts(self, frames: dict, reading: ScreenReading) -> None:
        """Every bet zone and the pot zone set, in big blinds."""
        from pokerlab.vision.amounts import amount_value

        zones = {seat: bet_region_name(self.players, seat) for seat in range(self.players)}
        stack_zones = {seat: stack_region_name(self.players, seat) for seat in range(self.players)}
        if not any(name in frames for name in [*zones.values(), *stack_zones.values(), POT]):
            return
        reader = self._get_amount_reader()

        def value(frame, zone: str = "") -> float | None:
            got = reader.read(frame, zone)
            if got is None:
                return None
            return 0.0 if not got.text else amount_value(got.text)

        bets = {}
        for seat, name in zones.items():
            if name in frames:
                amount = value(frames[name])
                if amount is None:
                    reading.problems.append(f"puntata del posto {seat} non leggibile")
                else:
                    bets[seat] = amount
        if any(name in frames for name in zones.values()):
            reading.bets = bets
        if POT in frames:
            reading.pot = value(frames[POT])
        stacks = {}
        for seat, name in stack_zones.items():
            if name in frames:
                amount = value(frames[name], name)
                if amount is None:
                    reading.problems.append(f"stack del posto {seat} non leggibile")
                else:
                    stacks[seat] = amount
        if any(name in frames for name in stack_zones.values()):
            reading.stacks = stacks


    def _read_turn(self, frames: dict, reading: ScreenReading) -> None:
        """Whether your countdown bar is showing; the reader is rebuilt when the
        number of labelled timer crops changes."""
        if TURN_TIMER not in frames:
            return
        from pokerlab.vision.labels import TURN_DIR
        from pokerlab.vision.turn import TurnReader, load_turns

        count = len(list(TURN_DIR.glob("*.png")))
        if self._turn_reader is None or count != self._turn_crop_count:
            self._turn_crop_count = count
            self._turn_reader = TurnReader(load_turns(TURN_DIR))
        reading.my_turn = self._turn_reader.read(frames[TURN_TIMER])


@dataclass
class ReadingChange:
    """What a new reading changes in the spot, compared with the last one."""

    hole: list[str] | None = None  # set when the hand changed
    board: list[str] | None = None  # set when the board changed
    dealer: int | None = None  # set when the button moved (client seat)
    seats: dict[int, str] | None = None  # set when any seat's state changed
    bets: dict[int, float] | None = None  # set when any amount changed
    new_hand: bool = False  # different hole cards: the old actions are stale

    @property
    def any(self) -> bool:
        return (self.hole is not None or self.board is not None or self.dealer is not None
                or self.seats is not None or self.bets is not None)


def diff_reading(previous: ScreenReading | None, current: ScreenReading,
                 last_hand: list[str] | None, last_dealer: int | None = None) -> ReadingChange:
    """Only what *the screen* changed since the previous reading.

    Comparing with the last reading rather than with the spot is what lets a
    manual correction stick: an unchanged screen changes nothing. A zone not
    known now (`None`) changes nothing either. `last_hand` is the last non-empty
    hand read, so a new hand is recognised even across a fold (cards, none,
    other cards) -- and the same cards coming back after a flicker are not one."""
    change = ReadingChange()
    before_hole = previous.hole if previous else None
    before_board = previous.board if previous else None
    if current.hole is not None and current.hole != before_hole:
        change.hole = current.hole
        change.new_hand = bool(current.hole) and current.hole != last_hand
    if current.board is not None and current.board != before_board:
        change.board = current.board
    # No button seen (between hands, mid-animation) moves nothing, and the
    # button reappearing where it was is not a move: compared with the last
    # seat it was seen on, not with the previous reading.
    if current.dealer is not None and current.dealer != last_dealer:
        change.dealer = current.dealer
    if current.seats is not None and current.seats != (previous.seats if previous else None):
        change.seats = current.seats
    if current.bets is not None and current.bets != (previous.bets if previous else None):
        change.bets = current.bets
    return change


def sit_out_blinds(seats: dict[int, str], bets: dict[int, float] | None) -> set[int]:
    """The players sitting out who have chips in front of them: a player in
    sit-out can still be the small or the big blind, posted for them, and the
    table only plays right with them in it -- otherwise the engine would hand
    their blind to the next player round."""
    from pokerlab.vision.labels import SEAT_SIT_OUT

    return {
        seat for seat, state in seats.items()
        if state == SEAT_SIT_OUT and (bets or {}).get(seat, 0.0) > 0
    }


def seated_seats(seats: dict[int, str], bets: dict[int, float] | None = None, *, antes: bool = False) -> set[int]:
    """The client seats to put at the table: everyone but empty seats and
    players sitting out. A folded player is still at the table (only out of
    this hand), and from one picture "out, folded" and "out, waiting to join"
    look the same -- so a newcomer is seated until the next hand shows better.
    A player sitting out *is* seated when a blind is in front of them
    (`sit_out_blinds`), and with `antes` always: at a tournament table they are
    dealt in and pay the ante like everyone else. Being out of the hand, they
    fold when their turn comes."""
    from pokerlab.vision.labels import SEAT_EMPTY, SEAT_SIT_OUT

    seated = {seat for seat, state in seats.items() if state not in (SEAT_EMPTY, SEAT_SIT_OUT)}
    if antes:
        seated |= {seat for seat, state in seats.items() if state == SEAT_SIT_OUT}
    return seated | sit_out_blinds(seats, bets)


def ante_from_pot(seats: dict[int, str], pot: float | None, board: list | None) -> float | None:
    """The ante each player paid, in big blinds, from a preflop reading: the client's
    pot leaves out the bets still in front of the players, so before the flop it holds
    the antes alone, paid by every seat that is not empty. None when the reading cannot
    tell (not preflop, no pot read, fewer than two players)."""
    from pokerlab.vision.labels import SEAT_EMPTY

    if pot is None or board is None or board:
        return None
    dealt = sum(1 for state in seats.values() if state != SEAT_EMPTY)
    if dealt < 2:
        return None
    return pot / dealt
