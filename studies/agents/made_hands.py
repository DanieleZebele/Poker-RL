"""Does the model recognise a straight or a flush (the wheel, A-2-3-4-5, included)?

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/agent_study.py --made-hands --model '#1'
    ... --per-class 400 --runouts 100 --jobs 16 --equity-model equity_evaluator/<file>.pt
    ... --folds 3                          # hands of the straights/flushes it folds, with their history

Not a self-play study: a set of counterfactual spots, like `hud_study.py`. Heads-up, 100 bb
deep, the small blind raises to 2.5 bb and the big blind calls; the streets before the one
studied go check-check. For each class of hand (nothing, pair, two pair, trips, straight,
wheel, flush, and on the river the straight or flush the board itself makes) the model is
asked twice, with the same cards:

- **leading**: the big blind speaks first on the street; what is read is how often it bets;
- **facing**: the big blind bets 75% of the pot into it; what is read is how often it folds,
  calls and raises.

Three things are compared. The hand's true equity against a random hand (a Monte Carlo over
`--runouts` opponents and boards, settled by the project's evaluator) against the equity
network's estimate; the model's actions per class; and, for the straights and flushes, a
**twin** of every deal -- the same board with one card of the combination replaced by a card
that breaks it -- so that the difference between a deal and its twin is the effect of the
combination alone, with the board unchanged. A linear probe on the 32 numbers the policy
receives from the equity encoder says whether the combination is readable there at all: if it
is and the model still plays it like its twin, the weakness is in the policy and not in the
cards.

Wherever the model folds a straight, the wheel or a flush facing the bet, the report shows a few
of those hands with their history (`--folds`): the cards, the board street by street and every
action up to the fold, so the spot can be judged by eye.

The true equity is against a *random* hand, and a hand that bets 75% of the pot has a range
better than random, so a low straight folding to it is not necessarily a mistake: read the
actions against the equity as a direction, not as a verdict.

The deals, the classification, the spots and the report are plain Python (the evaluator and
the engine), tested in the ordinary suite; torch is imported only to ask the model.
"""

from __future__ import annotations

import random
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pokerlab.cards.card import Card, Rank, Suit
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import Street
from pokerlab.evaluator.evaluator import HandCategory, evaluate
from pokerlab.gui.spot import Spot, replay
from pokerlab.players.base import Observation

SMALL_BLIND, BIG_BLIND, STACK = 50, 100, 10_000
OPEN_TO, BET_INTO_POT = 250, 375  # the raise to 2.5 bb, then 75% of the 500-chip pot
STREETS = {"flop": 3, "turn": 4, "river": 5}
DECK = [Card(rank, suit) for suit in Suit for rank in Rank]

BASE = ("niente", "coppia", "doppia coppia", "tris")
TARGETS = ("scala", "scala A2345", "colore")
BOARD_PLAYS = ("scala del board", "colore del board")
TWIN = "gemello di "
ROWS = (*BASE, *TARGETS, *(TWIN + target for target in TARGETS), *BOARD_PLAYS)

_NAMES = {
    HandCategory.PAIR: "coppia",
    HandCategory.TWO_PAIR: "doppia coppia",
    HandCategory.THREE_OF_A_KIND: "tris",
    HandCategory.FLUSH: "colore",
}
_BOARD_PLAYS = {HandCategory.STRAIGHT: "scala del board", HandCategory.FLUSH: "colore del board"}
WHEEL = (5,)  # the tiebreakers of the five-high straight

# (street, board, hands): spots picked by hand, shown at the end of the report
EXAMPLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("river", "5c 4d 3h Ks 9s", ("As 2d", "6s 2c", "7d 6c", "As 7c", "Ad Kc", "Jc Tc")),
    ("river", "Ac 4d 3h Ks 9s", ("5s 2d", "5s 6d", "Kd 2c", "Qd Jc")),
    ("flop", "4c 3d 2h", ("As 5d", "6s 5d", "As 6d", "Ad Kc", "4s 4d", "Jc Tc")),
    ("turn", "5c 4d 3h Ks", ("As 2d", "6s 2c", "7d 6c", "As 7c", "Jc Tc")),
    ("river", "Ah 8h 4h Kc 2s", ("Qh 7h", "7h 6h", "Qd 7d", "Kd Qd", "Ks Kd", "Jc Tc")),
    ("river", "9h 8c 7d 2s Kh", ("Jc Td", "Tc 6d", "6c 5d", "Tc 2d", "Ac Kd", "Kc 9d", "Qc 3d")),
    ("flop", "9h 8h 2c", ("Ah Kh", "Jc Td", "Ac Kd")),
    ("flop", "Th 8h 2h", ("Ah Kh", "6h 5h", "Ac Ad", "Kd Qc")),
)


def _board_category(board: Sequence[Card]) -> HandCategory:
    """What the board alone makes: only pairs and trips below the river, anything on it."""
    if len(board) == 5:
        return evaluate(board).category
    counts = sorted(Counter(card.rank for card in board).values(), reverse=True)
    if counts[0] == 3:
        return HandCategory.THREE_OF_A_KIND
    if counts[0] == 2:
        return HandCategory.TWO_PAIR if counts[1:2] == [2] else HandCategory.PAIR
    return HandCategory.HIGH_CARD


def classify(hole: Sequence[Card], board: Sequence[Card]) -> str | None:
    """The class of a hand by what its own cards add to the board, or None for a hand that
    adds nothing to the board's category (the board's pair played by a pair of its own, say)."""
    hand = evaluate([*hole, *board])
    category = hand.category
    if category == HandCategory.HIGH_CARD:
        return "niente"
    if len(board) == 5 and hand == evaluate(board):
        return _BOARD_PLAYS.get(category)
    if category <= _board_category(board):
        return None
    if category == HandCategory.STRAIGHT:
        return "scala A2345" if hand.tiebreakers == WHEEL else "scala"
    return _NAMES.get(category, "full o meglio")


def _fill(rng: random.Random, used: Sequence[Card], count: int) -> list[Card]:
    return rng.sample([card for card in DECK if card not in used], count)


def planted(rng: random.Random, target: str, board_cards: int) -> tuple[list[Card], list[Card]] | None:
    """A deal built to hold `target` (a draw can still miss it: the caller classifies and
    discards), or None when the cards drawn collide."""
    suits = list(Suit)
    if target in ("scala", "scala A2345", "scala del board"):
        if target == "scala A2345":
            ranks = [14, 2, 3, 4, 5]
        else:
            high = rng.randint(6, 14)
            ranks = list(range(high - 4, high + 1))
        if target == "scala del board":
            board = [Card(Rank(rank), rng.choice(suits)) for rank in ranks]
            return _fill(rng, board, 2), board
        mine = 2 if board_cards == 3 else rng.choice([1, 2])
        rng.shuffle(ranks)
        hole = [Card(Rank(rank), rng.choice(suits)) for rank in ranks[:mine]]
        shared = [Card(Rank(rank), rng.choice(suits)) for rank in ranks[mine:]][:board_cards]
        if len({*hole, *shared}) < len(hole) + len(shared):
            return None
    else:
        suit = rng.choice(suits)
        of_suit = [card for card in DECK if card.suit == suit]
        if target == "colore del board":
            board = rng.sample(of_suit, 5)
            return _fill(rng, board, 2), board
        mine = 2 if board_cards == 3 else rng.choice([1, 2])
        chosen = rng.sample(of_suit, 5)
        hole, shared = chosen[:mine], chosen[mine:][:board_cards]
    hole = hole + _fill(rng, hole + shared, 2 - mine)
    board = shared + _fill(rng, hole + shared, board_cards - len(shared))
    rng.shuffle(board)
    return hole, board


def twin_hole(rng: random.Random, hole: Sequence[Card], board: Sequence[Card]) -> list[Card] | None:
    """`hole` with one card of the best five replaced by a card that leaves a hand with nothing."""
    best = set(evaluate([*hole, *board]).best_five)
    mine = [i for i in range(2) if hole[i] in best]
    if not mine:
        return None
    for _ in range(300):
        replacement = rng.choice([card for card in DECK if card not in hole and card not in board])
        new = list(hole)
        new[rng.choice(mine)] = replacement
        if classify(new, board) == "niente":
            return new
    return None


def true_equity(rng: random.Random, hole: Sequence[Card], board: Sequence[Card], runouts: int) -> tuple[float, list[list[Card]]]:
    """The share of the pot against a random hand over `runouts` boards, and those hands."""
    won = 0.0
    opponents: list[list[Card]] = []
    for _ in range(runouts):
        rest = [card for card in DECK if card not in hole and card not in board]
        drawn = rng.sample(rest, 2 + 5 - len(board))
        opponent, full = drawn[:2], [*board, *drawn[2:]]
        mine, theirs = evaluate([*hole, *full]), evaluate([*opponent, *full])
        won += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        opponents.append(opponent)
    return won / runouts, opponents


@dataclass
class Deal:
    street: str
    cls: str
    hole: list[Card]
    board: list[Card]
    equity: float
    opponents: list[list[Card]]
    twin_of: int | None = None  # the index of the deal this one is the twin of


@dataclass(frozen=True)
class DealJob:
    seed: int
    street: str
    per_class: int
    runouts: int


def make_deals(job: DealJob) -> list[Deal]:
    """`per_class` deals of every class on one street; every straight and flush is followed
    by its twin where one exists (`twin_of` indexes into the returned list)."""
    rng = random.Random(job.seed)
    board_cards = STREETS[job.street]
    classes = [*BASE, *TARGETS, *(BOARD_PLAYS if job.street == "river" else ())]
    deals: list[Deal] = []
    for cls in classes:
        found = 0
        while found < job.per_class:
            if cls in BASE:
                drawn = rng.sample(DECK, 2 + board_cards)
                hole, board = drawn[:2], drawn[2:]
            else:
                made = planted(rng, cls, board_cards)
                if made is None:
                    continue
                hole, board = made
            if classify(hole, board) != cls:
                continue
            equity, opponents = true_equity(rng, hole, board, job.runouts)
            deals.append(Deal(job.street, cls, hole, board, equity, opponents))
            found += 1
            if cls in TARGETS and (other := twin_hole(rng, hole, board)) is not None:
                equity, opponents = true_equity(rng, other, board, job.runouts)
                deals.append(Deal(job.street, TWIN + cls, other, board, equity, opponents, twin_of=len(deals) - 1))
    return deals


def build_deals(per_class: int, runouts: int, *, jobs: int, seed: int) -> list[Deal]:
    rng = random.Random(seed)
    tasks = [
        DealJob(rng.randrange(2**31), street, per_class // jobs + (1 if i < per_class % jobs else 0), runouts)
        for street in STREETS
        for i in range(jobs)
    ]
    deals: list[Deal] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for part in executor.map(make_deals, [task for task in tasks if task.per_class]):
            base = len(deals)
            deals += [d if d.twin_of is None else Deal(**{**d.__dict__, "twin_of": d.twin_of + base}) for d in part]
    return deals


def script(street: str, facing: bool) -> list[Action]:
    actions = [Action(ActionType.RAISE, OPEN_TO), Action(ActionType.CALL)]
    for earlier in STREETS:
        if earlier == street:
            break
        actions += [Action(ActionType.CHECK), Action(ActionType.CHECK)]
    return actions + [Action(ActionType.BET, BET_INTO_POT)] if facing else actions


def spot_of(hole: Sequence[Card], board: Sequence[Card], street: str, facing: bool) -> tuple[Observation, list[LegalAction]]:
    """The decision of the hero in the spot, through a real `Table` (as `gui/spot.py` does)."""
    spot = Spot(num_players=2, starting_stack=STACK, small_blind=SMALL_BLIND, big_blind=BIG_BLIND,
                my_seat=0 if facing else 1, hole_cards=(hole[0], hole[1]), board=tuple(board),
                script=script(street, facing), seed=1)
    state = replay(spot)
    if state.observation is None or state.invalid_from is not None:
        raise ValueError(f"the spot does not reach a decision: {hole} {board} {street}")
    return state.observation, state.legal_actions


# ---- the report ---------------------------------------------------------------------------

_SEATS = {0: "SB", 1: "BB"}  # heads-up: seat 0 is the button and the small blind
_BOARD_AT = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


def history_lines(observation: Observation, board: Sequence[Card]) -> list[str]:
    """The actions the hero faces, street by street, as `agent_study.format_hand` writes a hand
    (amounts in big blinds, the hero's seat starred)."""
    by_street: dict[Street, list[str]] = {}
    for record in observation.action_history:
        if record.action_type == ActionType.POST_BLIND:
            continue
        amount = f" {record.amount / BIG_BLIND:g}" if record.action_type in (ActionType.BET, ActionType.RAISE) else ""
        seat = f"{'*' if record.seat == observation.my_seat else ''}{_SEATS[record.seat]}"
        by_street.setdefault(record.street, []).append(f"{seat} {record.action_type.value}{amount}")
    lines = []
    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        if street in by_street:
            shown = " ".join(str(card) for card in board[: _BOARD_AT[street]])
            lines.append(f"      {street.value:<7} {('[' + shown + '] ') if shown else ''}{', '.join(by_street[street])}")
    return lines


@dataclass(frozen=True)
class FoldedHand:
    """A deal in which the model's likeliest answer to the bet is a fold."""

    street: str
    cls: str
    hole: str
    board: str
    equity: float
    facing: tuple[float, float, float]
    history: tuple[str, ...]


def folded_hands(deals: Sequence[Deal], readings: Sequence[Reading],
                 spots: Sequence[tuple[Observation, list[LegalAction]]], *, per_group: int, seed: int,
                 ) -> tuple[list[FoldedHand], dict[tuple[str, str], tuple[int, int]]]:
    """`per_group` random hands, for each street and each of `TARGETS`, that the model folds
    facing the bet (fold is its likeliest action), and how many of the group it folds: (folded, all)."""
    rng = random.Random(seed)
    shown: list[FoldedHand] = []
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for street in STREETS:
        for cls in TARGETS:
            group = [i for i, d in enumerate(deals) if d.street == street and d.cls == cls]
            folds = [i for i in group if max(range(3), key=readings[i].facing.__getitem__) == 0]
            counts[(street, cls)] = (len(folds), len(group))
            for i in rng.sample(folds, min(per_group, len(folds))):
                deal, observation = deals[i], spots[i][0]
                shown.append(FoldedHand(
                    street, cls, " ".join(map(str, deal.hole)), " ".join(map(str, deal.board)), deal.equity,
                    readings[i].facing, tuple(history_lines(observation, deal.board)),
                ))
    return shown, counts



@dataclass(frozen=True)
class Reading:
    """What the model does in the two spots of a deal: (fold, call or check, raise or bet)."""

    net_equity: float
    leading: tuple[float, float, float]
    facing: tuple[float, float, float]


@dataclass(frozen=True)
class ExampleRow:
    street: str
    board: str
    hand: str
    cls: str | None
    equity: float
    leading: tuple[float, float, float]
    facing: tuple[float, float, float]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _gap(pairs: Sequence[tuple[int, int]], value) -> float:
    """The mean of `value(deal) - value(twin)` over the (deal, twin) index pairs."""
    return _mean([value(a) - value(b) for a, b in pairs])


def format_report(deals: Sequence[Deal], readings: Sequence[Reading], probe: dict[tuple[str, str], tuple[float, int]],
                  examples: Sequence[ExampleRow], *, label: str, min_count: int = 20,
                  folds: Sequence[FoldedHand] = (), fold_counts: dict[tuple[str, str], tuple[int, int]] | None = None,
                  ) -> list[str]:
    lines = [f"=== {label}: scale e colori (heads-up, 100 bb, rilancio a 2,5 bb chiamato) ===",
             "la equity vera e' contro una mano a caso; 'punta' = la volta che il BB parla per primo,",
             "'contro' = il BB punta 75% del piatto"]
    for street in STREETS:
        lines += ["", f"--- {street.upper()} ---",
                  f"{'classe':24}{'n':>5}{'eq vera':>9}{'eq rete':>9} |{'punta':>7} |{'fold':>7}{'call':>7}{'rilancia':>9}"]
        for cls in ROWS:
            at = [i for i, d in enumerate(deals) if d.street == street and d.cls == cls]
            if len(at) < min_count:
                continue
            lines.append(
                f"{cls:24}{len(at):5d}{_mean([deals[i].equity for i in at]):9.3f}"
                f"{_mean([readings[i].net_equity for i in at]):9.3f} |"
                f"{_mean([readings[i].leading[2] for i in at]):7.1%} |"
                f"{_mean([readings[i].facing[0] for i in at]):7.1%}{_mean([readings[i].facing[1] for i in at]):7.1%}"
                f"{_mean([readings[i].facing[2] for i in at]):9.1%}"
            )
    lines += ["", "--- GEMELLI: la stessa situazione con la combinazione rotta (combinazione meno gemello) ---",
              f"{'strada':7}{'classe':13}{'coppie':>7}{'d eq vera':>10}{'d eq rete':>10} |{'d punta':>8} |{'d fold':>8}{'d rilancia':>11}"]
    for street in STREETS:
        for cls in TARGETS:
            pairs = [(d.twin_of, i) for i, d in enumerate(deals)
                     if d.street == street and d.cls == TWIN + cls and d.twin_of is not None]
            if len(pairs) < min_count:
                continue

            lines.append(
                f"{street:7}{cls:13}{len(pairs):7d}{_gap(pairs, lambda i: deals[i].equity):+10.3f}"
                f"{_gap(pairs, lambda i: readings[i].net_equity):+10.3f} |"
                f"{_gap(pairs, lambda i: readings[i].leading[2]):+8.1%} |"
                f"{_gap(pairs, lambda i: readings[i].facing[0]):+8.1%}"
                f"{_gap(pairs, lambda i: readings[i].facing[2]):+11.1%}"
            )
    if probe:
        lines += ["", "--- SONDA LINEARE sui 32 numeri dell'encoder: combinazione o gemello? (5 parti) ---"]
        lines += [f"{street:7}{cls:13}{accuracy:7.1%} su {total} mani" for (street, cls), (accuracy, total) in probe.items()]
    if folds:
        lines += ["", "--- MANI FOLDATE: la risposta piu' probabile alla puntata e' il fold ---"]
        for (street, cls), (folded, total) in (fold_counts or {}).items():
            group = [f for f in folds if (f.street, f.cls) == (street, cls)]
            lines.append(f"{street} {cls}: la folda in {folded} casi su {total}")
            for hand in group:
                lines.append(
                    f"    {hand.hole} su [{hand.board}]  eq {hand.equity:.2f}  ->  "
                    f"fold {hand.facing[0]:.0%} call {hand.facing[1]:.0%} rilancia {hand.facing[2]:.0%}"
                )
                lines += hand.history
                lines.append("      *SB fold  <- il modello")
    if examples:
        lines += ["", "--- ESEMPI SCELTI A MANO ---"]
        last = None
        for row in examples:
            if (row.street, row.board) != last:
                lines += ["", f"{row.street} {row.board}"]
                last = (row.street, row.board)
            lines.append(
                f"  {row.hand:6} {row.cls or '-':16} eq {row.equity:4.2f} | punta {row.leading[2]:5.1%} | "
                f"contro: fold {row.facing[0]:5.1%} call {row.facing[1]:5.1%} rilancia {row.facing[2]:5.1%}"
            )
    return lines


# ---- asking the model (torch, imported lazily) ----------------------------------------------


def default_equity_model(project: Path = Path()) -> Path:
    """The equity network `config.toml` names (`equity_model`, in any section), both relative to the project root."""
    config = tomllib.loads((project / "config.toml").read_text())
    for section in (config, *[v for v in config.values() if isinstance(v, dict)]):
        if "equity_model" in section:
            return project / section["equity_model"]
    raise SystemExit("config.toml non ha 'equity_model': passa --equity-model")


def _plane(cards: Sequence[Card]):
    import torch

    from pokerlab.rl.features import card_index

    plane = torch.zeros(52)
    for card in cards:
        plane[card_index(card)] = 1.0
    return plane


def _network_equity(net, deals: Sequence[Deal], runouts: int) -> list[float]:
    """The equity network's share for the hero against the same opponents as the Monte Carlo."""
    import torch

    holes, boards = [], []
    for deal in deals:
        hero, board = _plane(deal.hole), _plane(deal.board)
        for opponent in deal.opponents:
            seats = torch.zeros(9, 52)
            seats[0], seats[1] = hero, _plane(opponent)
            holes.append(seats)
            boards.append(board)
    in_hand = torch.zeros(9, dtype=torch.bool)
    in_hand[:2] = True
    shares = []
    with torch.no_grad():
        for start in range(0, len(holes), 8192):
            chunk = slice(start, start + 8192)
            h, b = torch.stack(holes[chunk]), torch.stack(boards[chunk])
            shares.append(net(h, b, in_hand.expand(len(h), 9))[:, 0])
    return torch.cat(shares).view(len(deals), runouts).mean(1).tolist()


def _ask(model, spots: Sequence[tuple[Observation, list[LegalAction]]]) -> list[tuple[float, float, float]]:
    import torch

    from pokerlab.rl.action_space import CHECK_CALL_BIN, FOLD_BIN, legal_action_mask
    from pokerlab.rl.features import encode_observation

    masks = [legal_action_mask(obs, legal) for obs, legal in spots]
    features = [encode_observation(obs, big_blind=BIG_BLIND, starting_stack=STACK, legal_mask=mask)
                for (obs, _legal), mask in zip(spots, masks, strict=True)]
    with torch.no_grad():
        logits, _value = model(torch.tensor(features, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool))
        p = torch.softmax(logits, dim=-1)
    fold, passive = p[:, FOLD_BIN], p[:, CHECK_CALL_BIN]
    return list(zip(fold.tolist(), passive.tolist(), (1 - fold - passive).tolist(), strict=True))


def _pair_rows(latent, pairs: Sequence[tuple[int, int]], chosen: Sequence[int]):
    """The encoder's output for each chosen (deal, twin) pair, labelled 1 for the deal and 0 for its twin."""
    import torch

    index = [x for k in chosen for x in pairs[k]]
    return latent[index], torch.tensor([1.0, 0.0] * len(chosen))


def _fit_logistic(x, y):
    """A logistic regression with a small ridge, fitted by L-BFGS: (weights, bias)."""
    import torch

    w, b = torch.zeros(x.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + 1e-3 * (w * w).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w, b


def _probe(model, deals: Sequence[Deal]) -> dict[tuple[str, str], tuple[float, int]]:
    """Held-out accuracy of a ridge-regularised logistic regression on the encoder's output,
    telling a deal with the combination from its twin (5 folds over the pairs)."""
    import torch

    with torch.no_grad():
        latent = model.equity_encoder(torch.stack([_plane(d.hole) for d in deals]),
                                      torch.stack([_plane(d.board) for d in deals]))
    result = {}
    for street in STREETS:
        for cls in TARGETS:
            pairs = [(d.twin_of, i) for i, d in enumerate(deals) if d.street == street and d.cls == TWIN + cls]
            if len(pairs) < 20:
                continue
            order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0)).tolist()
            correct = total = 0
            for fold in range(5):
                test = {k for n, k in enumerate(order) if n % 5 == fold}

                chosen = [k for k in range(len(pairs)) if k not in test]
                w, b = _fit_logistic(*_pair_rows(latent, pairs, chosen))
                x_test, y_test = _pair_rows(latent, pairs, sorted(test))
                with torch.no_grad():
                    correct += ((x_test @ w + b > 0).float() == y_test).sum().item()
                    total += len(y_test)
            result[(street, cls)] = (correct / total, total)
    return result


def examples_of(model, runouts: int) -> list[ExampleRow]:
    rows = []
    for street, board_text, hands in EXAMPLES:
        board = [Card.parse(text) for text in board_text.split()]
        for hand_text in hands:
            hole = [Card.parse(text) for text in hand_text.split()]
            equity, _ = true_equity(random.Random(0), hole, board, runouts)
            leading, facing = (_ask(model, [spot_of(hole, board, street, f)])[0] for f in (False, True))
            rows.append(ExampleRow(street, board_text, hand_text, classify(hole, board), equity, leading, facing))
    return rows


def study(path: Path, equity_model: Path, *, label: str, per_class: int, runouts: int, jobs: int, seed: int,
          min_count: int = 20, folds: int = 2) -> list[str]:
    """The whole report for the model checkpoint at `path`."""
    import torch

    from pokerlab.rl.equity_net import equity_net_from_checkpoint, load_equity_checkpoint
    from pokerlab.rl.ppo import build_model_from_checkpoint

    torch.set_num_threads(4)
    model, _checkpoint = build_model_from_checkpoint(path)
    net = equity_net_from_checkpoint(load_equity_checkpoint(equity_model))
    print(f"genero le situazioni ({per_class} per classe e strada, {runouts} avversari ciascuna)...", file=sys.stderr, flush=True)
    deals = build_deals(per_class, runouts, jobs=jobs, seed=seed)
    print(f"{len(deals):,} situazioni; interrogo il modello...", file=sys.stderr, flush=True)
    shares = _network_equity(net, deals, runouts)
    leading = _ask(model, [spot_of(d.hole, d.board, d.street, False) for d in deals])
    facing_spots = [spot_of(d.hole, d.board, d.street, True) for d in deals]
    facing = _ask(model, facing_spots)
    readings = [Reading(s, a, b) for s, a, b in zip(shares, leading, facing, strict=True)]
    shown, counts = folded_hands(deals, readings, facing_spots, per_group=folds, seed=seed)
    return format_report(deals, readings, _probe(model, deals), examples_of(model, runouts), label=label,
                         min_count=min_count, folds=shown, fold_counts=counts)
