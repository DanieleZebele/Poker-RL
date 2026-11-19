"""Does the model see a dangerous board? Top pair on a dry board against top pair on a board
with four to a straight or four to a flush.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/board_danger.py --model '#1'
    ... --per-cell 400 --jobs 30 --equity-model equity_evaluator/<file>.pt

Counterfactual spots, like `made_hands.py` (and through its spot builder): heads-up, 100 bb,
the small blind raises to 2.5 bb, the big blind calls, the streets before go check-check. For
every hand class (top pair, overpair, two pair, set, and the straight or flush the board offers)
on every kind of board (dry, three or four to a straight, three or four to a flush), the model is
asked twice: leading (the big blind speaks first: how often it bets, and how often big, 75% of
the pot or more) and facing a 75% pot bet (fold, call, raise). Next to it, the hand's true equity
against a random hand (exact on the river, every one of the 990 hands; Monte Carlo on the turn).

Two readings of the encoder, which is all the policy sees of the cards: a linear probe on its 32
numbers telling a dangerous board from a dry one with the same hand class, and a ridge regression
of the true equity on them (R^2 held out). If the probe reads the danger and the model still plays
top pair the same on both boards, the weakness is in the policy; if the probe cannot, it is in the
encoder, and no amount of training can fix it.

The deals, the board classes and the equity are plain Python (the evaluator), tested in the
ordinary suite; torch is imported only to ask the model.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from made_hands import DECK, spot_of

from pokerlab.cards.card import Card
from pokerlab.evaluator.evaluator import HandCategory, evaluate

STREETS = {"turn": 4, "river": 5}
BOARDS = ("secco", "3 alla scala", "4 alla scala", "3 a colore", "4 a colore")
HANDS = ("top pair", "overpair", "doppia coppia", "tris", "scala", "colore")
BIG_BINS = range(6, 11)  # 75%, 100%, 150%, 200% of the pot and all-in


def straight_need(board: Sequence[Card]) -> int:
    """How many cards a hand must add to the board to make a straight (0: the board is one)."""
    ranks = {int(card.rank) for card in board}
    if 14 in ranks:
        ranks.add(1)
    return 5 - max(len(ranks & set(range(low, low + 5))) for low in range(1, 11))


def flush_need(board: Sequence[Card]) -> int:
    return 5 - max(Counter(card.suit for card in board).values())


def board_class(board: Sequence[Card]) -> str | None:
    """One danger at a time on an unpaired board, or None (a paired board, or two dangers).

    Three to a straight is the ordinary river (almost every five distinct ranks hold three
    within five), so a flush board may hold it: on the river "3 alla scala" is the reference."""
    if len({card.rank for card in board}) < len(board):
        return None
    straight, suited = 5 - straight_need(board), 5 - flush_need(board)
    if suited <= 2:
        kind = "secco" if straight <= 2 else f"{straight} alla scala"
    elif straight <= 3:
        kind = f"{suited} a colore"
    else:
        return None
    return kind if kind in BOARDS else None  # a board that is itself a straight or a flush is not studied


REFERENCE = {"turn": "secco", "river": "3 alla scala"}


def hand_class(hole: Sequence[Card], board: Sequence[Card]) -> str | None:
    """What the hand makes with its own cards, or None for anything else (a weaker pair, say)."""
    hand = evaluate([*hole, *board])
    top = max(int(card.rank) for card in board)
    held = [int(card.rank) for card in hole]
    on_board = {int(card.rank) for card in board}
    if hand.category == HandCategory.PAIR:
        if held[0] == held[1] and held[0] > top:
            return "overpair"
        if top in held and held.count(top) == 1:
            return "top pair"
        return None
    if hand.category == HandCategory.TWO_PAIR and all(rank in on_board for rank in held) and held[0] != held[1]:
        return "doppia coppia"
    if hand.category == HandCategory.THREE_OF_A_KIND and held[0] == held[1]:
        return "tris"
    if hand.category in (HandCategory.STRAIGHT, HandCategory.FLUSH):
        if len(board) == 5 and hand == evaluate(board):
            return None  # the board's own straight or flush, played by everyone
        return "scala" if hand.category == HandCategory.STRAIGHT else "colore"
    return None


def equity(rng: random.Random, hole: Sequence[Card], board: Sequence[Card], runouts: int) -> float:
    """The share of the pot against a random hand: exact on the river, `runouts` boards before."""
    rest = [card for card in DECK if card not in hole and card not in board]
    if len(board) == 5:
        mine = evaluate([*hole, *board])
        won = total = 0.0
        for i, first in enumerate(rest):
            for second in rest[i + 1:]:
                theirs = evaluate([first, second, *board])
                won += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
                total += 1
        return won / total
    won = 0.0
    for _ in range(runouts):
        drawn = rng.sample(rest, 2 + 5 - len(board))
        full = [*board, *drawn[2:]]
        mine, theirs = evaluate([*hole, *full]), evaluate([*drawn[:2], *full])
        won += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return won / runouts


def showdowns(hole: Sequence[Card], board: Sequence[Card]) -> tuple[list[tuple[Card, Card]], list[float]]:
    """Every hand an opponent can hold on a complete board, and what the hero wins against it (1, 0.5, 0)."""
    rest = [card for card in DECK if card not in hole and card not in board]
    mine = evaluate([*hole, *board])
    hands, results = [], []
    for i, first in enumerate(rest):
        for second in rest[i + 1:]:
            theirs = evaluate([first, second, *board])
            hands.append((first, second))
            results.append(1.0 if mine > theirs else 0.5 if mine == theirs else 0.0)
    return hands, results


BET_FRACTION = 0.75  # the river bet studied, as a fraction of the pot


@dataclass(frozen=True)
class RangeValue:
    """The hero's river spot against the opponent's range (the model's own play weighs the range)."""

    equity: float  # against the range the line leaves the opponent
    fold: float  # how often that range folds to the bet
    raise_: float  # how often it raises (counted as a call below)
    called: float  # the hero's share of the pot when the bet is not folded to
    ev_gain: float  # EV(bet) - EV(check), in pots: the bet is worth making when it is positive


def range_value(results: Sequence[float], weights: Sequence[float], folds: Sequence[float],
                raises: Sequence[float], bet: float = BET_FRACTION) -> RangeValue:
    """Checking wins the equity against the range (the hand checked down); betting `bet` pots wins
    the pot when the opponent folds and `called * (1 + 2 bet) - bet` when it does not (a raise is
    taken as a call: the hero's answer to it is not modelled)."""
    total = sum(weights)
    equity_ = sum(w * r for w, r in zip(weights, results, strict=True)) / total
    fold = sum(w * f for w, f in zip(weights, folds, strict=True)) / total
    raise_ = sum(w * r for w, r in zip(weights, raises, strict=True)) / total
    staying = [w * (1 - f) for w, f in zip(weights, folds, strict=True)]
    called = sum(s * r for s, r in zip(staying, results, strict=True)) / max(sum(staying), 1e-12)
    ev_bet = fold + (1 - fold) * (called * (1 + 2 * bet) - bet)
    return RangeValue(equity_, fold, raise_, called, ev_bet - equity_)


@dataclass
class Deal:
    street: str
    board_kind: str
    hand: str
    hole: list[Card]
    board: list[Card]
    equity: float


@dataclass(frozen=True)
class DealJob:
    seed: int
    street: str
    per_cell: int
    runouts: int
    max_draws: int = 1_500_000


def make_deals(job: DealJob) -> list[Deal]:
    """Up to `per_cell` random deals of every (board, hand) cell on one street, by rejection."""
    rng = random.Random(job.seed)
    cards = STREETS[job.street]
    found: Counter = Counter()
    wanted = len(BOARDS) * len(HANDS)
    deals: list[Deal] = []
    for _ in range(job.max_draws):
        drawn = rng.sample(DECK, 2 + cards)
        hole, board = drawn[:2], drawn[2:]
        kind = board_class(board)
        if kind is None:
            continue
        cls = hand_class(hole, board)
        if cls is None or found[(kind, cls)] >= job.per_cell:
            continue
        found[(kind, cls)] += 1
        deals.append(Deal(job.street, kind, cls, hole, board, equity(rng, hole, board, job.runouts)))
        if sum(1 for n in found.values() if n >= job.per_cell) >= wanted:
            break
    return deals


def build_deals(per_cell: int, runouts: int, *, jobs: int, seed: int) -> list[Deal]:
    rng = random.Random(seed)
    tasks = [
        DealJob(rng.randrange(2**31), street, per_cell // jobs + (1 if i < per_cell % jobs else 0), runouts)
        for street in STREETS
        for i in range(jobs)
    ]
    deals: list[Deal] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for part in executor.map(make_deals, [task for task in tasks if task.per_cell]):
            deals += part
    return deals


# ---- the report ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    leading: tuple[float, float]  # bet at all, bet 75% of the pot or more
    facing: tuple[float, float, float]  # fold, call, raise


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def format_report(deals: Sequence[Deal], readings: Sequence[Reading], probes: dict, *, label: str,
                  min_count: int = 20, ranges: dict[int, RangeValue] | None = None) -> list[str]:
    lines = [f"=== {label}: board pericolosi (heads-up, 100 bb, rilancio a 2,5 bb chiamato, check fino alla strada) ===",
             "eq = equity vera contro una mano a caso; 'punta' = il BB parla per primo (in totale / 75% del piatto o piu');",
             "'contro' = il BB riceve una puntata del 75% del piatto"]
    for street in STREETS:
        lines += ["", f"--- {street.upper()} ---",
                  f"{'mano':15}{'board':14}{'n':>5}{'eq':>7} |{'punta':>7}{'forte':>7} |{'fold':>7}{'call':>7}{'rilancia':>9}"]
        for hand in HANDS:
            for kind in BOARDS:
                at = [i for i, d in enumerate(deals) if d.street == street and d.hand == hand and d.board_kind == kind]
                if len(at) < min_count:
                    continue
                lines.append(
                    f"{hand:15}{kind:14}{len(at):5d}{_mean([deals[i].equity for i in at]):7.3f} |"
                    f"{_mean([readings[i].leading[0] for i in at]):7.1%}{_mean([readings[i].leading[1] for i in at]):7.1%} |"
                    f"{_mean([readings[i].facing[0] for i in at]):7.1%}{_mean([readings[i].facing[1] for i in at]):7.1%}"
                    f"{_mean([readings[i].facing[2] for i in at]):9.1%}"
                )
            lines.append("")
    if ranges:
        lines += ["--- RIVER CONTRO IL RANGE DELL'AVVERSARIO (l'avversario e' il modello stesso) ---",
                  "il range: ogni mano pesata per la probabilita' che il modello faccia la linea dell'avversario",
                  "(rilancio preflop, check al flop e al turn); poi la risposta del modello a una puntata del 75%.",
                  "eq rng = equity contro quel range; vinc. se chiam. = quota del piatto quando non folda;",
                  "dEV = EV(punta 75%) - EV(check), in piatti (un rilancio dell'avversario contato come call)",
                  (f"{'mano':15}{'board':14}{'n':>4}{'eq rnd':>8}{'eq rng':>8}{'fold':>7}{'rilan':>7}{'vinc.chiam':>11}"
                   f"{'dEV':>8} |{'punta':>7}{'forte':>7}{'  dEV>0':>8}")]
        for hand in HANDS:
            for kind in BOARDS:
                at = [i for i in ranges if deals[i].hand == hand and deals[i].board_kind == kind]
                if len(at) < min_count:
                    continue
                values = [ranges[i] for i in at]
                lines.append(
                    f"{hand:15}{kind:14}{len(at):4d}{_mean([deals[i].equity for i in at]):8.3f}"
                    f"{_mean([v.equity for v in values]):8.3f}{_mean([v.fold for v in values]):7.1%}"
                    f"{_mean([v.raise_ for v in values]):7.1%}{_mean([v.called for v in values]):11.1%}"
                    f"{_mean([v.ev_gain for v in values]):+8.3f} |"
                    f"{_mean([readings[i].leading[0] for i in at]):7.1%}{_mean([readings[i].leading[1] for i in at]):7.1%}"
                    f"{_mean([1.0 if v.ev_gain > 0 else 0.0 for v in values]):8.0%}"
                )
            lines.append("")
    if probes:
        lines += ["--- L'ENCODER (i 32 numeri che la policy vede delle carte) ---",
                  "sonda lineare: board pericoloso o di riferimento, a parita' di mano (accuratezza su dati tenuti fuori, 50% = cieco)"]
        for key, value in probes.items():
            if key[0] == "probe":
                _tag, street, hand, kind = key
                accuracy, total = value
                lines.append(f"  {street:6} {hand:14} {kind:13} vs {REFERENCE[street]:13}: {accuracy:6.1%} su {total} mani")
        lines.append("regressione dell'equity vera sui 32 numeri (R^2 su dati tenuti fuori)")
        for key, value in probes.items():
            if key[0] == "r2":
                _tag, street, hand = key
                r2, total = value
                lines.append(f"  {street:6} {hand:14} R^2 {r2:+.2f} su {total} mani")
    return lines


# ---- asking the model (torch, imported lazily) ----------------------------------------------


def _ask(model, spots) -> list[tuple[float, ...]]:
    """The model's whole distribution over the 11 bins in each spot."""
    import torch
    from made_hands import BIG_BLIND, STACK

    from pokerlab.rl.action_space import legal_action_mask
    from pokerlab.rl.features import encode_observation

    masks = [legal_action_mask(obs, legal) for obs, legal in spots]
    features = [encode_observation(obs, big_blind=BIG_BLIND, starting_stack=STACK, legal_mask=mask,
                                   version=model.feature_version)
                for (obs, _legal), mask in zip(spots, masks, strict=True)]
    out = []
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            chunk = slice(start, start + 4096)
            logits, _value = model(torch.tensor(features[chunk], dtype=torch.float32),
                                   torch.tensor(masks[chunk], dtype=torch.bool))
            out += torch.softmax(logits, dim=-1).tolist()
    return out


def _latent(model, deals: Sequence[Deal]):
    import torch
    from made_hands import _plane

    with torch.no_grad():
        return model.equity_encoder(torch.stack([_plane(d.hole) for d in deals]),
                                    torch.stack([_plane(d.board) for d in deals]))


def _folds(n: int, k: int = 5) -> list[set[int]]:
    import torch

    order = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    return [{j for m, j in enumerate(order) if m % k == fold} for fold in range(k)]


def probes(model, deals: Sequence[Deal]) -> dict:
    import torch
    from made_hands import _fit_logistic

    latent = _latent(model, deals)
    result: dict = {}
    for street in STREETS:
        for hand in ("top pair", "overpair", "doppia coppia", "tris"):
            reference = REFERENCE[street]
            dry = [i for i, d in enumerate(deals) if d.street == street and d.hand == hand and d.board_kind == reference]
            for kind in BOARDS:
                if kind in ("secco", reference):
                    continue
                wet = [i for i, d in enumerate(deals) if d.street == street and d.hand == hand and d.board_kind == kind]
                n = min(len(dry), len(wet))
                if n < 20:
                    continue
                index, y = dry[:n] + wet[:n], torch.tensor([0.0] * n + [1.0] * n)
                x = latent[index]
                correct = 0
                for test in _folds(2 * n):
                    train = [j for j in range(2 * n) if j not in test]
                    w, b = _fit_logistic(x[train], y[train])
                    held = sorted(test)
                    with torch.no_grad():
                        correct += ((x[held] @ w + b > 0).float() == y[held]).sum().item()
                result[("probe", street, hand, kind)] = (correct / (2 * n), 2 * n)
            at = [i for i, d in enumerate(deals) if d.street == street and d.hand == hand]
            if len(at) < 50:
                continue
            x = torch.cat([latent[at], torch.ones(len(at), 1)], dim=1)
            y = torch.tensor([deals[i].equity for i in at])
            residual = total = 0.0
            for test in _folds(len(at)):
                train = [j for j in range(len(at)) if j not in test]
                held = sorted(test)
                a = x[train]
                w = torch.linalg.solve(a.T @ a + 1e-2 * torch.eye(a.shape[1]), a.T @ y[train])
                residual += ((x[held] @ w - y[held]) ** 2).sum().item()
                total += ((y[held] - y[train].mean()) ** 2).sum().item()
            result[("r2", street, hand)] = (1 - residual / total, len(at))
    return result


def _river_spots(hole, board):
    """The opponent's four decisions on the line studied (it is the small blind, seat 0): the open,
    the check back on the flop and on the turn, the answer to the river bet."""
    from made_hands import BIG_BLIND, SMALL_BLIND, STACK, script

    from pokerlab.engine.actions import Action, ActionType
    from pokerlab.gui.spot import Spot, replay

    spots = []
    check = Action(ActionType.CHECK)
    for actions in ([], [*script("flop", False), check], [*script("turn", False), check], script("river", True)):
        spot = Spot(num_players=2, starting_stack=STACK, small_blind=SMALL_BLIND, big_blind=BIG_BLIND,
                    my_seat=0, hole_cards=(hole[0], hole[1]), board=tuple(board), script=actions, seed=1)
        state = replay(spot)
        if state.observation is None or state.invalid_from is not None or state.observation.my_seat != 0:
            raise ValueError(f"the opponent's spot does not reach its decision: {actions}")
        spots.append((state.observation, state.legal_actions))
    return spots


def range_values(model, deals: Sequence[Deal], chosen: Sequence[int], *, jobs: int) -> dict[int, RangeValue]:
    """`range_value` of every chosen river deal, the opponent's range weighed by the model's own play."""
    import torch
    from made_hands import BIG_BLIND, STACK

    from pokerlab.rl.action_space import CHECK_CALL_BIN, FOLD_BIN, legal_action_mask
    from pokerlab.rl.features import CARDS_DIM, card_index, encode_observation

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        enumerated = list(executor.map(showdowns, [deals[i].hole for i in chosen], [deals[i].board for i in chosen],
                                       chunksize=4))
    out: dict[int, RangeValue] = {}
    checked = False
    for i, (hands, results) in zip(chosen, enumerated, strict=True):
        deal = deals[i]
        probabilities = []
        for observation, legal in _river_spots(hands[0], deal.board):
            mask = legal_action_mask(observation, legal)
            base = torch.tensor(encode_observation(observation, big_blind=BIG_BLIND, starting_stack=STACK,
                                                   legal_mask=mask, version=model.feature_version))
            if not checked:  # nothing but the card planes may depend on the hole cards
                other, _ = _river_spots(hands[-1], deal.board)[len(probabilities)]
                again = encode_observation(other, big_blind=BIG_BLIND, starting_stack=STACK, legal_mask=mask,
                                           version=model.feature_version)
                assert torch.equal(base[CARDS_DIM:], torch.tensor(again)[CARDS_DIM:]), "features depend on the hole"
            rows = base.repeat(len(hands), 1)
            rows[:, 0:52] = 0.0
            index = torch.tensor([[card_index(a), card_index(b)] for a, b in hands])
            rows[:, 0:52].scatter_(1, index, 1.0)
            rows[:, 5 * 52:6 * 52] = rows[:, 0:52] + rows[:, 4 * 52:5 * 52]
            with torch.no_grad():
                logits, _value = model(rows, torch.tensor(mask, dtype=torch.bool).expand(len(hands), -1))
                probabilities.append(torch.softmax(logits, dim=-1))
        checked = True
        pre, flop, turn, river = probabilities
        weights = ((1 - pre[:, FOLD_BIN] - pre[:, CHECK_CALL_BIN]) * flop[:, CHECK_CALL_BIN] * turn[:, CHECK_CALL_BIN])
        folds = river[:, FOLD_BIN]
        raises = 1 - river[:, FOLD_BIN] - river[:, CHECK_CALL_BIN]
        out[i] = range_value(results, weights.tolist(), folds.tolist(), raises.tolist())
    return out


def study(path: Path, *, label: str, per_cell: int, runouts: int, jobs: int, seed: int, min_count: int,
          range_per_cell: int = 0) -> list[str]:
    import torch

    from pokerlab.rl.ppo import build_model_from_checkpoint

    torch.set_num_threads(4)
    model, _checkpoint = build_model_from_checkpoint(path)
    print(f"genero le situazioni ({per_cell} per casella e strada)...", file=sys.stderr, flush=True)
    deals = build_deals(per_cell, runouts, jobs=jobs, seed=seed)
    print(f"{len(deals):,} situazioni; interrogo il modello...", file=sys.stderr, flush=True)
    leading = _ask(model, [spot_of(d.hole, d.board, d.street, False) for d in deals])
    facing = _ask(model, [spot_of(d.hole, d.board, d.street, True) for d in deals])
    readings = [
        Reading((1 - p[0] - p[1], sum(p[i] for i in BIG_BINS)), (q[0], q[1], 1 - q[0] - q[1]))
        for p, q in zip(leading, facing, strict=True)
    ]
    ranges = None
    if range_per_cell:
        chosen = []
        for hand in HANDS:
            for kind in BOARDS:
                at = [i for i, d in enumerate(deals) if d.street == "river" and d.hand == hand and d.board_kind == kind]
                chosen += at[:range_per_cell]
        print(f"il range dell'avversario su {len(chosen):,} situazioni del river...", file=sys.stderr, flush=True)
        ranges = range_values(model, deals, chosen, jobs=jobs)
    return format_report(deals, readings, probes(model, deals), label=label, min_count=min_count, ranges=ranges)


def main(argv: Sequence[str] | None = None) -> int:
    from duel_power import resolve_model

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="#1", help="etichetta, percorso del checkpoint o #N (rango globale)")
    parser.add_argument("--per-cell", type=int, default=400, help="situazioni per (mano, board) e strada")
    parser.add_argument("--runouts", type=int, default=200, help="board per la equity al turn (al river e' esatta)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--range-per-cell", type=int, default=60,
                        help="situazioni del river per casella giudicate contro il range dell'avversario (0: no)")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    args = parser.parse_args(argv)
    label, path, rating = resolve_model(args.model, global_dir=args.global_dir, root=args.root)
    if rating is not None:
        label = f"{label} (rating {rating:.0f})"
    for line in study(path, label=label, per_cell=args.per_cell, runouts=args.runouts, jobs=args.jobs,
                      seed=args.seed, min_count=args.min_count, range_per_cell=args.range_per_cell):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
