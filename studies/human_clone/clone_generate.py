"""The decisions of real players, as our models' inputs: the data of the human clone.

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/human_clone python studies/human_clone/clone_generate.py \\
        --hands-dir dataset_real_hands/unified --every 20 --out checkpoints/studies/human_clone/data --jobs 40

Every hand of the chosen HandHQ files (`hands-handhq-*.jsonl.gz`, the unified data set) is played
again through the engine, seats relabelled so the button is seat 0, every player the script of
what it did, the deck arranged so the board and the cards shown are where they were; a session
keeps one statistics tracker, so every decision carries the HUD of the players in front as in a
rated session. At every voluntary decision: the features of feature version 3 (what a model reads,
normalised as `RLAgentPlayer` does), the legal mask, the bin of the action taken (`search._bin_of`,
a raise to the whole stack is the all-in bin), and the decision's **group**, one player in one hand.
Per group: its two cards when they were shown (else -1), the cards it cannot hold (the final
board and every other card shown) and the player (`<site>|<name>`, as `human_style.py` names it). A hand the engine plays differently (a site's rule, a misread
record) is left out and counted; so is a decision whose action is not in the mask (a free fold).

Written per input file: `clone-NNNN.npz` (`features` float16, `mask`, `action`, `group`, `street`,
`players`; per group `hole`, `excluded`, `session`, `player`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.arranged_deal import ArrangingRandom, deal_positions
from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import _hand_history_from_dict
from pokerlab.engine.stats import StatsTracker
from pokerlab.engine.table import Table
from pokerlab.players.base import Player
from pokerlab.rl.action_space import ALL_IN_BIN, legal_action_mask
from pokerlab.rl.features import card_index, encode_observation
from pokerlab.rl.search import _bin_of

FEATURE_STACK_BB = 100
STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
EXCLUDED = 21  # 5 board cards and 8 other players' two


class Diverged(Exception):
    pass


class Script:
    def __init__(self, steps) -> None:
        self.steps = steps
        self.position = 0
        self.decisions: list[tuple] = []


class Scripted(Player):
    def __init__(self, player_id: str, name: str, script: Script, sim_seat: int) -> None:
        super().__init__(player_id, name)
        self.script = script
        self.sim_seat = sim_seat

    def act(self, observation, legal_actions):
        script = self.script
        if script.position >= len(script.steps):
            raise Diverged("the engine asks for more than the hand had")
        seat, action = script.steps[script.position]
        if seat != observation.my_seat:
            raise Diverged(f"script seat {seat}, engine asks {observation.my_seat}")
        script.position += 1
        mask = legal_action_mask(observation, legal_actions)
        index = bin_of(action, observation, legal_actions, mask)
        bb = observation.big_blind
        features = encode_observation(observation, big_blind=bb, starting_stack=FEATURE_STACK_BB * bb,
                                      legal_mask=mask, version=3)
        script.decisions.append((seat, features, mask, index, STREET_INDEX[observation.street.value]))
        return action


def bin_of(action: Action, observation, legal_actions, mask) -> int | None:
    if action.action_type is ActionType.ALL_IN:
        return ALL_IN_BIN if mask[ALL_IN_BIN] else None
    if action.action_type in (ActionType.BET, ActionType.RAISE):
        top = max((la.max_amount for la in legal_actions if la.action_type is action.action_type), default=None)
        if top is not None and action.amount >= top and mask[ALL_IN_BIN]:
            return ALL_IN_BIN
    return _bin_of(action, observation, legal_actions)


def action_of(record) -> Action:
    if record.action_type in (ActionType.BET, ActionType.RAISE):
        return Action(record.action_type, record.amount)
    return Action(record.action_type)


def replay(hand: dict, tracker: StatsTracker):
    """The decisions of one hand: `(sim seat -> real seat, decisions, holes known)`."""
    history = _hand_history_from_dict(hand)
    seats = sorted(history.starting_stacks)
    button = seats.index(history.button_seat)
    order = seats[button:] + seats[:button]
    sim_of = {real: i for i, real in enumerate(order)}
    steps = [(sim_of[r.seat], action_of(r)) for r in history.actions
             if r.action_type not in (ActionType.POST_BLIND, ActionType.POST_ANTE)]
    script = Script(steps)
    players = [Scripted(history.seat_names[real], history.seat_names[real], script, i) for i, real in enumerate(order)]
    shown = {sim_of[int(seat)]: tuple(Card.parse(c) for c in cards)
             for seat, cards in (hand.get("hole_cards") or {}).items() if cards and len(cards) == 2}
    board = tuple(history.community_cards)
    positions = deal_positions(len(order))
    placements = {}
    for seat, cards in shown.items():
        placements[positions[f"seat{seat}"][0]], placements[positions[f"seat{seat}"][1]] = cards
    for i, slot in enumerate([*positions["flop"], *positions["turn"], *positions["river"]][: len(board)]):
        placements[slot] = board[i]
    config = GameConfig(num_players=len(order), starting_stack=max(history.starting_stacks.values()),
                        small_blind=history.small_blind, big_blind=history.big_blind)
    table = Table(config, players, rng=ArrangingRandom(placements, seed=0), stats_tracker=tracker)
    table.stacks = [history.starting_stacks[real] for real in order]
    played = table.play_hand()
    if script.position != len(steps):
        raise Diverged("the hand ended before the script")
    dealt = tuple(played.hand_history.community_cards)
    if dealt[: len(board)] != board[: len(dealt)]:
        raise Diverged("another board")
    return order, script.decisions, shown, board


def convert_file(path: Path, out: Path) -> dict:
    target = out / f"clone-{path.stem.split('.')[0].split('-')[-1]}-{path.stem.split('-')[1]}.npz"
    if target.exists():
        return {"file": path.name, "skipped": True}
    features, masks, actions, groups, streets, sizes = [], [], [], [], [], []
    holes, excluded, sessions, names = [], [], [], []
    counts = {"hands": 0, "diverged": 0, "not_in_mask": 0, "decisions": 0}
    reasons: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            session = json.loads(line)
            tracker = StatsTracker()
            for hand in session["hands"]:
                counts["hands"] += 1
                try:
                    order, decisions, shown, board = replay(hand, tracker)
                except Exception as error:  # noqa: BLE001 - a hand the engine plays otherwise is left out
                    counts["diverged"] += 1
                    reason = f"{type(error).__name__}: {str(error)[:40]}"
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                group_of: dict[int, int] = {}
                for seat, feats, mask, index, street in decisions:
                    if index is None or not mask[index]:
                        counts["not_in_mask"] += 1
                        continue
                    if seat not in group_of:
                        group_of[seat] = len(holes)
                        known = shown.get(seat)
                        holes.append([card_index(c) for c in known] if known else [-1, -1])
                        others = [card_index(c) for c in board] + [card_index(c) for s, cs in shown.items()
                                                                  if s != seat for c in cs]
                        excluded.append(others + [-1] * (EXCLUDED - len(others)))
                        sessions.append(session["session"])
                        names.append(f"{session['source']}|{hand['seat_names'][str(order[seat])]}")
                    features.append(np.asarray(feats, dtype=np.float16))
                    masks.append(mask)
                    actions.append(index)
                    groups.append(group_of[seat])
                    streets.append(street)
                    sizes.append(len(order))
                    counts["decisions"] += 1
    np.savez(target,
             features=np.stack(features) if features else np.zeros((0, 1), np.float16),
             mask=np.array(masks, dtype=bool), action=np.array(actions, dtype=np.int8),
             group=np.array(groups, dtype=np.int32), street=np.array(streets, dtype=np.int8),
             players=np.array(sizes, dtype=np.int8), hole=np.array(holes, dtype=np.int8).reshape(-1, 2),
             excluded=np.array(excluded, dtype=np.int8).reshape(-1, EXCLUDED),
             session=np.array(sessions, dtype=np.int64), player=np.array(names, dtype=str))
    return {"file": path.name, **counts, "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:5])}


def main(argv: list[str] | None = None) -> int:
    from pokerlab.rl.forking import run_in_copies

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hands-dir", type=Path, default=Path("dataset_real_hands/unified"))
    parser.add_argument("--source", default="handhq", help="whose hands: handhq, or pluribus (every card known)")
    parser.add_argument("--every", type=int, default=20, help="one file in this many")
    parser.add_argument("--limit", type=int, default=0, help="at most this many files (0: all chosen)")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/studies/human_clone/data"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.hands_dir.glob(f"hands-{args.source}-*.jsonl.gz"))[:: args.every]
    if args.limit:
        files = files[: args.limit]
    total: dict[str, int] = {}
    for result in run_in_copies(lambda job: convert_file(files[job], args.out), len(files), args.jobs):
        print(result, flush=True)
        for key, value in result.items():
            if isinstance(value, int) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
    print("totale", total, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
