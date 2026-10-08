"""`studies/agents/agent_study.py`: everything that reads a played hand is torch-free and
tested here on hands built by hand or played by test bots; only the last test plays a
(tiny) network."""

from __future__ import annotations

import random

import pytest
from agent_study import (
    analyse,
    decisions_of,
    effective_stack_bb,
    format_report,
    made_class,
    positions,
    river_standing,
)
from support import make_random_legal_bot

from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType
from pokerlab.engine.config import GameConfig
from pokerlab.engine.history import SCHEMA_VERSION, HandHistory
from pokerlab.engine.state import ActionRecord, Street
from pokerlab.engine.table import Table
from pokerlab.rl.hand_tiers import ALL_CLASSES, TIER_OF, TIERS, chen_score, combos, hand_class


def cards(text: str) -> list[Card]:
    return [Card.parse(card) for card in text.split()]


def test_the_169_starting_hands_and_their_combinations():
    assert len(ALL_CLASSES) == 169
    assert sum(combos(cls) for cls in ALL_CLASSES) == 1326
    assert hand_class(cards("Ah Kh")) == "AKs"
    assert hand_class(cards("2c 7d")) == "72o"
    assert hand_class(cards("Td Tc")) == "TT"


def test_chen_points_of_a_few_well_known_hands():
    assert chen_score("AA") == 20
    assert chen_score("AKs") == 12
    assert chen_score("22") == 5
    assert chen_score("72o") == -1


def test_the_bands_split_the_combinations_as_named():
    assert TIER_OF["AA"] == TIER_OF["KK"] == TIER_OF["AKs"] == TIERS[0]
    assert TIER_OF["72o"] == TIER_OF["32o"] == TIERS[-1]
    share = {tier: sum(combos(c) for c in ALL_CLASSES if TIER_OF[c] == tier) / 1326 for tier in TIERS}
    assert share[TIERS[0]] == pytest.approx(0.05, abs=0.03)
    assert share[TIERS[-1]] == pytest.approx(0.40, abs=0.05)


@pytest.mark.parametrize(
    ("hole", "board", "expected"),
    [
        ("7c 7d", "7h Kd 2s", "tris"),  # a set
        ("Ks 3c", "Kd 2h 8c", "top pair+"),
        ("Qs Qc", "Jd 2h 8c", "top pair+"),  # an overpair
        ("4s 4c", "Jd 9h 8c", "coppia"),  # an underpair
        ("8s 3c", "Kd 8h 2c", "coppia"),  # a middle pair
        ("Kc 8s", "Kd 8h 2c", "due coppie"),
        ("8s 3c", "8d 8h 2c", "tris"),  # trips with the board's pair
        ("Ah 3h", "Kh 7h 2c", "progetto"),  # a flush draw
        ("9s 8c", "7d 6h 2c", "progetto"),  # open-ended
        ("As 3c", "Kd 9h 7c", "nulla"),
        ("Ah 3h", "Kh 7h 2h", "mostro"),  # a flush
        ("Ac Kd", "2s 3h 4d 5c 6s", "nulla"),  # the board's straight plays
        ("7c Kd", "2s 3h 4d 5c 6s", "mostro"),  # a higher straight than the board's
        ("Ac Kd", "Qd Qh 7c 3s", "nulla"),  # only the board's pair, no draw
    ],
)
def test_what_a_hand_holds_after_the_flop(hole, board, expected):
    assert made_class(cards(hole), cards(board)) == expected


def _hand(starting: dict[int, int], button: int, actions, *, holes, board="", final=None) -> HandHistory:
    return HandHistory(
        schema_version=SCHEMA_VERSION, hand_id="h", started_at=0.0, num_players=len(starting),
        small_blind=1, big_blind=2, button_seat=button, starting_stacks=starting,
        seat_names={seat: f"p{seat}" for seat in starting}, community_cards=cards(board),
        actions=actions, hole_cards={seat: tuple(cards(h)) for seat, h in holes.items()},
        payouts={}, final_stacks=final or dict(starting),
    )


def act(street: Street, seat: int, kind: ActionType, amount: int, *, stack_after: int = 50) -> ActionRecord:
    return ActionRecord(street, seat, f"p{seat}", kind, amount, 100, stack_after, 3, 0.0)


SIX = {seat: 100 for seat in range(6)}
SIX_HOLES = {0: "Ah Ad", 1: "7c 2d", 2: "Kh Qh", 3: "9s 9c", 4: "5d 4d", 5: "Jc Tc"}


def test_positions_follow_the_order_of_play():
    assert positions(_hand(SIX, 0, [], holes=SIX_HOLES)) == {
        0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO",
    }
    heads_up = _hand({0: 100, 1: 100}, 1, [], holes={0: "Ah Ad", 1: "7c 2d"})
    assert positions(heads_up) == {1: "BTN", 0: "BB"}


def test_a_decision_knows_what_it_faced():
    """UTG opens, CO 3-bets, BTN shoves short (a call), the blinds fold, UTG calls."""
    p = Street.PREFLOP
    hand = _hand(SIX, 0, [
        act(p, 1, ActionType.POST_BLIND, 1), act(p, 2, ActionType.POST_BLIND, 2),
        act(p, 3, ActionType.RAISE, 6), act(p, 4, ActionType.FOLD, 0), act(p, 5, ActionType.RAISE, 18),
        act(p, 0, ActionType.ALL_IN, 10, stack_after=0), act(p, 1, ActionType.FOLD, 1),
        act(p, 2, ActionType.FOLD, 2), act(p, 3, ActionType.CALL, 18),
    ], holes=SIX_HOLES)
    found = {}
    for decision in decisions_of(hand):
        found.setdefault(decision.seat, decision)  # each seat's first decision
    assert (found[3].raises_before, found[3].to_call, found[3].kind) == (0, 2, "raise")
    assert (found[5].raises_before, found[5].to_call, found[5].kind) == (1, 6, "raise")
    assert (found[0].kind, found[0].all_in) == ("call", True)  # an all-in for less than the bet
    assert len(decisions_of(hand)) == 7  # every action but the blinds


def test_a_river_bet_is_scored_against_the_cards_still_in():
    hand = _hand({0: 100, 1: 100, 2: 100}, 0, [], holes={0: "Ah Kh", 1: "8c 5d", 2: "Qs Qd"},
                 board="Qc 9h 4s 3d 2c")
    p = Street.RIVER
    hand.actions.append(act(p, 0, ActionType.BET, 10))
    [decision] = decisions_of(hand)
    assert river_standing(hand, decision, [0, 2]) == "dietro"
    assert river_standing(hand, decision, [0, 1]) == "davanti"


def played_hands(count: int, seed: int = 3) -> list[HandHistory]:
    rng = random.Random(seed)
    players = [make_random_legal_bot(f"p{seat}", rng=rng) for seat in range(5)]
    table = Table(GameConfig(num_players=5, starting_stack=200, small_blind=1, big_blind=2), players,
                  rng=random.Random(seed))
    hands = []
    for _ in range(count):
        table.stacks = [rng.randint(20, 200) for _ in range(5)]
        hands.append(table.play_hand().hand_history)
    return hands


def test_every_voluntary_action_of_a_played_hand_is_a_decision():
    for hand in played_hands(60):
        voluntary = [r for r in hand.actions if r.action_type != ActionType.POST_BLIND]
        assert len(decisions_of(hand)) == len(voluntary)


def test_the_study_reads_played_hands_and_the_report_holds_every_section():
    hands = played_hands(400)
    study = analyse(hands, examples=1)
    assert study.hands == 400 and study.seat_hands == 2000
    # Chips are conserved, so what the seats win sums to zero.
    assert sum(sum(values) for values in study.results.values()) == pytest.approx(0.0, abs=1e-6)
    text = "\n".join(format_report(study, label="bot", min_count=5))
    for title in (
        "PER NUMERO DI GIOCATORI", "PER STACK EFFETTIVO", "PREFLOP", "VPIP", "POSTFLOP", "BLUFF",
        "slowplay", "RISULTATI", "ESEMPI",
    ):
        assert title in text


def test_every_seat_hand_falls_in_exactly_one_group_of_each_dimension():
    hands = played_hands(300)
    study = analyse(hands, examples=0)
    assert set(study.groups["giocatori"]) == {"4-6"}  # five-handed only
    for groups in study.groups.values():
        assert sum(group.seat_hands for group in groups.values()) == study.seat_hands
        assert sum(group.stats_chances[0] for group in groups.values()) == study.stats_chances[0]
        decisions = sum(group.preflop.total(key) for group in groups.values() for key in group.preflop.counts)
        assert decisions == sum(study.preflop.total(key) for key in study.preflop.counts)
    assert set(study.groups["stack"]) <= {"<10", "10-30", "30+"}


def test_the_effective_stack_is_the_smaller_of_its_own_and_the_deepest_other():
    hand = _hand({0: 2000, 1: 600, 2: 1400}, 0, [], holes={0: "Ah Ad", 1: "7c 2d", 2: "Kh Qh"})
    assert effective_stack_bb(hand, 0) == 700  # big blind 2: the deepest other has 1400 chips
    assert effective_stack_bb(hand, 1) == 300


def test_the_model_plays_against_itself_in_every_seat(tmp_path):
    pytest.importorskip("torch")
    from agent_study import PlayJob, play_job
    from support import tiny_model

    from pokerlab.rl.ppo import save_checkpoint

    path = tmp_path / "agent.pt"
    save_checkpoint(path, tiny_model())
    job = PlayJob(seed=1, hands=25, path=str(path), weights=(0, 1, 0, 0, 0, 0, 0, 0),
                  stack_min_bb=10.0, stack_max_bb=50.0, small_blind=50, big_blind=100,
                  session_hands=10, device="cpu")
    hands = play_job(job)
    assert len(hands) == 25
    assert {len(hand.starting_stacks) for hand in hands} == {3}
