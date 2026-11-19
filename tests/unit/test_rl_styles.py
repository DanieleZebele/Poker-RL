"""`rl/styles.py`: the opponents' styles, a push on the logits by state. Pure Python."""

from __future__ import annotations

import random

import pytest

from pokerlab.cards.card import Card
from pokerlab.engine.state import Street
from pokerlab.players.base import Observation
from pokerlab.rl.action_space import ACTION_DIM, ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN, RAISE_MIN_BIN
from pokerlab.rl.hand_tiers import ALL_CLASSES, TIER_OF, TIERS, chen_score, combos, hand_class
from pokerlab.rl.styles import (
    AXES,
    GROUPS,
    SITUATIONS,
    Style,
    draw_style,
    group_of,
    situation_of,
    style_fn,
)


def state(street: Street, *, hole: str = "9h 8h", to_call: int = 0) -> Observation:
    return Observation(
        street=street, hole_cards=tuple(Card.parse(c) for c in hole.split()),
        community_cards=(), pot_size=10, current_bet_to_match=to_call, min_raise=2, my_seat=0,
        my_stack=100, my_current_bet=0, seats=(), button_seat=0, action_history=(),
    )


def style(**axes: float) -> Style:
    return Style(tuple(axes.get(name, 0.0) for name in AXES))


def test_the_bins_fall_in_their_groups():
    assert [group_of(i) for i in (FOLD_BIN, CHECK_CALL_BIN, RAISE_MIN_BIN, ALL_IN_BIN)] == [
        "fold", "call", "small", "allin"]
    assert group_of(ALL_IN_BIN - 1) == "big"
    assert {group_of(i) for i in range(ACTION_DIM)} == set(GROUPS)


def test_the_situation_is_preflop_checked_to_or_facing_a_bet():
    assert situation_of(state(Street.PREFLOP, to_call=2)) == "preflop"
    assert situation_of(state(Street.FLOP)) == "checked"
    assert situation_of(state(Street.TURN, to_call=6)) == "facing"


def test_a_loose_player_folds_less_preflop_and_only_preflop():
    push = style(larghezza=1.0).push(state(Street.PREFLOP))
    assert push[FOLD_BIN] < 0 < push[CHECK_CALL_BIN]
    assert style(larghezza=1.0).push(state(Street.FLOP, to_call=4)) == [0.0] * ACTION_DIM


def test_looseness_moves_marginal_hands_more_than_the_best_ones():
    marginal = style(larghezza=1.0).push(state(Street.PREFLOP, hole="9h 8h"))[FOLD_BIN]
    aces = style(larghezza=1.0).push(state(Street.PREFLOP, hole="Ah Ad"))[FOLD_BIN]
    assert TIER_OF[hand_class(state(Street.PREFLOP, hole="Ah Ad").hole_cards)] == TIERS[0]
    assert abs(aces) < abs(marginal)


def test_a_tenacious_player_calls_more_only_facing_a_bet():
    facing = style(tenacia=1.0).push(state(Street.RIVER, to_call=8))
    assert facing[FOLD_BIN] < 0 < facing[CHECK_CALL_BIN]
    assert style(tenacia=1.0).push(state(Street.RIVER)) == [0.0] * ACTION_DIM


def test_the_sign_and_the_scale_of_an_axis_are_linear():
    observation = state(Street.FLOP)
    one = style(aggressivita_postflop=1.0).push(observation)
    assert style(aggressivita_postflop=-1.0).push(observation) == [-x for x in one]
    scales = [1.0] * len(AXES)
    scales[AXES.index("aggressivita_postflop")] = 3.0
    assert style(aggressivita_postflop=1.0).push(observation, scales) == pytest.approx([3 * x for x in one])


def test_the_noise_shifts_its_own_situation_only():
    jitter = [0.0] * (len(SITUATIONS) * len(GROUPS))
    jitter[SITUATIONS.index("checked") * len(GROUPS) + GROUPS.index("big")] = 0.7
    noisy = Style(jitter=tuple(jitter))
    assert noisy.push(state(Street.FLOP))[ALL_IN_BIN - 1] == pytest.approx(0.7)
    assert noisy.push(state(Street.PREFLOP)) == [0.0] * ACTION_DIM


def test_a_style_refuses_the_wrong_shape():
    with pytest.raises(ValueError):
        Style((0.0,))
    with pytest.raises(ValueError):
        Style(temperature=0.0)


def test_drawn_styles_are_mostly_mild_and_within_bounds():
    rng = random.Random(3)
    drawn = [draw_style(rng, spread=0.5, temperature_spread=0.3, jitter=0.1) for _ in range(2000)]
    values = [v for s in drawn for v in s.axes]
    assert all(-1.0 <= v <= 1.0 for v in values)
    assert sum(abs(v) < 0.5 for v in values) / len(values) > 0.6
    assert all(0.5 - 1e-9 <= s.temperature <= 2.0 + 1e-9 for s in drawn)
    assert len({s.axes for s in drawn}) == len(drawn)  # no two alike


def test_style_fn_hands_the_push_and_the_temperature():
    observation = state(Street.PREFLOP)
    bias, temperature = style_fn(Style(temperature=1.5), [1.0] * len(AXES))(observation)
    assert bias == [0.0] * ACTION_DIM and temperature == 1.5


def test_the_169_starting_hands_and_their_bands():
    assert len(ALL_CLASSES) == 169 and sum(combos(c) for c in ALL_CLASSES) == 1326
    assert chen_score("AA") == 20 and chen_score("72o") == -1
    assert TIER_OF["AA"] == TIERS[0] and TIER_OF["72o"] == TIERS[-1]


def test_a_switched_off_config_takes_nothing_from_the_rng():
    from pokerlab.rl.styles import StyleConfig

    rng, untouched = random.Random(5), random.Random(5)
    assert StyleConfig().draw(rng) is None
    assert rng.random() == untouched.random()


def test_the_share_decides_how_many_seats_get_a_style():
    from pokerlab.rl.styles import StyleConfig

    rng = random.Random(6)
    config = StyleConfig(share=0.3)
    styled = sum(config.draw(rng) is not None for _ in range(4000))
    assert styled / 4000 == pytest.approx(0.3, abs=0.03)
    assert all(StyleConfig(share=1.0).draw(rng) is not None for _ in range(20))


def test_a_style_config_refuses_nonsense():
    from pokerlab.rl.styles import StyleConfig

    for bad in ({"share": 1.5}, {"spread": -0.1}, {"scales": (1.0,)}, {"scales": (1.0, 1.0, 1.0, 1.0, -1.0)}):
        with pytest.raises(ValueError):
            StyleConfig(**bad)


def test_the_style_flags_round_trip_through_the_loop_to_a_worker():
    import argparse

    from pokerlab.rl.styles import (
        StyleConfig,
        add_style_arguments,
        style_arguments,
        style_config_from_args,
    )

    config = StyleConfig(share=0.25, spread=0.4, temperature_spread=0.1, jitter=0.05, scales=(1.5, 1.0, 0.5, 2.0, 1.0))
    parser = argparse.ArgumentParser()
    add_style_arguments(parser)
    assert style_config_from_args(parser.parse_args(style_arguments(config))) == config
    assert style_config_from_args(parser.parse_args([])) == StyleConfig()


def test_a_style_setting_that_cannot_be_used_is_refused_before_anything_starts(tmp_path):
    """`poker-train` and the supervisor both check the styles at startup (with the network
    arguments), so a wrong `style_scales` in the shared file stops them with a message
    instead of killing every worker."""
    import argparse

    from pokerlab.rl.styles import add_style_arguments
    from pokerlab.rl.train import check_network_arguments

    equity = tmp_path / "equity.pt"
    equity.write_bytes(b"x")
    parser = argparse.ArgumentParser()
    add_style_arguments(parser)
    shape = {"hidden": 8, "num_layers": 1, "head_hidden": 8, "head_layers": 0, "critic_bins": 0, "equity_model": str(equity),
             "feature_version": 3, "belief_model": "", "belief_oracle": 0.0, "belief_oracle_iterations": 0, "belief_oracle_mode": "mix", "belief_oracle_near": [64], "belief_oracle_mode": "mix", "belief_oracle_near": [64]}
    good = argparse.Namespace(**shape, **vars(parser.parse_args([])))
    check_network_arguments(good)
    for flags in (["--style-scales", "1", "2"], ["--style-share", "2"]):
        bad = argparse.Namespace(**shape, **vars(parser.parse_args(flags)))
        with pytest.raises(ValueError):
            check_network_arguments(bad)


# ---- the axes of detail ----------------------------------------------------------------


def _observations(size: int = 6, hands: int = 30, seed: int = 0) -> list[Observation]:
    """Every observation of a few hands of random players."""
    from support import make_random_legal_bot

    from pokerlab.engine.config import GameConfig
    from pokerlab.engine.table import Table
    from pokerlab.players.base import Player

    seen: list[Observation] = []

    class Keep(Player):
        def __init__(self, seat):
            super().__init__(f"p{seat}", f"p{seat}")
            self.bot = make_random_legal_bot(f"p{seat}", rng=random.Random(seed * 10 + seat))

        def act(self, observation, legal):
            seen.append(observation)
            return self.bot.act(observation, legal)

    table = Table(GameConfig(size, 2000, 10, 20), [Keep(s) for s in range(size)], rng=random.Random(seed))
    for _ in range(hands):
        table.stacks = [2000] * size
        table.play_hand()
    return seen


def detailed(**axes: float) -> Style:
    return Style(tuple(axes.get(name, 0.0) for name in AXES))


def test_five_scales_are_the_first_five_axes_and_leave_the_detail_off():
    from pokerlab.rl.styles import DETAIL_AXES, LEGACY_AXES, StyleConfig, full_scales

    assert AXES[: len(LEGACY_AXES)] == LEGACY_AXES and len(AXES) == len(LEGACY_AXES) + len(DETAIL_AXES)
    assert full_scales([5, 8, 8, 4, 8]) == (5, 8, 8, 4, 8) + (0.0,) * len(DETAIL_AXES)
    assert StyleConfig(scales=(5.0, 8.0, 8.0, 4.0, 8.0)).scales[len(LEGACY_AXES):] == (0.0,) * len(DETAIL_AXES)
    every_axis = Style((1.0,) * len(AXES))
    for observation in _observations(hands=10):
        old = Style((1.0,) * len(LEGACY_AXES) + (0.0,) * len(DETAIL_AXES)).push(observation, (5, 8, 8, 4, 8))
        assert every_axis.push(observation, (5, 8, 8, 4, 8)) == old  # the detail is off


def test_the_button_acts_last_the_first_seat_after_the_blinds_first():
    from pokerlab.rl.styles import position_of

    for observation in _observations(size=6, hands=10):
        where = position_of(observation)
        assert -1.0 <= where <= 1.0
        if observation.my_seat == observation.button_seat:
            assert where == 1.0
    heads_up = _observations(size=2, hands=5)
    assert {position_of(o) for o in heads_up} <= {0.0, 1.0}


def test_the_preflop_raises_are_counted_and_mine_is_known():
    from pokerlab.engine.actions import ActionType
    from pokerlab.rl.styles import preflop_raises

    for observation in _observations(hands=20):
        raises, mine = preflop_raises(observation)
        preflop = [r for r in observation.action_history if r.street is Street.PREFLOP]
        bets = [r for r in preflop if r.action_type in (ActionType.BET, ActionType.RAISE)]
        assert raises >= len(bets)
        if any(r.seat == observation.my_seat for r in bets):
            assert mine


@pytest.mark.parametrize(
    ("hole", "board", "expected"),
    [
        ("Ah Kd", "2c 7s 9h", "debole"),
        ("Ah Kd", "Ac 7s 9h", "media"),
        ("9d 9c", "9h 7s 2c", "forte"),
        ("Ah 7d", "Ac 7s 2h", "forte"),
        ("Ah Kd", "7c 7s 2h", "debole"),  # the pair is the board's
        ("Ah 2d", "7c 7s 2h", "media"),  # two pair, one of them the board's
        ("Ah Kh", "Qh Jh 2h", "forte"),
    ],
)
def test_the_strength_of_a_hand_is_what_my_cards_add_to_the_board(hole, board, expected):
    from pokerlab.rl.styles import strength_of

    cards = lambda text: tuple(Card.parse(c) for c in text.split())
    assert strength_of(cards(hole), cards(board)) == expected


def test_each_axis_of_detail_pushes_only_where_it_belongs():
    from pokerlab.rl.styles import (
        DETAIL_AXES,
        SHORT_FROM_BB,
        position_of,
        preflop_raises,
        strength_of,
    )

    for observation in _observations(size=6, hands=40, seed=3):
        situation = situation_of(observation)
        for axis in DETAIL_AXES:
            push = detailed(**{axis: 1.0}).push(observation)
            moved = any(push)
            if axis.startswith("aggressivita_"):
                assert moved == (observation.street.value == axis.split("_")[1])
            elif axis in ("bluff", "slowplay"):
                if moved:
                    strength = strength_of(tuple(observation.hole_cards), tuple(observation.community_cards))
                    assert situation != "preflop" and strength == ("debole" if axis == "bluff" else "forte")
            elif axis in ("tre_bet", "fold_al_tre_bet") and moved:
                raises, mine = preflop_raises(observation)
                assert situation == "preflop"
                assert (raises == 1 and not mine) if axis == "tre_bet" else (raises >= 2 and mine)
            elif axis == "check_raise" and moved:
                assert situation == "facing"
            elif axis == "posizione" and position_of(observation) == 0.0:
                assert not moved
            elif axis == "shove_corto":
                left_bb = (observation.my_stack + observation.my_current_bet) / observation.big_blind
                assert moved == (left_bb < SHORT_FROM_BB)  # what is left of the stack, mid-hand


def test_a_short_stack_shoves_more_the_shorter_it_is():
    from dataclasses import replace

    from pokerlab.rl.action_space import ALL_IN_BIN as SHOVE
    from pokerlab.rl.styles import SHORT_FROM_BB, SHORT_FULL_BB

    base = replace(_observations(hands=1)[0], big_blind=20)
    pushes = [
        detailed(shove_corto=1.0).push(replace(base, my_stack=int(bb * 20), my_current_bet=0))[SHOVE]
        for bb in (SHORT_FROM_BB + 5, SHORT_FROM_BB - 5, SHORT_FULL_BB, 2)
    ]
    assert pushes[0] == 0.0 < pushes[1] < pushes[2] == pushes[3] == 1.0
