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
    shape = {"hidden": 8, "num_layers": 1, "head_hidden": 8, "head_layers": 0, "equity_model": str(equity)}
    good = argparse.Namespace(**shape, **vars(parser.parse_args([])))
    check_network_arguments(good)
    for flags in (["--style-scales", "1", "2"], ["--style-share", "2"]):
        bad = argparse.Namespace(**shape, **vars(parser.parse_args(flags)))
        with pytest.raises(ValueError):
            check_network_arguments(bad)
