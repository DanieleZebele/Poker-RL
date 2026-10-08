"""How many hands a session needs before the stronger model wins it.

`studies/agents/duel_power.py` splits in two: a torch-free statistics half that decides what
the experiment concludes, and a playing half that only produces the numbers it
chews on. This file tests the first half, which is where a wrong answer would
actually come from -- and, like the rest of the ranking arithmetic, it needs no
extra dependency to run.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest
from duel_power import (
    DuelMeasurement,
    block_outcomes,
    format_distributions,
    format_report,
    hands_for_probability,
    histogram,
    normal_cdf,
    normal_quantile,
    power_table,
    sample_sessions,
    wilson_interval,
    win_probability,
)


def test_block_outcomes_counts_wins_draws_and_losses():
    outcome = block_outcomes([[1.0, 1.0, -3.0, 1.0, 0.0, 0.0]], hands=2)
    assert (outcome.sessions, outcome.wins, outcome.draws, outcome.losses) == (3, 1, 1, 1)
    # A draw is half a win for each side, the Elo convention.
    assert outcome.score == 1.5
    assert outcome.rate == 0.5


def test_a_trailing_partial_block_is_dropped_not_counted_short():
    """Seven hands hold three sessions of two, not three and a half.

    Scoring a remainder as if it were a full session would silently mix two
    different session lengths into one row of the table.
    """
    outcome = block_outcomes([[1.0] * 7], hands=2)
    assert outcome.sessions == 3
    assert outcome.wins == 3


def test_blocks_never_straddle_two_streams():
    """Each stream is one seating from one deal sequence: a block spanning two
    of them would not be a session anyone could have played."""
    outcome = block_outcomes([[1.0, -5.0], [1.0, -5.0]], hands=2)
    assert outcome.sessions == 2
    assert outcome.losses == 2


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = wilson_interval(20, 20)
    assert 0.0 <= low <= 1.0 and high == 1.0
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0.0 <= high <= 1.0


def test_normal_quantile_inverts_the_cdf():
    for p in (0.01, 0.25, 0.5, 0.75, 0.95, 0.99):
        assert normal_cdf(normal_quantile(p)) == pytest.approx(p, abs=1e-9)


def test_win_probability_grows_with_session_length():
    edge, sigma = 0.3, 30.0
    probabilities = [win_probability(edge, sigma, hands) for hands in (10, 100, 1000, 100000)]
    assert probabilities == sorted(probabilities)
    assert probabilities[0] < 0.55  # ten hands say essentially nothing
    assert probabilities[-1] > 0.99  # a hundred thousand say everything


def test_no_edge_means_no_session_length_ever_decides():
    assert win_probability(0.0, 30.0, 10**9) == 0.5
    assert hands_for_probability(0.0, 30.0, 0.95) is None
    assert hands_for_probability(-1.0, 30.0, 0.95) is None


def test_hands_for_probability_is_the_inverse_of_win_probability():
    edge, sigma = 0.26, 29.0
    for target in (0.75, 0.9, 0.95, 0.99):
        hands = hands_for_probability(edge, sigma, target)
        assert win_probability(edge, sigma, hands) == pytest.approx(target, abs=1e-3)


def test_the_needed_hands_scale_with_the_square_of_the_noise_over_the_edge():
    """Halving the edge quadruples the hands. This is the whole reason a short
    session cannot rank two close models: the cost is quadratic in how close
    they are."""
    wide = hands_for_probability(0.4, 30.0, 0.95)
    narrow = hands_for_probability(0.2, 30.0, 0.95)
    assert narrow == pytest.approx(4 * wide, rel=0.01)


def _measurement(normal, mirrored=()):
    return DuelMeasurement(
        label_a="A", label_b="B", seats_per_team=3, big_blind=2,
        normal=[list(s) for s in normal], mirrored=[list(s) for s in mirrored],
    )


def test_mirroring_keeps_the_estimate_and_sharpens_it():
    """Duplicate decks do not move the measured edge -- they measure it better.

    The skill term is a flat +2 per hand and the card luck is huge and
    symmetric, so the plain estimate is buried in noise while the mirrored one
    reads the +2 straight off.
    """
    rng = random.Random(7)
    skill = 2.0
    luck = [rng.gauss(0.0, 200.0) for _ in range(4000)]
    normal = [skill + value for value in luck]
    mirrored = [skill - value for value in luck]

    plain_edge, plain_error = _measurement([normal]).edge()
    mirrored_edge, mirrored_error = _measurement([normal], [mirrored]).edge()

    assert mirrored_edge == pytest.approx(skill, abs=1e-9)
    assert mirrored_error == pytest.approx(0.0, abs=1e-9)
    # The same hands, without the mirror: the edge is not even distinguishable
    # from zero.
    assert abs(plain_edge - skill) > mirrored_error
    assert plain_error > 1.0


def test_the_table_reports_the_stronger_model_even_when_it_is_b():
    """The question is "how often does the stronger model win", so a negative
    edge has to flip the whole table rather than report B losing."""
    rows = power_table(_measurement([[-1.0] * 100]), lengths=(10,), orientation=-1)
    assert rows[0].measured.wins == 10
    assert rows[0].measured.losses == 0


def test_a_measured_frequency_matches_the_predicted_one():
    """The empirical column and the analytic column are two ways of answering
    the same question, and they have to agree once a session is long enough for
    the central limit theorem to hold. This is the check that the extrapolation
    past the hands actually played is not fiction."""
    rng = random.Random(11)
    edge, sigma, hands = 1.0, 20.0, 400
    streams = [[rng.gauss(edge, sigma) for _ in range(hands * 200)]]
    measurement = _measurement(streams)
    row = power_table(measurement, lengths=(hands,), orientation=1)[0]
    assert row.measured.rate == pytest.approx(row.predicted, abs=0.05)


def test_the_report_warns_when_it_does_not_know_which_model_is_stronger():
    rng = random.Random(3)
    noise = _measurement([[rng.gauss(0.0, 50.0) for _ in range(500)]])
    text = "\n".join(format_report(noise, lengths=(10, 100)))
    assert "ATTENZIONE" in text

    decided = _measurement([[10.0 + rng.gauss(0.0, 1.0) for _ in range(500)]])
    assert "ATTENZIONE" not in "\n".join(format_report(decided, lengths=(10, 100)))


def test_the_report_shows_a_prediction_for_lengths_nobody_played():
    """400 hands cannot be chopped into a 100,000-hand session, and that row is
    exactly the one worth reading."""
    rng = random.Random(5)
    measurement = _measurement([[rng.gauss(1.0, 30.0) for _ in range(400)]])
    lines = format_report(measurement, lengths=(100, 100000))
    row = next(line for line in lines if "100,000" in line)
    assert "%" in row


def test_sessions_are_drawn_round_robin_across_streams():
    """A hundred consecutive blocks out of one stream are a hundred sessions
    from one seating. Taking block 0 of every stream first spends the
    independence the streams were played for."""
    streams = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]
    assert sample_sessions(streams, hands=1, limit=3) == [1.0, 10.0, 100.0]
    assert sample_sessions(streams, hands=1, limit=5) == [1.0, 10.0, 100.0, 2.0, 20.0]


def test_asking_for_more_sessions_than_were_played_gives_what_there_is():
    assert len(sample_sessions([[1.0] * 10], hands=4, limit=100)) == 2


def test_a_session_length_nobody_played_yields_nothing():
    assert sample_sessions([[1.0] * 10], hands=100, limit=5) == []


def test_zero_is_always_a_bin_edge_of_the_histogram():
    """The sign of a session is the only thing the ranking reads, so a bin
    straddling zero would hide exactly the split being counted."""
    lines = histogram([-90.0, -10.0, 5.0, 60.0, 120.0], bins=6)
    assert any("zero" in line for line in lines)
    for line in lines:
        if "zero" in line or "|" not in line:
            continue
        low, _, rest = line.partition("..")
        start = float(low.strip().replace(",", ""))
        end = float(rest.split("|")[0].strip().replace(",", ""))
        assert not (start < 0 < end), f"il bin {start}..{end} scavalca lo zero"


def test_every_value_lands_in_exactly_one_bin():
    rng = random.Random(17)
    values = [rng.gauss(30.0, 200.0) for _ in range(500)]
    counted = sum(
        int(line.rsplit(maxsplit=1)[-1])
        for line in histogram(values)
        if "|" in line and "zero" not in line
    )
    assert counted == len(values)


def test_the_distribution_tightens_as_sessions_get_longer():
    """The point of the whole exercise, as a test: the same edge measured over
    longer sessions is the same median and a smaller spread, and that is what
    turns a scattered result into a reliable one."""
    rng = random.Random(23)
    edge, sigma = 1.0, 40.0
    measurement = _measurement([[rng.gauss(edge, sigma) for _ in range(200_000)]])
    spreads = []
    for hands in (100, 1000, 10000):
        drawn = sample_sessions(measurement.streams, hands, limit=100)
        results = [value / hands for value in drawn]
        spreads.append(statistics.stdev(results))
    assert spreads == sorted(spreads, reverse=True)
    # Ten times the hands, about a third of the spread.
    assert spreads[0] / spreads[1] == pytest.approx(math.sqrt(10), rel=0.25)


def test_the_summary_table_reports_one_row_per_length():
    rng = random.Random(29)
    measurement = _measurement([[rng.gauss(2.0, 50.0) for _ in range(20_000)]])
    lines = format_distributions(measurement, lengths=(100, 1000), sessions=20)
    text = "\n".join(lines)
    assert "20 partite da 100 mani" in text
    assert "20 partite da 1,000 mani" in text
    assert "riepilogo" in text


def test_the_duel_plays_its_hands_with_opponent_statistics_like_every_rated_session():
    from duel_power import _play

    from pokerlab.engine.config import GameConfig
    from pokerlab.players.rl_agent import PolicyDecision
    from pokerlab.rl.features import (
        CARDS_DIM,
        FIELD_SCALARS_DIM,
        POT_SCALARS_DIM,
        SEAT_BASE_FEATURES,
        STREET_DIM,
    )

    flag = CARDS_DIM + STREET_DIM + POT_SCALARS_DIM + FIELD_SCALARS_DIM + SEAT_BASE_FEATURES
    seen: list[float] = []
    rng = random.Random(1)

    def policy(features, mask):
        seen.append(features[flag])
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    game = GameConfig(num_players=3, starting_stack=200, small_blind=1, big_blind=2)
    _play(game, dict.fromkeys(range(3), policy), plus=[0], minus=[1, 2], hands=30, seed=4)
    assert seen[0] == 0.0 and seen[-1] == 1.0
