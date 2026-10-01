"""Ratings: Elo arithmetic, persistence, seat filling and the K schedule.

No torch here on purpose -- `pool_registry` is pure Python so that the part
that decides *which models survive* is covered by the ordinary suite.
"""

from __future__ import annotations

import json

import pytest

from pokerlab.rl.pool_registry import (
    DEFAULT_K_FACTOR,
    DEFAULT_K_SCHEDULE,
    DEFAULT_RATING,
    PoolMember,
    PoolRegistry,
    interpolated_percentile,
    k_for_games,
    pairwise_elo_delta,
)


def make_registry(tmp_path, max_models: int = 3) -> PoolRegistry:
    return PoolRegistry(directory=tmp_path, max_models=max_models)


def add_model_file(
    registry: PoolRegistry, label: str, *, rating: float, games: int = 10, frozen: bool = False
) -> None:
    (registry.directory / f"{label}.pt").write_bytes(b"not a real checkpoint")
    registry.members[label] = PoolMember(
        label=label, kind="model", ref=f"{label}.pt", rating=rating, games=games, frozen=frozen
    )


# ---- Elo ----------------------------------------------------------------


def test_the_bigger_stack_gains_rating_and_the_smaller_one_loses_it():
    deltas = pairwise_elo_delta({"a": 50.0, "b": -50.0}, {"a": 1500.0, "b": 1500.0})
    assert deltas["a"] > 0
    assert deltas["b"] < 0


def test_evenly_matched_players_trade_nothing_on_a_chopped_pot():
    """A chopped pot leaves both stacks unchanged -- a real outcome in poker,
    not a degenerate case (see CLAUDE.md), so it must score as a draw."""
    deltas = pairwise_elo_delta({"a": 0.0, "b": 0.0}, {"a": 1500.0, "b": 1500.0})
    assert deltas["a"] == pytest.approx(0.0)
    assert deltas["b"] == pytest.approx(0.0)


def test_a_session_is_zero_sum_in_rating():
    results = {"a": 120.0, "b": -30.0, "c": -90.0}
    ratings = {"a": 1600.0, "b": 1500.0, "c": 1400.0}
    deltas = pairwise_elo_delta(results, ratings)
    assert sum(deltas.values()) == pytest.approx(0.0, abs=1e-9)


def test_beating_a_stronger_opponent_is_worth_more():
    underdog = pairwise_elo_delta({"a": 10.0, "b": -10.0}, {"a": 1200.0, "b": 1800.0})
    favourite = pairwise_elo_delta({"a": 10.0, "b": -10.0}, {"a": 1800.0, "b": 1200.0})
    assert underdog["a"] > favourite["a"]


def test_table_size_does_not_rescale_the_rating_step():
    """Without the per-opponent normalisation a 6-handed session would move a
    rating five times as far as a heads-up one, silently making K depend on
    how many seats happened to be filled."""
    heads_up = pairwise_elo_delta({"a": 10.0, "b": -10.0}, {"a": 1500.0, "b": 1500.0})
    six_max = pairwise_elo_delta(
        {"a": 10.0, **{f"o{i}": -10.0 for i in range(5)}},
        {"a": 1500.0, **{f"o{i}": 1500.0 for i in range(5)}},
    )
    assert six_max["a"] == pytest.approx(heads_up["a"], rel=1e-9)


def test_updates_are_computed_from_pre_session_ratings(tmp_path):
    """Every participant must be scored against the table as it stood before
    the session, otherwise the result depends on iteration order."""
    results = {"a": 30.0, "b": 10.0, "c": -40.0}
    ratings = {"a": 1500.0, "b": 1500.0, "c": 1500.0}
    forward = pairwise_elo_delta(results, ratings)
    reversed_order = pairwise_elo_delta(dict(reversed(list(results.items()))), ratings)
    for label in results:
        assert forward[label] == pytest.approx(reversed_order[label])


# ---- persistence -----------------------------------------


def test_the_registry_survives_a_save_load_round_trip(tmp_path):
    registry = make_registry(tmp_path)
    add_model_file(registry, "strong", rating=1800.0, games=12)
    add_model_file(registry, "weak", rating=1200.0, games=12)
    registry.record_session({"strong": 10.0, "weak": -10.0})
    registry.save()

    reloaded = PoolRegistry.load(tmp_path, max_models=3)
    assert reloaded.members["strong"].rating == pytest.approx(registry.members["strong"].rating)
    assert reloaded.members["strong"].games == registry.members["strong"].games
    assert "weak" in reloaded.members


def test_a_corrupt_registry_loads_as_empty_instead_of_raising(tmp_path):
    """Ratings are derived data that rebuild over a few sessions; refusing to
    start training over a bad file would be the worse failure."""
    (tmp_path / "registry.json").write_text("{not json", encoding="utf-8")
    assert PoolRegistry.load(tmp_path).members == {}


def test_an_unknown_field_in_a_stored_member_is_skipped_not_fatal(tmp_path):
    (tmp_path / "registry.json").write_text(
        json.dumps({"members": [{"label": "x", "kind": "model", "ref": "x.pt", "bogus": 1}]}),
        encoding="utf-8",
    )
    assert PoolRegistry.load(tmp_path).members == {}


# ---- seat filling --------------------------------------------------------


def test_fill_slots_returns_exactly_the_requested_count(tmp_path):
    registry = make_registry(tmp_path, max_models=20)
    add_model_file(registry, "a", rating=1900.0)
    assert len(registry.fill_slots(20)) == 20


def test_a_registry_can_hold_a_model_directly(tmp_path):
    registry = make_registry(tmp_path)
    add_model_file(registry, "m", rating=1720.0)
    assert registry.members["m"].rating == 1720.0
    assert registry.models()[0].label == "m"


def test_fill_slots_prefers_models_then_duplicates(tmp_path):
    registry = make_registry(tmp_path, max_models=20)
    add_model_file(registry, "a", rating=1900.0)
    add_model_file(registry, "b", rating=1800.0)

    slots = registry.fill_slots(9)

    assert [m.label for m in slots[:2]] == ["a", "b"]
    # only the two models exist, so everything past them is a repeat
    assert {m.label for m in slots[2:]} <= {"a", "b"}


def test_fill_slots_is_empty_when_there_is_nothing_to_seat(tmp_path):
    assert make_registry(tmp_path).fill_slots(6) == []


# ---- recording -----------------------------------------------------------


def test_recording_a_session_moves_ratings_and_counts_games(tmp_path):
    registry = make_registry(tmp_path)
    add_model_file(registry, "winner", rating=1500.0, games=0)
    add_model_file(registry, "loser", rating=1500.0, games=0)

    registry.record_session({"winner": 80.0, "loser": -80.0})

    assert registry.members["winner"].rating > 1500.0
    assert registry.members["loser"].rating < 1500.0
    assert registry.members["winner"].games == 1


def test_an_unregistered_participant_is_rated_but_not_persisted(tmp_path):
    """The live learner plays every evaluation session but is not a pool
    member; the caller carries its rating forward itself."""
    registry = make_registry(tmp_path)
    add_model_file(registry, "member", rating=1500.0)

    deltas = registry.record_session({"learner": 50.0, "member": -50.0})

    assert deltas["learner"] > 0
    assert "learner" not in registry.members


# ---- frozen members --------------------------------------------------------


def test_a_frozen_members_rating_never_moves_but_games_still_counts(tmp_path):
    registry = make_registry(tmp_path)
    add_model_file(registry, "anchor", rating=1500.0, games=5, frozen=True)
    add_model_file(registry, "challenger", rating=1500.0, games=5)

    registry.record_session({"anchor": -80.0, "challenger": 80.0})

    assert registry.members["anchor"].rating == 1500.0
    assert registry.members["anchor"].games == 6
    assert registry.members["challenger"].rating > 1500.0


def test_a_frozen_members_pinned_rating_still_shapes_its_opponents_delta(tmp_path):
    """A frozen anchor still counts as a real opponent for everyone else's
    Elo math -- only its own rating is held fixed."""
    weak_anchor = make_registry(tmp_path)
    add_model_file(weak_anchor, "anchor", rating=1200.0, games=5, frozen=True)
    add_model_file(weak_anchor, "challenger", rating=1500.0, games=5)
    weak_deltas = weak_anchor.record_session({"anchor": -10.0, "challenger": 10.0})

    strong_anchor = make_registry(tmp_path)
    add_model_file(strong_anchor, "anchor", rating=1800.0, games=5, frozen=True)
    add_model_file(strong_anchor, "challenger", rating=1500.0, games=5)
    strong_deltas = strong_anchor.record_session({"anchor": -10.0, "challenger": 10.0})

    # Beating a stronger anchor is worth more than beating a weaker one.
    assert strong_deltas["challenger"] > weak_deltas["challenger"]


# ---- percentiles ------------------------------------------------------------


def test_interpolated_percentile_of_a_single_value():
    assert interpolated_percentile([42.0], 37.0) == 42.0


def test_interpolated_percentile_median_of_an_odd_length_list():
    assert interpolated_percentile([10, 30, 20], 50) == 20.0


def test_interpolated_percentile_median_of_an_even_length_list():
    assert interpolated_percentile([10, 20, 30, 40], 50) == 25.0


def test_interpolated_percentile_extremes_are_min_and_max():
    values = [5, 1, 9, 3]
    assert interpolated_percentile(values, 0) == 1.0
    assert interpolated_percentile(values, 100) == 9.0


def test_interpolated_percentile_matches_a_hand_computed_reference():
    """[10, 20, 30, 40] at the 25th percentile: rank = 0.25 * 3 = 0.75, so
    3/4 of the way from 10 to 20 -- 17.5. A concrete anchor so a future
    change to the interpolation method shows up as a failing assertion,
    not a silent drift."""
    assert interpolated_percentile([10, 20, 30, 40], 25) == 17.5


def test_interpolated_percentile_rejects_an_empty_sequence():
    with pytest.raises(ValueError):
        interpolated_percentile([], 50)


# ---- population-scale elimination -------------------------------------------


def test_eliminate_lowest_rated_removes_the_worst_up_to_count(tmp_path):
    registry = make_registry(tmp_path, max_models=10**9)
    for i in range(20):
        add_model_file(registry, f"m{i:03d}", rating=1000.0 + i, games=50)

    doomed = registry.eliminate_lowest_rated(count=5, games_threshold=0)

    assert {m.label for m in doomed} == {"m000", "m001", "m002", "m003", "m004"}
    assert len(registry.members) == 15


def test_eliminate_lowest_rated_ignores_games_below_the_threshold(tmp_path):
    registry = make_registry(tmp_path, max_models=10**9)
    for i in range(19):
        add_model_file(registry, f"veteran{i:03d}", rating=1500.0, games=100)
    # Worst rating in the whole registry, but far too few games to trust it.
    add_model_file(registry, "newcomer", rating=1.0, games=1)

    doomed = registry.eliminate_lowest_rated(count=5, games_threshold=50)

    assert "newcomer" not in {m.label for m in doomed}
    assert "newcomer" in registry.members


def test_eliminate_lowest_rated_threshold_is_inclusive(tmp_path):
    """`games_threshold` must accept a member with exactly that many games,
    not only strictly more."""
    registry = make_registry(tmp_path, max_models=10**9)
    add_model_file(registry, "at_threshold", rating=1.0, games=50)
    add_model_file(registry, "above_threshold", rating=2.0, games=200)

    doomed = registry.eliminate_lowest_rated(count=2, games_threshold=50)

    assert {m.label for m in doomed} == {"at_threshold", "above_threshold"}


def test_eliminate_lowest_rated_caps_rather_than_requiring_the_full_count(tmp_path):
    """Fewer than `count` qualify: every one of them goes rather than none --
    the alternative (all-or-nothing) can stall forever when `count` is sized
    against a much larger population than what is ever rated at once."""
    registry = make_registry(tmp_path, max_models=10**9)
    for i in range(3):
        add_model_file(registry, f"m{i:03d}", rating=1000.0 + i, games=50)

    doomed = registry.eliminate_lowest_rated(count=100, games_threshold=0)

    assert len(doomed) == 3
    assert len(registry.members) == 0


def test_eliminate_lowest_rated_excludes_frozen_members(tmp_path):
    registry = make_registry(tmp_path, max_models=10**9)
    for i in range(5):
        add_model_file(registry, f"m{i:03d}", rating=1000.0 + i, games=50)
    # Lowest-rated of all, plenty of games -- must never be touched.
    add_model_file(registry, "anchor", rating=1.0, games=1000, frozen=True)

    doomed = registry.eliminate_lowest_rated(count=10, games_threshold=0)

    assert "anchor" not in {m.label for m in doomed}
    assert "anchor" in registry.members


def test_eliminate_lowest_rated_restricts_to_among_when_given(tmp_path):
    """`among` lets the caller exclude labels whose checkpoint no longer
    exists on disk, even though the registry itself has no way to know that."""
    registry = make_registry(tmp_path, max_models=10**9)
    add_model_file(registry, "gone", rating=1.0, games=50)
    add_model_file(registry, "present", rating=2.0, games=50)

    doomed = registry.eliminate_lowest_rated(count=10, games_threshold=0, among={"present"})

    assert {m.label for m in doomed} == {"present"}
    assert "gone" in registry.members


def test_eliminate_lowest_rated_never_touches_the_checkpoint_file_itself(tmp_path):
    """The registry method only ever edits `self.members` -- the actual
    deletion of a non-promoted doomed model's file is the caller's job
    (`global_arena.delete_checkpoints`), not this method's."""
    registry = make_registry(tmp_path, max_models=10**9)
    for i in range(20):
        add_model_file(registry, f"m{i:03d}", rating=1000.0 + i, games=50)

    doomed = registry.eliminate_lowest_rated(count=5, games_threshold=0)

    assert doomed
    for member in doomed:
        assert (registry.directory / f"{member.label}.pt").exists()
    assert not (registry.directory / "retired").exists()


# ---- K-factor that falls with experience ---------------------------------------


def test_k_falls_through_the_tiers_as_games_accumulate():
    games = (0, 19, 20, 44, 45, 79, 80, 139, 140, 499, 500, 749, 750,
             2499, 2500, 60_000, 500_000)
    ks = [k_for_games(g) for g in games]
    assert ks == [
        16.0, 16.0, 11.0, 11.0, 7.4, 7.4, 5.0, 5.0, 3.4, 1.6, 1.05, 1.05, 0.72,
        0.33, 0.22, 0.01, 0.01,
    ]
    assert ks == sorted(ks, reverse=True)  # never rises with experience


def test_the_staircase_tracks_the_hyperbolic_gain_it_approximates():
    """The tiers are not chosen by feel: they are a 10-step staircase on
    `K_t = 1/(slope*(t + V/P0))`, the Kalman gain for a rating measured by
    1000-hand sessions (slope 0.001421, V = 238^2, offset 35.5 from inheriting the
    parent's rating). Ten steps were measured to cost 3% of final precision
    against the exact curve; a tier that drifted far off it would cost much more,
    so each one is checked against the value at the middle of its own range."""
    slope, offset = 0.0014208, 35.5
    hyperbolic = lambda t: 1.0 / (slope * (t + offset))
    thresholds = [t for t, _ in DEFAULT_K_SCHEDULE]
    last = len(DEFAULT_K_SCHEDULE) - 1
    for index, (lo, value) in enumerate(DEFAULT_K_SCHEDULE):
        hi = thresholds[index + 1] if index + 1 < len(thresholds) else lo * 2
        middle = ((lo + offset) * (hi + offset)) ** 0.5 - offset
        ratio = value / hyperbolic(middle)
        # The bottom tier is open-ended, so it cannot track a falling curve and is
        # held to a loose bound; every other tier is within 5%, which is what going
        # from 10 tiers to 20 bought (the 10-tier version swung 25% inside a tier).
        bound = 1.25 if index == last else 1.05
        assert 1 / bound < ratio < bound, f"tier at {lo} games is {ratio:.2f}x the curve"


def test_the_schedule_has_twenty_steps_and_bottoms_out_at_a_hundredth():
    """Twenty was the number asked for, and the two ends are what fix the ratio
    between the tiers: 16 / 1.4745^19 = 0.01. Adding tiers is free precision;
    removing them is not, and moving either end changes every tier between."""
    assert len(DEFAULT_K_SCHEDULE) == 20
    assert DEFAULT_K_SCHEDULE[0][1] == 16.0
    assert DEFAULT_K_SCHEDULE[-1][1] == 0.01
    ratios = [
        DEFAULT_K_SCHEDULE[i][1] / DEFAULT_K_SCHEDULE[i + 1][1]
        for i in range(len(DEFAULT_K_SCHEDULE) - 1)
    ]
    assert all(1.35 < r < 1.6 for r in ratios), "a constant factor per tier, near 1.4745"


def test_the_schedule_burns_in_fast_and_then_settles_below_the_flat_k():
    """The schedule used to be uniformly gentler than the flat K, on the grounds
    that the flat one was for ratings that must converge in a handful of sessions
    and the schedule for ratings accumulated over hundreds. That split was wrong
    at one end: a rating starting at 1500 has ~80 points to travel before it says
    anything, and neither 3.0 nor a flat 8 could carry it there inside a run. The
    schedule now does both jobs -- a bounded burn-in first, then the settled
    tiers -- which is why the learner's own evaluation was moved onto it."""
    assert k_for_games(0) == DEFAULT_K_SCHEDULE[0][1]
    # Fast where a rating is still unknown...
    assert k_for_games(0) > DEFAULT_K_FACTOR
    # ...and it has to come down inside a run, or it is not a burn-in but a
    # permanently jumpy rating. A run plays ~100 rated sessions of validation
    # before it ever reaches the anchors, so by then it must already be gentler
    # than the flat K.
    assert k_for_games(100) < DEFAULT_K_FACTOR
    # And by the end of the round against the anchors, materially gentler still:
    # that is the distinction the schedule exists for -- hold the order steady
    # rather than chase the last session.
    assert k_for_games(600) <= DEFAULT_K_FACTOR / 4


def test_a_veteran_moves_less_than_a_newcomer_at_the_same_table(tmp_path):
    registry = PoolRegistry(directory=tmp_path, k_schedule=DEFAULT_K_SCHEDULE)
    registry.members["vet"] = PoolMember(label="vet", kind="model", ref="v.pt", games=1500)
    registry.members["new"] = PoolMember(label="new", kind="model", ref="n.pt", games=0)

    registry.record_session({"vet": 50.0, "new": -50.0})

    vet_gain = registry.members["vet"].rating - DEFAULT_RATING
    new_loss = DEFAULT_RATING - registry.members["new"].rating
    # Written off `k_for_games` rather than as literals: the tiers have moved
    # five times and a literal only pins whichever value it happened to agree
    # with. What this test is about is the ordering, not the numbers.
    assert vet_gain == pytest.approx(k_for_games(1500) * 0.5)
    assert new_loss == pytest.approx(k_for_games(0) * 0.5)
    assert vet_gain < new_loss, "which is the whole point"


def test_without_a_schedule_every_member_uses_the_flat_k(tmp_path):
    registry = PoolRegistry(directory=tmp_path)
    registry.members["vet"] = PoolMember(label="vet", kind="model", ref="v.pt", games=1500)
    registry.members["new"] = PoolMember(label="new", kind="model", ref="n.pt", games=0)

    registry.record_session({"vet": 50.0, "new": -50.0})

    # Equally rated, so the expected score is 0.5 and the winner moves by half
    # the flat K. Written off the constant rather than as a literal: the value
    # has moved once (24 -> 12) and what this test is about is that *both*
    # members use the same one, whatever it is.
    assert registry.members["vet"].rating == pytest.approx(DEFAULT_RATING + DEFAULT_K_FACTOR * 0.5)
    assert registry.members["new"].rating == pytest.approx(DEFAULT_RATING - DEFAULT_K_FACTOR * 0.5)


def test_a_session_is_rated_at_the_experience_the_model_had_when_it_sat_down(tmp_path):
    """games is incremented by the session itself; the K must come from before
    that, so a session played on the last game of a tier is rated at that tier and
    not at the next one. Taken at a real tier boundary, or the test would pass
    whichever value were used."""
    boundary = DEFAULT_K_SCHEDULE[3][0]
    registry = PoolRegistry(directory=tmp_path, k_schedule=DEFAULT_K_SCHEDULE)
    registry.members["a"] = PoolMember(label="a", kind="model", ref="a.pt", games=boundary - 1)
    registry.members["b"] = PoolMember(label="b", kind="model", ref="b.pt", games=boundary - 1)

    registry.record_session({"a": 10.0, "b": -10.0})

    assert k_for_games(boundary - 1) != k_for_games(boundary), "a real boundary"
    expected = DEFAULT_RATING + k_for_games(boundary - 1) * 0.5
    assert registry.members["a"].rating == pytest.approx(expected)
    assert registry.members["a"].games == boundary


def test_an_unregistered_participant_keeps_the_default_k_under_a_schedule(tmp_path):
    registry = PoolRegistry(directory=tmp_path, k_schedule=DEFAULT_K_SCHEDULE)
    registry.members["vet"] = PoolMember(label="vet", kind="model", ref="v.pt", games=1500)

    deltas = registry.record_session({"vet": 50.0, "learner": -50.0})

    assert deltas["vet"] == pytest.approx(k_for_games(1500) * 0.5)
    assert deltas["learner"] == pytest.approx(DEFAULT_K_FACTOR * -0.5)


def test_a_caller_can_give_an_unregistered_participant_its_own_k(tmp_path):
    """The learner of a training run is deliberately never a member -- it changes
    every iteration, so persisting it would rate a moving target -- which leaves
    the schedule with no `games` to read for it. `SelfPlayTrainer` counts its own
    rated sessions and passes the K that follows, so the learner walks the same
    staircase as a registered model instead of sitting on the flat fallback."""
    registry = PoolRegistry(directory=tmp_path, k_schedule=DEFAULT_K_SCHEDULE)
    registry.members["vet"] = PoolMember(label="vet", kind="model", ref="v.pt", games=1500)
    results = {"vet": -50.0, "learner": 50.0}
    ratings = {"vet": DEFAULT_RATING, "learner": DEFAULT_RATING}

    deltas = registry.record_session_with_ratings(
        results, ratings, k_factors={"learner": 24.0}
    )

    assert deltas["learner"] == pytest.approx(24.0 * 0.5)
    # The override names one participant and must not disturb anyone else's K,
    # which still comes from their own games.
    assert deltas["vet"] == pytest.approx(k_for_games(1500) * -0.5)


def test_an_override_wins_over_the_schedule_for_the_same_label(tmp_path):
    registry = PoolRegistry(directory=tmp_path, k_schedule=DEFAULT_K_SCHEDULE)
    registry.members["vet"] = PoolMember(label="vet", kind="model", ref="v.pt", games=1500)
    registry.members["other"] = PoolMember(label="other", kind="model", ref="o.pt", games=1500)

    deltas = registry.record_session_with_ratings(
        {"vet": 50.0, "other": -50.0},
        {"vet": DEFAULT_RATING, "other": DEFAULT_RATING},
        k_factors={"vet": 10.0},
    )

    assert deltas["vet"] == pytest.approx(10.0 * 0.5)
    assert deltas["other"] == pytest.approx(k_for_games(1500) * -0.5)


def test_the_k_schedule_is_gentle_enough_to_keep_the_top_in_order():
    """The settled tiers were lowered three times after the ranking was shown to
    be wrong at the top: measured with duplicate decks, the #20 lost to the #100
    by 37 bb/100 (t = -6.1) although the ranking put it 16 points higher. The
    route was 24/16/12/8/6 -> 12/8/6/4/3 -> 4.0/1.0/0.5/0.1 -> 3.0/1.0/0.3/0.1 ->
    a burn-in prepended to that -> ten steps -> today's twenty, which replace the
    whole thing with an approximation of the optimal gain. A K this size is what
    keeps a well-played rating from being shoved around by one session; at 0.22 it
    is nearly held still, which is the intended trade and also the cost."""
    assert DEFAULT_K_SCHEDULE == (
        (0, 16.0), (20, 11.0), (45, 7.4), (80, 5.0), (140, 3.4),
        (225, 2.3), (325, 1.6), (500, 1.05), (750, 0.72), (1200, 0.49),
        (1700, 0.33), (2500, 0.22), (3750, 0.15), (5500, 0.10), (8500, 0.07),
        (12000, 0.047), (18000, 0.032), (27500, 0.022), (40000, 0.015), (60000, 0.01),
    )
    thresholds = [t for t, _ in DEFAULT_K_SCHEDULE]
    values = [v for _, v in DEFAULT_K_SCHEDULE]
    assert thresholds == sorted(thresholds) and values == sorted(values, reverse=True)
    # The tiers that decide anything have to be reachable by the models the
    # schedule can actually *move* -- the non-frozen ones, measured at median 911
    # rated games, 90th percentile ~1,780 and max 2,727 over the 9,433 on the
    # volume. Twelve of the twenty tiers sit inside that range and the rest are
    # deliberate headroom, reached by nobody who can move: the schedule follows the
    # curve down to K = 0.01 at ~70,000 games because the population's game counts
    # only grow, and grow faster than they used to now that a model publishes with
    # the ~600 rated sessions it earned rather than with zero.
    reachable = [t for t in thresholds if t < 2727]
    assert len(reachable) >= 12, "the live part of the schedule must stay fine-grained"
    assert len(reachable) > len(thresholds) / 2, "most tiers live, not headroom"
    assert thresholds[-1] > 2727
    # The tier a model lands in on publication day: ~100 validation sessions plus
    # the round against the anchors. It must be well down the schedule -- that is
    # the whole reason the count continues from training -- but not at the floor,
    # or later population rounds could never move it again.
    published = k_for_games(600)
    assert values[-1] < published < values[0] / 4


def test_a_veteran_moves_far_less_than_a_newcomer_for_the_same_result():
    """The whole point of the schedule: the same session must not shove a rating
    built on a thousand games as hard as one built on ten."""
    results = {"new": 100.0, "old": -100.0}
    ratings = {"new": 1500.0, "old": 1500.0}
    deltas = pairwise_elo_delta(
        results, ratings, k_factors={"new": k_for_games(0), "old": k_for_games(1500)}
    )
    assert deltas["new"] > 0 and deltas["old"] < 0
    assert abs(deltas["new"]) > 3 * abs(deltas["old"])
