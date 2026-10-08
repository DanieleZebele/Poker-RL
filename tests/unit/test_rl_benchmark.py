"""The frozen anchors and the pass that rates a model against them."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from support import fixed_mix, tiny_model

from pokerlab.engine.stats import STATS
from pokerlab.rl.benchmark import (
    anchor_paths,
    rate_against_benchmark,
)
from pokerlab.rl.ppo import save_checkpoint

GAME = fixed_mix(3)


# ---- rating a fresh model against each frozen series ------------------------


def make_series(root, series, per_series=3):
    """The frozen set as it is stored on disk: `benchmark_<N>/` directories.

    The pass no longer cares which directory a checkpoint sits in -- it draws
    opponents from the whole set -- but the layout is still what pruning creates,
    so the fixture keeps it.
    """
    for index in range(series):
        directory = root / f"benchmark_{index + 1}"
        directory.mkdir(parents=True, exist_ok=True)
        for n in range(per_series):
            save_checkpoint(
                directory / f"s{index + 1}-m{n}.pt",
                tiny_model(hidden=16, num_layers=1),
            )
    return root


def test_every_frozen_checkpoint_is_a_candidate_whatever_directory_it_is_in(tmp_path):
    """The pass draws from the whole set, so discovery is flat and recursive.
    Sorted, because `rglob` order is filesystem order and a seeded draw has to
    repeat across the machines sharing the volume."""
    make_series(tmp_path / "benchmark", series=3, per_series=2)
    (tmp_path / "benchmark" / "loose.pt").write_bytes(b"x")

    found = anchor_paths(tmp_path / "benchmark")

    assert found == sorted(found), "sorted as paths, which is what a seeded draw repeats"
    assert len(found) == 7, "three series of two, plus the loose root file"
    assert {p.name for p in found} >= {"loose.pt", "s1-m0.pt", "s3-m1.pt"}


def test_a_missing_benchmark_directory_is_empty_not_an_error(tmp_path):
    assert anchor_paths(tmp_path / "nope") == []


def test_the_pass_rates_the_model_over_every_session(tmp_path):
    game = fixed_mix(3)
    make_series(tmp_path / "benchmark", series=3, per_series=2)

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=6, hands=3, resident=4,
        rotate_every=2, seed=1,
    )

    assert rated is not None
    assert rated.sessions == 6 and rated.hands == 18
    assert len(rated.raw_sessions) == 6
    assert all("fresh" in session for session in rated.raw_sessions)
    assert rated.rating_after != 1500.0, "the rating must actually move"


def test_the_pass_reports_how_the_model_played_over_all_its_hands(tmp_path):
    game = fixed_mix(3)
    make_series(tmp_path / "benchmark", series=3, per_series=2)

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=4, hands=3, resident=4, seed=1,
    )

    assert rated.style_hands == 12  # the learner sat in every hand of every session
    assert list(rated.style) == list(STATS)
    events, chances = rated.style["vpip"]
    assert chances == 12 and 0 <= events <= chances


def test_the_session_count_continues_from_training(tmp_path):
    """The K a session is rated at comes from how many rated sessions the learner
    has played *in total*, validation passes included -- that continuity is the
    whole reason the pass takes `games` rather than starting at zero."""
    game = fixed_mix(3)
    make_series(tmp_path / "benchmark", series=2, per_series=2)

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, games=100, sessions=4, hands=3, seed=1,
    )

    assert rated is not None
    assert rated.games_after == 104


def test_a_later_session_moves_the_rating_less_than_an_early_one(tmp_path):
    """The schedule's falling K is what makes 500 sessions worth 500 sessions of
    evidence. A flat K settles at a fixed jitter however many are played."""
    from pokerlab.rl.pool_registry import k_for_games

    assert k_for_games(0) > k_for_games(100) > k_for_games(600) > k_for_games(2000)


def test_a_frozen_set_too_small_to_seat_a_table_is_not_fatal(tmp_path):
    game = fixed_mix(6)
    make_series(tmp_path / "benchmark", series=1, per_series=2)  # need 5, have 2
    skipped = []

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=1, hands=2, seed=1,
        on_skip=lambda path, why: skipped.append((path, why)),
    )

    assert rated is None, "a run with no anchors still publishes, it just is not rated here"
    assert skipped and "ancore" in skipped[0][1]


def test_an_empty_benchmark_directory_yields_nothing(tmp_path):
    game = fixed_mix(3)
    assert rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "nope", game,
        label="fresh", rating=1500.0, sessions=1, hands=2,
    ) is None


def test_the_pass_reports_its_progress_session_by_session(tmp_path):
    """The pass plays 500,000 hands and prints almost nothing, so without this
    the 10-minute stale-log warning fires on a perfectly healthy worker."""
    game = fixed_mix(3)
    make_series(tmp_path / "benchmark", series=2, per_series=2)
    seen = []

    rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=4, hands=2, seed=1,
        on_progress=lambda done, total, detail: seen.append((done, total)),
    )

    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_a_broken_checkpoint_is_struck_off_rather_than_reported_every_slice(tmp_path):
    """A bad file must be tried once, not once per slice: ten slices would
    otherwise report the same failure ten times, and a slice that happened to be
    entirely bad could spin."""
    game = fixed_mix(3)
    make_series(tmp_path / "benchmark", series=2, per_series=3)
    (tmp_path / "benchmark" / "benchmark_1" / "broken.pt").write_bytes(b"not a checkpoint")
    skipped = []

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=8, hands=2, resident=3,
        rotate_every=1, seed=1,
        on_skip=lambda path, why: skipped.append(path.name),
    )

    assert rated is not None and rated.sessions == 8
    assert skipped.count("broken.pt") == 1


def test_the_pass_is_deliberately_not_queued_for_the_global_merge(tmp_path):
    """`raw_sessions` is offered but `train.main()` does not queue it, because the
    evidence is already in the rating the model publishes with: re-applying it at
    the merge measured 10-20% worse. This pins the property that makes that safe
    -- the pass hands back the session count so `publish_model` can register the
    games it really earned."""
    game = fixed_mix(3)
    make_series(tmp_path / "benchmark", series=2, per_series=2)

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, games=100, sessions=3, hands=2, seed=1,
    )

    assert rated is not None
    assert rated.games_after == 100 + rated.sessions
    assert len(rated.raw_sessions) == rated.sessions


def test_the_pass_rests_on_enough_hands_to_mean_something():
    """500 sessions of 1,000 hands is 500,000, a 95% interval of about +/-8
    bb/100 -- which is what lets `benchmark_bb100` be the sweep's response
    variable. The old arrangement's 1,000-hand readings carried +/-180."""
    from pokerlab.rl.benchmark import DEFAULT_BENCHMARK_SESSIONS, DEFAULT_SESSION_HANDS

    assert DEFAULT_BENCHMARK_SESSIONS * DEFAULT_SESSION_HANDS >= 400_000


def test_one_parameter_sets_the_session_length_everywhere():
    """The Elo scale is defined by how often a session of this length picks the
    stronger model, so a pass of a different length would be a different scale
    silently sharing the same numbers. There is one length -- `--session-hands`,
    1,000 by default -- and every program that plays rated sessions reads it from
    the same flag, with the same default."""
    from importlib import import_module

    from pokerlab.config import _actions
    from pokerlab.rl.benchmark import rate_against_benchmark
    from pokerlab.rl.siblings import TORCH_FREE, WITH_TORCH
    from pokerlab.rl.table_mix import DEFAULT_SESSION_HANDS

    assert DEFAULT_SESSION_HANDS == 1000
    for name in TORCH_FREE + WITH_TORCH:
        if name.endswith("dashboard"):
            continue  # plays nothing
        action = _actions(import_module(name).build_parser())["session_hands"]
        assert action.default == DEFAULT_SESSION_HANDS, name
    import inspect

    assert inspect.signature(rate_against_benchmark).parameters["hands"].default == DEFAULT_SESSION_HANDS


def test_the_published_session_count_keeps_a_new_model_out_of_the_top_tier():
    """A model published with the ~600 rated sessions it really played is refined
    gently by later population passes. Published at zero it would be shoved
    around at the schedule's first tier on evidence it already has."""
    from pokerlab.rl.benchmark import DEFAULT_BENCHMARK_SESSIONS
    from pokerlab.rl.pool_registry import DEFAULT_K_SCHEDULE, k_for_games

    earned = 100 + DEFAULT_BENCHMARK_SESSIONS
    assert k_for_games(earned) < k_for_games(0)
    assert k_for_games(earned) > DEFAULT_K_SCHEDULE[-1][1], "and not in the bottom tier either"


# ---- a mixture of table sizes ------------------------------------------------


def test_the_pass_draws_a_table_size_per_session_from_the_mixture(tmp_path):
    from pokerlab.rl.table_mix import TableMix

    mix = TableMix(weights=(1, 1, 1, 0, 0, 0, 0, 0), stack_min_bb=5.0, stack_max_bb=50.0,
                   small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=2, per_series=2)  # four anchors: enough for 4-handed

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", mix,
        label="fresh", rating=1500.0, sessions=30, hands=2, seed=3,
    )

    assert rated is not None and rated.sessions == 30
    # Every session is one table: its size is the number of participants, and it
    # is one the mixture allows.
    assert {len(session) for session in rated.raw_sessions} == {2, 3, 4}
    assert set(rated.bb_per_100_by_size) == {2, 3, 4}


def test_the_sizes_a_pass_plays_are_reproducible_from_the_seed(tmp_path):
    from pokerlab.rl.table_mix import TableMix

    mix = TableMix(weights=(1, 1, 1, 0, 0, 0, 0, 0), stack_min_bb=5.0, stack_max_bb=50.0,
                   small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=2, per_series=2)

    def sizes(seed):
        rated = rate_against_benchmark(
            tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", mix,
            label="fresh", rating=1500.0, sessions=12, hands=1, seed=seed,
        )
        return [len(session) for session in rated.raw_sessions]

    assert sizes(5) == sizes(5)


def test_a_frozen_set_that_cannot_seat_the_largest_table_is_not_rated(tmp_path):
    """A pass that quietly dropped the big tables would measure a different
    mixture than the weights say, so it does not run at all."""
    from pokerlab.rl.table_mix import TableMix

    mix = TableMix(weights=(1, 0, 0, 0, 0, 0, 0, 1), stack_min_bb=5.0, stack_max_bb=50.0,
                   small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=1, per_series=5)  # a 9-handed table needs 8
    skipped = []

    rated = rate_against_benchmark(
        tiny_model(hidden=16, num_layers=1), tmp_path / "benchmark", mix,
        label="fresh", rating=1500.0, sessions=2, hands=1, seed=1,
        on_skip=lambda path, why: skipped.append(why),
    )

    assert rated is None and skipped
