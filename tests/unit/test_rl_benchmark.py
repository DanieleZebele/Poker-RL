"""The frozen benchmark: the only cross-generation measurement in the project.

Its whole value rests on being reproducible -- a number that wobbles between two
runs of the same model cannot tell a real improvement from noise -- so that is
what most of these assert.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pokerlab.engine.config import GameConfig
from pokerlab.rl.benchmark import (
    BenchmarkResult,
    anchor_paths,
    load_benchmark_opponents,
    rate_against_benchmark,
    run_benchmark,
)
from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.ppo import save_checkpoint

GAME = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)


@pytest.fixture
def benchmark_dir(tmp_path):
    for index in range(4):
        torch.manual_seed(index)
        save_checkpoint(tmp_path / f"bench-{index}.pt", PokerActorCritic(hidden=32, num_layers=1))
    return tmp_path


def small_model(seed: int) -> PokerActorCritic:
    torch.manual_seed(seed)
    return PokerActorCritic(hidden=32, num_layers=1)


def test_the_set_loads_in_a_fixed_order(benchmark_dir):
    """Sorted by name, not by mtime: copying the directory must not change the
    measurement."""
    labels = [o.label for o in load_benchmark_opponents(benchmark_dir, GAME)]
    assert labels == sorted(labels) == ["bench-0", "bench-1", "bench-2", "bench-3"]


def test_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert load_benchmark_opponents(tmp_path / "nope", GAME) == []


def test_nested_series_directories_are_included_by_default(benchmark_dir):
    """The root benchmark directory is now just a container: all numbered
    benchmark_<N>/ subdirectories are part of the same fixed benchmark set."""
    nested = benchmark_dir / "benchmark_1"
    nested.mkdir()
    save_checkpoint(nested / "promoted.pt", PokerActorCritic(hidden=32, num_layers=1))
    labels = [o.label for o in load_benchmark_opponents(benchmark_dir, GAME)]
    assert "promoted" in labels
    assert len(labels) == 5


def test_recursive_can_still_be_disabled_explicitly(benchmark_dir):
    """The recursive default is the intended behaviour, but callers can still
    opt into root-only selection when they need a stable, non-growing set."""
    nested = benchmark_dir / "benchmark_1"
    nested.mkdir()
    save_checkpoint(nested / "promoted.pt", PokerActorCritic(hidden=32, num_layers=1))
    labels = [o.label for o in load_benchmark_opponents(benchmark_dir, GAME, recursive=False)]
    assert "promoted" not in labels
    assert len(labels) == 4


def test_an_unreadable_checkpoint_is_skipped_with_a_reason(benchmark_dir):
    (benchmark_dir / "broken.pt").write_text("not a checkpoint")
    skipped: list[str] = []
    opponents = load_benchmark_opponents(
        benchmark_dir, GAME, on_skip=lambda path, why: skipped.append(path.name)
    )
    assert skipped == ["broken.pt"]
    assert len(opponents) == 4


def test_the_same_model_scores_identically_twice(benchmark_dir):
    """The property the whole benchmark rests on."""
    opponents = load_benchmark_opponents(benchmark_dir, GAME)
    model = small_model(7)
    first = run_benchmark(model, opponents, GAME, hands=60, seed=99)
    second = run_benchmark(model, opponents, GAME, hands=60, seed=99)
    assert first.bb_per_100 == second.bb_per_100
    assert first.won_chips == second.won_chips


def test_different_models_score_differently(benchmark_dir):
    """A perfectly stable number that never moves would measure nothing."""
    opponents = load_benchmark_opponents(benchmark_dir, GAME)
    first = run_benchmark(small_model(7), opponents, GAME, hands=120, seed=99)
    second = run_benchmark(small_model(123), opponents, GAME, hands=120, seed=99)
    assert first.bb_per_100 != second.bb_per_100


def test_a_different_seed_deals_different_hands(benchmark_dir):
    opponents = load_benchmark_opponents(benchmark_dir, GAME)
    model = small_model(7)
    assert (
        run_benchmark(model, opponents, GAME, hands=60, seed=1).bb_per_100
        != run_benchmark(model, opponents, GAME, hands=60, seed=2).bb_per_100
    )


def test_too_few_opponents_is_a_clear_error(tmp_path):
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "only.pt", PokerActorCritic(hidden=32, num_layers=1))
    opponents = load_benchmark_opponents(tmp_path, GAME)
    with pytest.raises(ValueError, match="benchmark has 1 opponents"):
        run_benchmark(small_model(1), opponents, GAME, hands=10)


def test_the_result_reports_what_was_played(benchmark_dir):
    opponents = load_benchmark_opponents(benchmark_dir, GAME)
    result = run_benchmark(small_model(7), opponents, GAME, hands=100, seed=5)
    assert isinstance(result, BenchmarkResult)
    assert result.hands == 100
    assert result.opponents == 4
    assert result.per_opponent, "no per-opponent breakdown was produced"


def test_every_benchmark_opponent_is_actually_faced(benchmark_dir):
    """A deterministic walk must still cover the set, not circle two of them."""
    opponents = load_benchmark_opponents(benchmark_dir, GAME)
    result = run_benchmark(small_model(7), opponents, GAME, hands=400, seed=5, rotate_every=10)
    assert set(result.per_opponent) == {o.label for o in opponents}


# ---- rating a fresh model against each frozen series ------------------------


def make_series(root, series, per_series=3):
    """The frozen set as it is stored on disk: `benchmark_<N>/` directories.

    The round no longer cares which directory a checkpoint sits in -- it draws
    opponents from the whole set -- but the layout is still what pruning creates,
    so the fixture keeps it.
    """
    from pokerlab.rl.policy import PokerActorCritic
    from pokerlab.rl.ppo import save_checkpoint

    for index in range(series):
        directory = root / f"benchmark_{index + 1}"
        directory.mkdir(parents=True, exist_ok=True)
        for n in range(per_series):
            save_checkpoint(
                directory / f"s{index + 1}-m{n}.pt",
                PokerActorCritic(hidden=16, num_layers=1),
            )
    return root


def test_every_frozen_checkpoint_is_a_candidate_whatever_directory_it_is_in(tmp_path):
    """The round draws from the whole set, so discovery is flat and recursive.
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


def test_the_round_rates_the_model_over_every_session(tmp_path):
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=3, per_series=2)

    rated = rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=6, hands=3, resident=4,
        rotate_every=2, seed=1,
    )

    assert rated is not None
    assert rated.sessions == 6 and rated.hands == 18
    assert len(rated.raw_sessions) == 6
    assert all("fresh" in session for session in rated.raw_sessions)
    assert rated.rating_after != 1500.0, "the rating must actually move"


def test_the_session_count_continues_from_training(tmp_path):
    """The K a session is rated at comes from how many rated sessions the learner
    has played *in total*, validation rounds included -- that continuity is the
    whole reason the round takes `games` rather than starting at zero."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=2, per_series=2)

    rated = rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "benchmark", game,
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
    game = GameConfig(num_players=6, starting_stack=100, small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=1, per_series=2)  # need 5, have 2
    skipped = []

    rated = rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=1, hands=2, seed=1,
        on_skip=lambda path, why: skipped.append((path, why)),
    )

    assert rated is None, "a run with no anchors still publishes, it just is not rated here"
    assert skipped and "ancore" in skipped[0][1]


def test_an_empty_benchmark_directory_yields_nothing(tmp_path):
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    assert rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "nope", game,
        label="fresh", rating=1500.0, sessions=1, hands=2,
    ) is None


def test_the_round_reports_its_progress_session_by_session(tmp_path):
    """The round plays 500,000 hands and prints almost nothing, so without this
    the 10-minute stale-log warning fires on a perfectly healthy worker."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=2, per_series=2)
    seen = []

    rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=4, hands=2, seed=1,
        on_progress=lambda done, total, detail: seen.append((done, total)),
    )

    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_a_broken_checkpoint_is_struck_off_rather_than_reported_every_slice(tmp_path):
    """A bad file must be tried once, not once per slice: ten slices would
    otherwise report the same failure ten times, and a slice that happened to be
    entirely bad could spin."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=2, per_series=3)
    (tmp_path / "benchmark" / "benchmark_1" / "broken.pt").write_bytes(b"not a checkpoint")
    skipped = []

    rated = rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, sessions=8, hands=2, resident=3,
        rotate_every=1, seed=1,
        on_skip=lambda path, why: skipped.append(path.name),
    )

    assert rated is not None and rated.sessions == 8
    assert skipped.count("broken.pt") == 1


def test_the_round_is_deliberately_not_queued_for_the_global_merge(tmp_path):
    """`raw_sessions` is offered but `train.main()` does not queue it, because the
    evidence is already in the rating the model publishes with: re-applying it at
    the merge measured 10-20% worse. This pins the property that makes that safe
    -- the round hands back the session count so `publish_model` can register the
    games it really earned."""
    game = GameConfig(num_players=3, starting_stack=100, small_blind=1, big_blind=2)
    make_series(tmp_path / "benchmark", series=2, per_series=2)

    rated = rate_against_benchmark(
        PokerActorCritic(hidden=16, num_layers=1), tmp_path / "benchmark", game,
        label="fresh", rating=1500.0, games=100, sessions=3, hands=2, seed=1,
    )

    assert rated is not None
    assert rated.games_after == 100 + rated.sessions
    assert len(rated.raw_sessions) == rated.sessions


def test_the_round_rests_on_enough_hands_to_mean_something():
    """500 sessions of 1,000 hands is 500,000, a 95% interval of about +/-8
    bb/100 -- which is what lets `benchmark_bb100` be the sweep's response
    variable. The old arrangement's 1,000-hand readings carried +/-180."""
    from pokerlab.rl.benchmark import DEFAULT_BENCHMARK_SESSIONS, DEFAULT_SESSION_HANDS

    assert DEFAULT_BENCHMARK_SESSIONS * DEFAULT_SESSION_HANDS >= 400_000


def test_a_session_is_a_thousand_hands_everywhere():
    """The Elo scale is defined by how often a session of this length picks the
    stronger model, so a round of a different length would be a different scale
    silently sharing the same numbers."""
    from pokerlab.rl.benchmark import DEFAULT_SESSION_HANDS
    from pokerlab.rl.global_arena import DEFAULT_HANDS_PER_GAME
    from pokerlab.rl.train import SESSION_HANDS

    assert DEFAULT_SESSION_HANDS == SESSION_HANDS == DEFAULT_HANDS_PER_GAME == 1000


def test_the_published_session_count_keeps_a_new_model_out_of_the_top_tier():
    """A model published with the ~600 rated sessions it really played is refined
    gently by later population rounds. Published at zero it would be shoved
    around at the schedule's first tier on evidence it already has."""
    from pokerlab.rl.benchmark import DEFAULT_BENCHMARK_SESSIONS
    from pokerlab.rl.pool_registry import DEFAULT_K_SCHEDULE, k_for_games

    earned = 100 + DEFAULT_BENCHMARK_SESSIONS
    assert k_for_games(earned) < k_for_games(0)
    assert k_for_games(earned) > DEFAULT_K_SCHEDULE[-1][1], "and not in the bottom tier either"
