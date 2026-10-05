"""Who a training run plays against, and how models reach the shared store.

No torch: the draw, the publication and the ranking reads are pure Python.
"""

from __future__ import annotations

import json
import random

import pytest

from pokerlab.rl.global_store import (
    load_ranking,
    publish_model,
    read_member,
    read_sidecar,
    write_member,
    write_sidecar,
    write_snapshot,
)
from pokerlab.rl.pool_registry import DEFAULT_RATING, PoolMember, PoolRegistry
from pokerlab.rl.training_pool import (
    available_labels,
    build_training_registry,
    draw_training_pool,
    format_parent_tiers,
    parse_parent_tiers,
    pick_parents,
)


def ranking_of(**groups):
    """Build a ranking from `name=(count, rating_of_first, games)` groups."""
    ranking: dict[str, PoolMember] = {}
    for prefix, (count, top_rating, games) in groups.items():
        for index in range(count):
            label = f"{prefix}{index:03d}"
            ranking[label] = PoolMember(
                label=label, kind="model", ref=f"{label}.pt", rating=top_rating - index, games=games
            )
    return ranking


def store(*labels_lists):
    return [label for labels in labels_lists for label in labels]


# ---- the draw ---------------------------------------------------------------------


def test_half_the_seats_come_from_the_top_and_the_rest_from_anywhere():
    """The top share is exact; the remaining seats are uniform over the whole
    store, so they land overwhelmingly outside a top-100 that is a fraction of
    it -- the field is a strong half plus an honest random half, never only the
    current elite."""
    ranking = ranking_of(top=(100, 2000.0, 200), mid=(300, 1500.0, 50))
    labels = list(ranking) + [f"new{i:03d}" for i in range(100)]  # 100 never-rated models

    drawn = draw_training_pool(ranking, labels, size=20, rng=random.Random(0))

    assert len(drawn) == 20 and len({m.label for m in drawn}) == 20
    picked = [m.label for m in drawn]
    assert sum(label.startswith("top") for label in picked) >= 10  # the top share
    assert sum(not label.startswith("top") for label in picked) >= 5  # drawn at large


def test_the_top_share_is_the_only_thing_that_favours_anyone():
    """Nothing biases the remaining seats -- not the least-played models. Over many draws from a store whose
    never-rated models are a fifth of it, they take about a fifth of those
    seats."""
    ranking = ranking_of(top=(100, 2000.0, 200), mid=(300, 1500.0, 50))
    labels = list(ranking) + [f"new{i:03d}" for i in range(100)]

    newcomers = sum(
        label.startswith("new")
        for seed in range(40)
        for label in (m.label for m in draw_training_pool(ranking, labels, size=20,
                                                          rng=random.Random(seed)))
    )

    assert 0.10 < newcomers / (40 * 10) < 0.35  # a fifth of the 10 non-top seats, not a share


def test_two_runs_with_different_seeds_face_different_fields():
    ranking = ranking_of(top=(100, 2000.0, 200), mid=(300, 1500.0, 50))
    labels = list(ranking)

    fields = {
        frozenset(m.label for m in draw_training_pool(ranking, labels, rng=random.Random(seed)))
        for seed in range(6)
    }

    assert len(fields) == 6, "every run must draw its own pool"


def test_the_same_seed_reproduces_the_same_draw():
    ranking = ranking_of(top=(100, 2000.0, 200), mid=(100, 1500.0, 50))
    first = draw_training_pool(ranking, list(ranking), rng=random.Random(3))
    second = draw_training_pool(ranking, list(ranking), rng=random.Random(3))
    assert [m.label for m in first] == [m.label for m in second]


def test_drawn_members_are_frozen_copies_pointing_into_the_store():
    ranking = ranking_of(top=(30, 2000.0, 200))

    drawn = draw_training_pool(ranking, list(ranking), size=10, rng=random.Random(0))

    assert all(m.frozen for m in drawn)
    assert all(m.ref == f"{m.label}.pt" for m in drawn)
    assert not any(ranking[m.label].frozen for m in drawn)  # the ranking itself is untouched
    drawn[0].rating = 1.0
    assert ranking[drawn[0].label].rating != 1.0


def test_a_model_the_ranking_has_never_seen_is_drawn_as_a_blank_unrated_one():
    drawn = draw_training_pool({}, ["brand-new"], size=5, rng=random.Random(0))

    assert [m.label for m in drawn] == ["brand-new"]
    assert drawn[0].rating == DEFAULT_RATING and drawn[0].games == 0


def test_a_small_store_yields_everything_it_has_and_an_empty_one_nothing():
    ranking = ranking_of(a=(3, 1600.0, 20))
    assert {m.label for m in draw_training_pool(ranking, list(ranking), size=20, rng=random.Random(0))} == set(ranking)
    assert draw_training_pool(ranking, [], size=20, rng=random.Random(0)) == []
    assert draw_training_pool(ranking, list(ranking), size=0, rng=random.Random(0)) == []


def test_a_short_top_source_hands_its_seats_to_the_random_one():
    """No model has been rated yet: the top share has nobody to draw, and the
    pool must still come out full."""
    labels = [f"new{i:03d}" for i in range(50)]
    assert len(draw_training_pool({}, labels, size=20, rng=random.Random(0))) == 20


def test_the_training_registry_is_in_memory_and_writes_nothing(tmp_path):
    drawn = draw_training_pool(ranking_of(a=(5, 1600.0, 20)), [f"a{i:03d}" for i in range(5)],
                               size=5, rng=random.Random(0))

    registry = build_training_registry(tmp_path / "models", drawn)

    assert set(registry.members) == {m.label for m in drawn}
    assert registry.directory == tmp_path / "models"
    assert not (tmp_path / "models").exists()  # building it touched nothing on disk


# ---- parents -----------------------------------------------------------------------


def test_parents_are_distinct_rated_models_from_the_top():
    ranking = ranking_of(top=(20, 2000.0, 100), weak=(50, 1000.0, 100))

    parents = pick_parents(ranking, list(ranking), 8, rng=random.Random(0), tiers=(20,))

    assert len(parents) == 8 and len(set(parents)) == 8
    assert all(label.startswith("top") for label in parents)


def test_parents_exclude_unrated_and_frozen_models():
    ranking = ranking_of(good=(2, 1800.0, 50))
    ranking["anchor"] = PoolMember(label="anchor", kind="model", ref="a.pt", rating=2500.0, games=99, frozen=True)

    parents = pick_parents(ranking, [*ranking, "unrated"], 4, rng=random.Random(0))

    assert set(parents) <= {"good000", "good001"}


def test_parents_cycle_when_there_are_fewer_models_than_requested():
    ranking = ranking_of(only=(2, 1800.0, 50))
    parents = pick_parents(ranking, list(ranking), 5, rng=random.Random(0))
    assert len(parents) == 5 and set(parents) == set(ranking)


def test_no_rated_models_means_no_parents():
    assert pick_parents({}, ["x"], 3, rng=random.Random(0)) == []
    assert pick_parents(ranking_of(a=(2, 1600.0, 5)), ["a000"], 0, rng=random.Random(0)) == []


# ---- the store ----------------------------------------------------------------------


def test_available_labels_are_the_pt_files_in_the_store(tmp_path):
    store_dir = tmp_path / "models"
    store_dir.mkdir()
    for name in ("b.pt", "a.pt", ".c.pt", "d.pt.partial", "notes.txt"):
        (store_dir / name).write_bytes(b"x")

    assert available_labels(store_dir) == ["a", "b"]
    assert available_labels(tmp_path / "missing") == []


def test_publishing_puts_the_model_in_the_store_and_the_ranking(tmp_path):
    source = tmp_path / "scratch" / "agent-x.pt"
    source.parent.mkdir()
    source.write_bytes(b"weights")

    member = publish_model(
        source, models_dir=tmp_path / "models", global_dir=tmp_path / "global",
        name="host-a-gen0001-w00-agent-x.pt", rating=1655.0, iteration=50, machine="host-a",
    )

    assert (tmp_path / "models" / "host-a-gen0001-w00-agent-x.pt").read_bytes() == b"weights"
    assert member.label == "host-a-gen0001-w00-agent-x"
    stored = read_member(tmp_path / "global", member.label)
    assert (stored.rating, stored.games, stored.iteration, stored.frozen) == (1655.0, 0, 50, False)
    assert stored.ref == str(tmp_path / "models" / "host-a-gen0001-w00-agent-x.pt")
    assert not list((tmp_path / "models").glob(".*"))  # no partial file left behind


def test_publishing_is_write_once(tmp_path):
    source = tmp_path / "agent.pt"
    source.write_bytes(b"first")
    kwargs = {"models_dir": tmp_path / "models", "global_dir": tmp_path / "global",
              "name": "m.pt", "rating": 1600.0, "iteration": 1, "machine": "a"}
    assert publish_model(source, **kwargs) is not None

    source.write_bytes(b"second")
    assert publish_model(source, **kwargs) is None

    assert (tmp_path / "models" / "m.pt").read_bytes() == b"first"


def test_publishing_never_overwrites_a_rating_the_ranking_already_holds(tmp_path):
    write_member(tmp_path / "global", PoolMember(label="m", kind="model", ref="x", rating=1777.0, games=40))
    source = tmp_path / "agent.pt"
    source.write_bytes(b"w")

    publish_model(source, models_dir=tmp_path / "models", global_dir=tmp_path / "global",
                  name="m.pt", rating=1500.0, iteration=1, machine="a")

    assert read_member(tmp_path / "global", "m").rating == 1777.0


def test_a_sidecar_round_trips_and_defaults_when_missing(tmp_path):
    checkpoint = tmp_path / "agent.pt"
    assert read_sidecar(checkpoint) == (DEFAULT_RATING, 0)
    write_sidecar(checkpoint, rating=1712.5, iteration=25)
    assert read_sidecar(checkpoint) == (1712.5, 25)
    checkpoint.with_suffix(".json").write_text("{not json")
    assert read_sidecar(checkpoint) == (DEFAULT_RATING, 0)


# ---- reading the ranking cheaply ------------------------------------------------------


def test_load_ranking_prefers_the_single_file_snapshot(tmp_path):
    write_member(tmp_path, PoolMember(label="a", kind="model", ref="a.pt", rating=1600.0))
    write_snapshot(tmp_path, machine="m", force=True)
    write_member(tmp_path, PoolMember(label="a", kind="model", ref="a.pt", rating=1700.0))

    # Deliberately the snapshot's (older) value: it is a cheap read, not the truth.
    assert load_ranking(tmp_path).members["a"].rating == 1600.0


def test_load_ranking_falls_back_to_the_member_files_when_there_is_no_snapshot(tmp_path):
    write_member(tmp_path, PoolMember(label="a", kind="model", ref="a.pt", rating=1650.0))
    assert load_ranking(tmp_path).members["a"].rating == 1650.0


def test_load_ranking_on_an_empty_directory_is_empty(tmp_path):
    assert load_ranking(tmp_path / "nothing").members == {}


def test_a_legacy_snapshot_only_registry_is_still_readable(tmp_path):
    registry = PoolRegistry(directory=tmp_path, max_models=10**9)
    registry.members["old"] = PoolMember(label="old", kind="model", ref="o.pt", rating=1550.0)
    registry.save()
    assert json.loads((tmp_path / "registry.json").read_text())["members"]
    assert load_ranking(tmp_path).members["old"].rating == pytest.approx(1550.0)


def test_parents_come_from_every_band_of_the_ranking():
    ranking = ranking_of(m=(3000, 1500.0, 10))
    ranked = sorted(ranking.values(), key=lambda m: (-m.rating, m.label))
    position = {m.label: i for i, m in enumerate(ranked)}
    rng = random.Random(0)
    ranks = [position[pick_parents(ranking, list(ranking), 1, rng=rng)[0]] for _ in range(2000)]
    assert 0.22 < sum(r < 10 for r in ranks) / 2000 < 0.36
    assert 0.10 < sum(r >= 1000 for r in ranks) / 2000 < 0.24


def test_the_parent_bands_round_trip_through_their_text_form():
    from pokerlab.rl.training_pool import PARENT_TIERS

    assert parse_parent_tiers(format_parent_tiers(PARENT_TIERS)) == PARENT_TIERS
    assert parse_parent_tiers("10, 10 ,ALL") == (10, 10, None)


@pytest.mark.parametrize("text", ["", " , ", "ten", "0", "-3", "10, x"])
def test_a_parent_band_that_is_not_a_size_or_all_is_refused(text):
    with pytest.raises(ValueError):
        parse_parent_tiers(text)
