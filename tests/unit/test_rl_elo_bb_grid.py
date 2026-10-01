"""The bb/100-per-Elo-gap study: everything but the playing needs no torch."""

from __future__ import annotations

import pytest

from pokerlab.rl.elo_bb_grid import (
    BIG_BLIND,
    Cell,
    GridModel,
    cell_from_scores,
    fill_grid,
    format_grid,
    gap_points,
    load_models,
    load_study,
    pair_seed,
    pairs,
    resumable,
    save_study,
    slope_through_origin,
    thin,
)
from pokerlab.rl.global_store import write_member
from pokerlab.rl.pool_registry import PoolMember


def models(ratings):
    return [GridModel(f"m{k}", f"/x/m{k}.pt", r) for k, r in enumerate(ratings)]


def test_every_unordered_pair_is_played_once_including_each_model_against_itself():
    assert pairs(3) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    assert len(pairs(40)) == 40 * 41 // 2


def test_models_are_loaded_weakest_first_with_their_global_rating(tmp_path):
    bench, global_dir = tmp_path / "benchmark", tmp_path / "global"
    for label, rating in (("a", 1600.0), ("b", 1400.0), ("c", 1500.0)):
        (bench / "sub").mkdir(parents=True, exist_ok=True)
        (bench / "sub" / f"{label}.pt").write_bytes(b"w")
        write_member(global_dir, PoolMember(label=label, kind="model", ref="", rating=rating, frozen=True))
    assert [m.label for m in load_models(bench, global_dir)] == ["b", "c", "a"]


def test_thinning_keeps_the_ends_and_spreads_the_rest():
    picked = thin(models(range(0, 100, 10)), 4)
    assert [m.rating for m in picked] == [0, 30, 60, 90]
    assert thin(models([1, 2, 3]), None) == models([1, 2, 3])
    assert len(thin(models([1, 2, 3]), 10)) == 3


def test_a_pairs_seed_does_not_depend_on_the_run():
    assert pair_seed(0, 1, 2) == pair_seed(0, 1, 2)
    assert pair_seed(0, 1, 2) != pair_seed(0, 2, 2)
    assert pair_seed(0, 1, 2) != pair_seed(1, 1, 2)


def test_a_cell_is_the_rows_own_bb_per_100_per_seat():
    # Team A's seats beat team B's by 6 chips a hand in total; with no third party
    # A's own total is +3, so one of its 3 seats earns 1 chip = 0.5 bb a hand.
    hands = [6.0] * 100
    cell = cell_from_scores(0, 1, hands, [6.0] * 100)
    assert cell.bb100 == pytest.approx(1.0 / BIG_BLIND * 100)
    assert cell.hands == 200


def test_the_grid_is_antisymmetric_and_marks_unplayed_cells():
    cells = [Cell(0, 0, 0.5, 1, 10), Cell(0, 1, -20.0, 1, 10)]
    grid = fill_grid(3, cells)
    assert grid[0][1] == -20.0 and grid[1][0] == 20.0
    assert grid[0][0] == 0.5
    assert grid[2][2] is None
    assert "." in format_grid(grid)[-1]


def test_the_fit_recovers_a_known_slope_through_the_origin():
    points = [(gap, 0.4 * gap) for gap in (10, 50, 100, 200)]
    slope, r2 = slope_through_origin(points)
    assert slope == pytest.approx(0.4) and r2 == pytest.approx(1.0)
    assert slope_through_origin([]) is None


def test_gap_points_read_the_stronger_models_edge_as_positive():
    ms = models([1400.0, 1500.0])
    # Cell [0,1]: the weaker model earns -30 against the stronger one.
    assert gap_points(ms, [Cell(0, 1, -30.0, 1, 1), Cell(0, 0, 1.0, 1, 1)]) == [(100.0, 30.0)]


def test_a_saved_study_resumes_only_as_the_same_experiment(tmp_path):
    ms = models([1400.0, 1500.0])
    cells = [Cell(0, 1, -3.0, 1.0, 100)]
    path = tmp_path / "study.json"
    save_study(path, ms, cells, hands=100, seed=7)

    assert resumable(load_study(path), ms, hands=100, seed=7) == cells
    with pytest.raises(SystemExit):
        resumable(load_study(path), ms, hands=200, seed=7)
    with pytest.raises(SystemExit):
        resumable(load_study(path), models([1.0, 2.0, 3.0]), hands=100, seed=7)


def test_a_real_grid_plays_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    from pokerlab.rl.elo_bb_grid import run_grid
    from pokerlab.rl.policy import PokerActorCritic
    from pokerlab.rl.ppo import save_checkpoint

    ms = []
    for k in range(2):
        path = tmp_path / f"m{k}.pt"
        torch.manual_seed(k)
        save_checkpoint(path, PokerActorCritic(hidden=8, num_layers=1))
        ms.append(GridModel(f"m{k}", str(path), 1500.0 + k))

    cells = run_grid(ms, [], hands=5, seed=1, device="cpu", jobs=1)

    assert {(c.i, c.j) for c in cells} == {(0, 0), (0, 1), (1, 1)}
    assert [c.hands for c in sorted(cells, key=lambda c: (c.i, c.j))] == [5, 10, 5]
    again = run_grid(ms, cells, hands=5, seed=1, device="cpu", jobs=1)
    assert len(again) == 3, "nothing is replayed on resume"


def test_the_csv_is_rewritten_during_the_run_not_only_at_the_end(tmp_path, monkeypatch):
    import pokerlab.rl.elo_bb_grid as module

    ms = models([1400.0, 1500.0])
    monkeypatch.setattr(module, "load_models", lambda *a, **k: ms)

    def fake_run_grid(models_, done, *, on_cell, **kwargs):
        cells = []
        for i, j in pairs(len(models_)):
            cell = Cell(i, j, 1.0, 1.0, 10)
            cells.append(cell)
            on_cell(cell, len(cells), 3)
            if len(cells) == 2:
                # Mid-run: the file already exists and holds the unplayed cell as empty.
                text = (tmp_path / "g.csv").read_text(encoding="utf-8")
                assert text.splitlines()[-1].endswith(",")
        return cells

    monkeypatch.setattr(module, "run_grid", fake_run_grid)
    code = module.main(["--out", str(tmp_path / "s.json"), "--csv", str(tmp_path / "g.csv"),
                        "--csv-every", "0"])
    assert code == 0
