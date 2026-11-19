"""`studies/critic_loss`: the bins and targets of the HL-Gauss critic, the split and the
explained variance by group."""

from __future__ import annotations

from itertools import pairwise

import pytest
from critic_loss import (
    SIGMA_RATIO,
    VALUE_RANGE,
    bin_edges,
    bin_values,
    explained_variance,
    grouped,
    hl_config,
    hl_gauss,
    run_name,
    size_group,
    split_of,
    symexp,
    symlog,
    training_jobs,
)

UNIT = 0.01  # one big blind in the value's scale


@pytest.mark.parametrize("value", [-1.0, -0.013, 0.0, 0.004, 0.37, 7.9])
def test_symlog_is_undone_by_symexp(value):
    assert symexp(symlog(value, UNIT), UNIT) == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("kind", ["hl-symlog", "hl-uniform"])
def test_the_bins_cover_the_range_and_their_values_are_in_order(kind):
    values = bin_values(kind, UNIT)
    assert values == sorted(values)
    assert VALUE_RANGE[0] <= values[0] < values[-1] <= VALUE_RANGE[1]
    assert len(bin_edges(kind, UNIT)) == len(values) + 1


def test_the_symlog_bins_are_fine_near_zero_and_coarse_in_the_tails():
    values = bin_values("hl-symlog", UNIT)
    widths = [b - a for a, b in pairwise(values)]
    zero = min(range(len(values)), key=lambda i: abs(values[i]))
    assert widths[zero] < 0.003 < widths[-1] / 100   # under a third of a bb at 0, over 30 bb at +8


@pytest.mark.parametrize("target", [-0.5, 0.0, 0.031, 2.0])
def test_the_hl_gauss_target_is_a_distribution_whose_mean_is_the_target(target):
    edges = bin_edges("hl-uniform", UNIT)
    probabilities = hl_gauss(target, edges, 0.75 * (edges[1] - edges[0]))
    assert sum(probabilities) == pytest.approx(1.0)
    assert min(probabilities) >= 0
    centres = bin_values("hl-uniform", UNIT)
    assert sum(p * c for p, c in zip(probabilities, centres, strict=True)) == pytest.approx(target, abs=1e-3)


def test_a_target_outside_the_range_goes_to_its_end():
    edges = bin_edges("hl-uniform", UNIT)
    probabilities = hl_gauss(50.0, edges, 0.75 * (edges[1] - edges[0]))
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[-1] > 0.5


def test_explained_variance_is_one_for_a_perfect_critic_and_zero_for_the_mean():
    targets = [1.0, -1.0, 0.5, 0.0]
    assert explained_variance(targets, targets) == 1.0
    assert explained_variance(targets, [0.125] * 4) == pytest.approx(0.0)
    by = grouped(targets, targets, ["a", "a", "b", "b"])
    assert by == {"a": (1.0, 2), "b": (1.0, 2)}


def test_the_split_is_fixed_per_trajectory_and_roughly_70_15_15():
    sides = [split_of(i) for i in range(20_000)]
    assert sides == [split_of(i) for i in range(20_000)]
    assert 0.68 < sides.count("train") / len(sides) < 0.72
    assert 0.13 < sides.count("test") / len(sides) < 0.17


def test_table_sizes_fall_in_the_groups_of_the_style_lines():
    assert [size_group(n) for n in (2, 3, 4, 6, 7, 9)] == ["2-3", "2-3", "4-6", "4-6", "7-9", "7-9"]


def test_the_batched_target_is_the_one_target_at_a_time():
    torch = pytest.importorskip("torch")
    from critic_loss import hl_gauss_batch

    edges = bin_edges("hl-symlog", UNIT)
    sigma = 0.75 * (edges[1] - edges[0])
    targets = [-6.0, -0.3, 0.0, 1.7, 9.0]
    batch = hl_gauss_batch(torch.tensor(targets, dtype=torch.float64), torch.tensor(edges, dtype=torch.float64), sigma)
    for row, target in zip(batch.tolist(), targets, strict=True):
        assert row == pytest.approx(hl_gauss(target, edges, sigma), abs=1e-9)


def test_the_critic_shape_is_the_models_unless_the_command_line_overrides_it():
    import argparse

    from critic_loss import critic_shape

    model = {"hidden": 512, "num_layers": 3, "head_hidden": 256, "head_layers": 1}
    none = argparse.Namespace(hidden=None, num_layers=None, head_hidden=None, head_layers=None)
    assert critic_shape(model, none) == model
    some = argparse.Namespace(hidden=1024, num_layers=None, head_hidden=None, head_layers=3)
    assert critic_shape(model, some) == {"hidden": 1024, "num_layers": 3, "head_hidden": 256, "head_layers": 3}


def test_a_smaller_training_fraction_keeps_a_subset_of_a_larger_one():
    from critic_loss import kept

    half = {h for h in range(20_000) if kept(h, 0.5)}
    quarter = {h for h in range(20_000) if kept(h, 0.25)}
    assert quarter <= half
    assert abs(len(half) / 20_000 - 0.5) < 0.02 and abs(len(quarter) / 20_000 - 0.25) < 0.02
    assert all(kept(h, 1.0) for h in range(100))


def test_the_bins_can_be_any_number_over_the_same_range():
    for count in (11, 101, 400):
        edges = bin_edges("hl-symlog", UNIT, count)
        assert len(edges) == count + 1
        assert edges[0] == pytest.approx(symlog(VALUE_RANGE[0], UNIT))
        assert edges[-1] == pytest.approx(symlog(VALUE_RANGE[1], UNIT))
    assert bin_edges("hl-symlog", UNIT) == bin_edges("hl-symlog", UNIT, 101)


def test_a_configuration_is_bins_and_the_gaussians_width_in_bins():
    import argparse

    assert hl_config("51:1.5") == (51, 1.5)
    assert hl_config("201") == (201, SIGMA_RATIO)
    for bad in ("1:0.75", "51:0", "51:-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            hl_config(bad)


def test_every_hl_loss_runs_once_per_configuration_and_the_squared_errors_once():
    jobs = training_jobs(["mse", "hl-symlog"], [(11, 0.75), (101, 3.0)], seeds=2)
    assert jobs == [("mse", None, None, 0), ("mse", None, None, 1),
                    ("hl-symlog", 11, 0.75, 0), ("hl-symlog", 11, 0.75, 1),
                    ("hl-symlog", 101, 3.0, 0), ("hl-symlog", 101, 3.0, 1)]
    assert training_jobs(["hl-uniform"], None, seeds=1) == [("hl-uniform", 401, SIGMA_RATIO, 0)]
    assert run_name({"loss": "hl-symlog", "bins": 101, "sigma_ratio": 0.375, "seed": 1}) == "hl-symlog 101b s0.375 seme 1"
    assert run_name({"loss": "mse", "bins": None, "sigma_ratio": None, "seed": 0}) == "mse seme 0"
