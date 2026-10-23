"""The pure parts of `studies/agents/search_distill.py`: the target the search's values make and the
loss that teaches it. The phases that play hands are run by hand, on the machine that has the cores."""

from __future__ import annotations

import argparse
import random
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from search_distill import (
    MASK_FILL,
    agreement,
    block_hands,
    distill_loss,
    policy_kl,
    soft_targets,
    table_plan,
)
from search_study import deal_of, players_text, table_choice

BINS = 11


def tensor(rows):
    return torch.tensor(rows, dtype=torch.float32)


def candidates_of(*indices):
    mask = torch.zeros(1, BINS, dtype=torch.bool)
    mask[0, list(indices)] = True
    return mask


def test_the_target_lives_on_the_candidates_and_favours_the_better_value():
    values = torch.zeros(1, BINS)
    values[0, 1], values[0, 2], values[0, 4] = 1.0, 3.0, -2.0
    target = soft_targets(values, candidates_of(1, 2, 4), tau=1.0)

    assert target.sum().item() == pytest.approx(1.0)
    assert target[0, 2] > target[0, 1] > target[0, 4]
    outside = [i for i in range(BINS) if i not in (1, 2, 4)]
    assert target[0, outside].sum().item() == 0.0


def test_a_small_temperature_is_the_best_candidate_and_a_large_one_the_average():
    values = torch.zeros(1, BINS)
    values[0, 1], values[0, 2] = 0.0, 1.0
    sharp = soft_targets(values, candidates_of(1, 2), tau=0.01)
    flat = soft_targets(values, candidates_of(1, 2), tau=1000.0)
    assert sharp[0, 2].item() == pytest.approx(1.0, abs=1e-6)
    assert flat[0, 2].item() == pytest.approx(0.5, abs=1e-2)


def test_the_loss_is_lower_when_the_policy_prefers_what_the_search_prefers():
    mask = torch.ones(1, BINS, dtype=torch.bool)
    candidates = candidates_of(1, 2)
    values = torch.zeros(1, BINS)
    values[0, 2] = 5.0  # the search likes bin 2
    teacher = torch.zeros(1, BINS)

    towards = torch.zeros(1, BINS)
    towards[0, 2] = 3.0
    away = torch.zeros(1, BINS)
    away[0, 1] = 3.0
    loss_towards, ce_towards, _ = distill_loss(towards, teacher, mask, candidates, values, tau=1.0, beta=0.0)
    loss_away, ce_away, _ = distill_loss(away, teacher, mask, candidates, values, tau=1.0, beta=0.0)
    assert ce_towards < ce_away and loss_towards < loss_away


def test_a_policy_left_as_it_was_costs_no_kl():
    mask = torch.ones(1, BINS, dtype=torch.bool)
    logits = torch.randn(1, BINS)
    _, _, kl = distill_loss(logits, logits.clone(), mask, candidates_of(1, 2), torch.ones(1, BINS), 1.0, 1.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)

    moved = logits.clone()
    moved[0, 3] += 4.0
    _, _, kl_moved = distill_loss(moved, logits, mask, candidates_of(1, 2), torch.ones(1, BINS), 1.0, 1.0)
    assert kl_moved.item() > 0.1, "moving probability to a bin the search did not look at is paid for"


def test_the_kl_ignores_the_actions_that_are_not_legal():
    legal = torch.zeros(1, BINS, dtype=torch.bool)
    legal[0, [1, 2]] = True
    teacher = torch.full((1, BINS), MASK_FILL)
    teacher[0, 1], teacher[0, 2] = 0.0, 0.0
    logits = teacher.clone()
    _, _, kl = distill_loss(logits, teacher, legal, legal, torch.ones(1, BINS), 1.0, 1.0)
    assert torch.isfinite(kl) and kl.item() == pytest.approx(0.0, abs=1e-6)


def test_agreement_counts_the_decisions_where_the_policys_pick_is_the_searchs():
    candidates = candidates_of(1, 2).repeat(2, 1)
    values = tensor([[0, 0, 5] + [0] * 8, [0, 5, 0] + [0] * 8])
    logits = tensor([[0, 0, 9] + [0] * 8, [0, 0, 9] + [0] * 8])  # right on the first, wrong on the second
    assert agreement(logits, candidates, values) == pytest.approx(0.5)


def test_a_fixed_size_deals_what_it_dealt_before_the_mix_existed():
    """The rule before the mix: the stacks from `Random(seed)`, the hero by rotation. Runs saved
    before must stay reproducible."""
    for hand in range(10):
        seed = 5 * 1_000_003 + hand
        rng = random.Random(seed)
        stacks = [max(100, round(rng.uniform(20.0, 100.0) * 100)) for _ in range(4)]
        assert deal_of(5, hand, 4, 20.0, 100.0) == (seed, stacks, hand % 4, 4)


def test_a_mix_draws_a_size_every_hand_with_the_weights():
    only_four = (0, 0, 1, 0, 0, 0, 0, 0)
    for hand in range(50):
        _seed, stacks, hero, size = deal_of(3, hand, 9, 20.0, 100.0, only_four)
        assert size == 4 and len(stacks) == 4 and 0 <= hero < 4

    weights = (11, 14, 14, 14, 14, 14, 14, 5)
    sizes = Counter(deal_of(3, hand, 9, 20.0, 100.0, weights)[3] for hand in range(20000))
    for size, weight in zip(range(2, 10), weights, strict=True):
        assert sizes[size] / 20000 == pytest.approx(weight / 100, abs=0.015)
    heroes = Counter(deal_of(3, hand, 9, 20.0, 100.0, (0, 0, 0, 0, 1, 0, 0, 0))[2] for hand in range(6000))
    assert set(heroes) == set(range(6)), "the hero sits in every seat of a six-handed table"


def test_the_table_choice_is_a_size_or_the_mix_of_the_config(tmp_path):
    def choice(players, weights=None, config=tmp_path / "config.toml"):
        return table_choice(argparse.Namespace(players=players, table_weights=weights, config=config))

    assert choice("6") == (6, None)
    with pytest.raises(SystemExit):
        choice("10")
    assert choice("mix", [0, 1, 1, 0, 0, 0, 0, 0]) == (4, (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    (tmp_path / "config.toml").write_text("[game]\ntable_weights = [1, 0, 0, 0, 0, 0, 0, 3]\n")
    players, weights = choice("mix")
    assert players == 9 and weights == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0)
    assert players_text(players, weights) == "a mix of tables (2:25% 9:75%)"
    assert players_text(3, None) == "3 players"


def test_the_improve_target_leaves_the_policy_alone_where_the_search_sees_no_difference():
    """Equal values: the target is the policy's own probabilities over the candidates, not an even
    split that would push it towards the action it plays least."""
    teacher = torch.full((1, BINS), -5.0)
    teacher[0, 1], teacher[0, 2] = 3.0, 0.0
    candidates = candidates_of(1, 2)
    equal = torch.zeros(1, BINS)
    improve = soft_targets(equal, candidates, 1.0, "improve", teacher)
    soft = soft_targets(equal, candidates, 1.0, "soft")
    policy = teacher.masked_fill(~candidates, MASK_FILL).softmax(dim=-1)
    assert improve == pytest.approx(policy, abs=1e-6)
    assert soft[0, 1].item() == pytest.approx(0.5)

    better = torch.zeros(1, BINS)
    better[0, 2] = 2.0
    moved = soft_targets(better, candidates, 1.0, "improve", teacher)
    assert moved[0, 2] > improve[0, 2], "a better value moves mass towards it"


def test_the_best_target_is_the_search_s_choice_alone():
    values = torch.zeros(1, BINS)
    values[0, 1], values[0, 2], values[0, 4] = 1.0, 3.0, 2.9
    best = soft_targets(values, candidates_of(1, 2, 4), 1.0, "best")
    assert best[0, 2].item() == 1.0 and best.sum().item() == 1.0


def test_the_improve_target_needs_the_original_policy():
    with pytest.raises(ValueError, match="original policy"):
        soft_targets(torch.zeros(1, BINS), candidates_of(1, 2), 1.0, "improve")


def test_the_policy_kl_is_zero_for_the_same_policy():
    logits = torch.randn(3, BINS)
    mask = torch.ones(3, BINS, dtype=torch.bool)
    assert policy_kl(logits, logits.clone(), mask).abs().max().item() < 1e-6


def table_args(**overrides):
    values = {"players": 9, "stack_min_bb": 20.0, "stack_max_bb": 100.0,
              "table_weights": (11, 14, 14, 14, 14, 14, 14, 5), "table_hands": 200}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_table_is_drawn_by_its_number_with_the_weights():
    meta = {"opponent": "o.pt"}
    assert table_plan(table_args(), meta, 3, 1) == table_plan(table_args(), meta, 3, 1)
    sizes = Counter(table_plan(table_args(), meta, job, 1)[0] for job in range(20000))
    for size, weight in zip(range(2, 10), table_args().table_weights, strict=True):
        assert sizes[size] / 20000 == pytest.approx(weight / 100, abs=0.015)
    size, hero, paths = table_plan(table_args(table_weights=None, players=4), meta, 0, 1)
    assert size == 4 and sorted([hero, *paths]) == [0, 1, 2, 3] and set(paths.values()) == {"o.pt"}


def test_a_table_of_the_pool_seats_distinct_models_around_the_hero():
    meta = {"opponent": "o.pt", "pool": [f"m{i}.pt" for i in range(50)]}
    for job in range(300):
        size, hero, paths = table_plan(table_args(), meta, job, 2)
        assert sorted(paths) == [seat for seat in range(size) if seat != hero]
        assert len(set(paths.values())) == size - 1 and set(paths.values()) <= set(meta["pool"])
    _size, _hero, paths = table_plan(table_args(table_weights=None), {"opponent": "o", "pool": ["a", "b"]}, 0, 1)
    assert set(paths.values()) <= {"a", "b"} and len(paths) == 8, "a pool smaller than the table repeats"


def test_the_blocks_cover_the_hands_asked_for_and_no_more():
    args = table_args()
    assert [block_hands(args, job, 450) for job in range(4)] == [200, 200, 50, 0]


def test_a_gap_the_search_is_not_sure_of_is_closed_and_a_sure_one_kept():
    from search_distill import significant_values

    teacher = torch.full((1, BINS), -5.0)
    teacher[0, 1] = 3.0  # the policy's likeliest candidate is bin 1
    candidates = candidates_of(1, 2, 4)
    values = torch.zeros(1, BINS)
    values[0, 1], values[0, 2], values[0, 4] = 1.0, 1.5, 4.0
    errors = torch.zeros(1, BINS)
    errors[0, 2], errors[0, 4] = 1.0, 1.0  # bin 2 is 0.5 se away, bin 4 is 3 se away
    kept = significant_values(values, errors, candidates, teacher, 2.0)
    assert kept[0, 2].item() == 1.0, "half a standard error: as good as the policy's own"
    assert kept[0, 4].item() == 4.0 and kept[0, 1].item() == 1.0
    assert significant_values(values, errors, candidates, teacher, 0.0) is values
    unknown = significant_values(values, torch.zeros(1, BINS), candidates, teacher, 2.0)
    assert torch.equal(unknown, values), "a gap with no recorded error is left as it is"
