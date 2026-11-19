"""Which models the pruning by diversity removes (`rl/diversity.py`)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pokerlab.rl import diversity
from pokerlab.rl.diversity import distance_matrix, redundant_models


def _signature(probabilities):
    """A signature from per-probe action probabilities `(probes, bins)`."""
    return np.sqrt(np.asarray(probabilities, dtype=np.float32))


def _policy(rng, lean: float, probes: int = 50, bins: int = 11):
    logits = rng.normal(size=(probes, bins)) * 0.3
    logits[:, 0] += lean
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def test_the_distance_is_zero_for_one_policy_and_one_for_policies_that_never_agree():
    a = _signature([[1.0, 0.0], [0.0, 1.0]])
    b = _signature([[0.0, 1.0], [1.0, 0.0]])
    d = distance_matrix([a, a.copy(), b])
    assert d[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert d[0, 2] == pytest.approx(1.0)


def _family(rng):
    base, other = _policy(rng, 0.0), _policy(rng, 3.0)
    near = base * 0.98 + 0.02 / base.shape[1]
    labels = ["top", "copy_hi", "copy_lo", "different"]
    signatures = [_signature(base), _signature(near), _signature(near * 0.999 + 0.001 / base.shape[1]),
                  _signature(other)]
    ratings = {"top": 2000.0, "copy_hi": 1500.0, "copy_lo": 1450.0, "different": 1300.0}
    return labels, signatures, ratings


def test_the_near_copies_go_first_the_lower_rated_of_two_and_the_protected_never():
    labels, signatures, ratings = _family(np.random.default_rng(0))
    removable = {"copy_hi", "copy_lo", "different"}
    assert redundant_models(labels, signatures, removable=removable, count=1, ratings=ratings) == ["copy_lo"]
    order = redundant_models(labels, signatures, removable=removable, count=3, ratings=ratings)
    assert order[:2] == ["copy_lo", "copy_hi"] and order[2] == "different"
    assert redundant_models(labels, signatures, removable=set(), count=3, ratings=ratings) == []


def test_too_many_numbers_read_every_model_on_one_fixed_subset_of_the_probes():
    assert diversity.probe_subset(models=10, probes=100, bins=11, limit=10**6) is None
    subset = diversity.probe_subset(models=100, probes=1000, bins=11, limit=110_000)
    assert len(subset) == 100 and list(subset) == sorted(set(subset))
    again = diversity.probe_subset(models=100, probes=1000, bins=11, limit=110_000)
    assert list(subset) == list(again)  # the same subset every time


def test_a_subset_of_the_probes_keeps_the_choice():
    labels, signatures, ratings = _family(np.random.default_rng(1))
    removable = {"copy_hi", "copy_lo", "different"}
    exact = redundant_models(labels, signatures, removable=removable, count=2, ratings=ratings)
    subset = diversity.probe_subset(models=4, probes=50, bins=11, limit=4 * 11 * 25)
    cut = [signature[subset] for signature in signatures]
    assert redundant_models(labels, cut, removable=removable, count=2, ratings=ratings) == exact
