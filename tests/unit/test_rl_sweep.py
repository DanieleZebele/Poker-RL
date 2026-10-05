from __future__ import annotations

import json
import math
import random
from argparse import Namespace

import pytest

from pokerlab.rl.loop import (
    _HP_COMPLEMENT_AXES,
    HP_AXES,
    HP_MULTIPLIERS,
    Hyperparameters,
    build_sweep_policy,
    hyperparameter_plan,
    starting_hyperparameters,
)
from pokerlab.rl.sweep_log import (
    SWEEP_DIRNAME,
    SweepObservation,
    read_observations,
    write_observation,
)
from pokerlab.rl.sweep_optimizer import (
    MIN_FIT,
    SEED_STEP,
    build_policy,
    fit_estimate,
    log_step,
    report_lines,
)

AXES = ("lr", "hands", "ppo_epochs", "clip_epsilon", "gae_lambda")
COMPLEMENT = frozenset({"gae_lambda"})
BASE = {"lr": 3e-4, "hands": 512, "ppo_epochs": 4, "clip_epsilon": 0.2, "gae_lambda": 0.95}


def synthetic(
    count: int,
    *,
    beta: dict[str, float],
    cost_elasticity: dict[str, float] | None = None,
    noise: float = 13.0,
    seed: int = 0,
) -> list[SweepObservation]:
    """Children whose gain and CPU follow a known law of the step they took."""
    rng = random.Random(seed)
    elasticity = cost_elasticity or {}
    found = []
    for index in range(count):
        parent = {axis: BASE[axis] * math.exp(rng.uniform(-0.3, 0.3)) for axis in BASE}
        parent["gae_lambda"] = 0.95
        child = dict(parent)
        steps = {}
        for axis in AXES:
            multiplier = rng.choice([0.8, 1.0, 1.2])
            steps[axis] = math.log(multiplier)
            if axis == "gae_lambda":
                child[axis] = 1 - (1 - parent[axis]) * multiplier
            else:
                child[axis] = parent[axis] * multiplier
        parent_rating = rng.gauss(1600, 60)
        gain = 5 - 0.1 * (parent_rating - 1600) + sum(beta.get(a, 0) * steps[a] for a in AXES)
        gain += rng.gauss(0, noise)
        cpu = 1000.0 * math.exp(
            sum(elasticity.get(a, 0) * math.log(child[a] / BASE[a]) for a in AXES if a != "gae_lambda")
            + rng.gauss(0, 0.03)
        )
        found.append(
            SweepObservation(
                f"m{index}", "p", parent_rating, parent_rating + gain, cpu, child, parent, "x", float(index)
            )
        )
    return found


# ---- the log -----------------------------------------------------------------


def observation(label: str, **overrides) -> SweepObservation:
    fields = {
        "label": label, "parent": "p", "parent_rating": 1500.0, "rating": 1520.0,
        "cpu_seconds": 900.0, "settings": {"lr": 3e-4}, "parent_settings": {"lr": 2.5e-4},
        "machine": "m", "time": 1.0,
    }
    fields.update(overrides)
    return SweepObservation(**fields)


def test_an_observation_round_trips_through_the_shared_directory(tmp_path):
    written = write_observation(tmp_path, observation("a"))
    assert written == tmp_path / SWEEP_DIRNAME / "a.json"
    [read] = read_observations(tmp_path)
    assert read == observation("a")
    assert read.gain == 20.0


def test_an_observation_is_written_once_and_never_overwritten(tmp_path):
    write_observation(tmp_path, observation("a", rating=1520.0))
    assert write_observation(tmp_path, observation("a", rating=9999.0)) is None
    assert read_observations(tmp_path)[0].rating == 1520.0


def test_unreadable_files_are_skipped_not_raised(tmp_path):
    write_observation(tmp_path, observation("good"))
    directory = tmp_path / SWEEP_DIRNAME
    (directory / "torn.json").write_text('{"schema": 1, "label": "torn", "par', encoding="utf-8")
    (directory / "foreign.json").write_text(json.dumps({"schema": 99}), encoding="utf-8")
    (directory / "nan.json").write_text(
        '{"schema": 1, "label": "n", "parent": "p", "parent_rating": NaN, "rating": 1, '
        '"cpu_seconds": 1, "time": 1, "settings": {}, "parent_settings": {}}',
        encoding="utf-8",
    )
    (directory / ".half.json.partial").write_text("{", encoding="utf-8")
    assert [o.label for o in read_observations(tmp_path)] == ["good"]


def test_no_directory_means_no_observations(tmp_path):
    assert read_observations(tmp_path / "nowhere") == []


def test_only_the_most_recent_observations_are_read(tmp_path):
    import os

    for index in range(5):
        path = write_observation(tmp_path, observation(f"o{index}", time=float(index)))
        os.utime(path, (1000 + index, 1000 + index))
    assert [o.label for o in read_observations(tmp_path, limit=2)] == ["o3", "o4"]


# ---- the steps ---------------------------------------------------------------


def test_a_step_is_the_log_ratio_and_a_lambda_step_is_taken_on_its_complement():
    assert log_step("lr", 1e-3, 1.2e-3, COMPLEMENT) == pytest.approx(math.log(1.2))
    # 0.95 -> 0.94 is a x1.2 step in 1 - lambda (0.05 -> 0.06), not a 1% one.
    assert log_step("gae_lambda", 0.95, 0.94, COMPLEMENT) == pytest.approx(math.log(1.2))


def test_a_move_off_zero_has_a_finite_recorded_step():
    assert log_step("entropy_coef", 0.0, 1e-3, frozenset()) == SEED_STEP
    assert log_step("entropy_coef", 0.0, 0.0, frozenset()) == 0.0


# ---- the estimate ------------------------------------------------------------


def test_a_planted_effect_is_recovered_with_its_sign():
    found = synthetic(500, beta={"lr": 20.0, "clip_epsilon": -10.0})
    estimate = fit_estimate(found, AXES, COMPLEMENT)
    index = {axis: i for i, axis in enumerate(estimate.axes)}
    assert estimate.beta[index["lr"]] == pytest.approx(20.0, abs=8.0)
    assert estimate.beta[index["clip_epsilon"]] == pytest.approx(-10.0, abs=8.0)
    for quiet in ("hands", "ppo_epochs", "gae_lambda"):
        assert abs(estimate.beta[index[quiet]]) < 8.0


def test_pure_noise_leaves_every_effect_inside_its_own_uncertainty():
    estimate = fit_estimate(synthetic(300, beta={}), AXES, COMPLEMENT)
    for i in range(len(estimate.axes)):
        assert abs(estimate.beta[i]) < 3 * estimate.beta_sd(i)


def test_the_cost_elasticity_is_recovered():
    found = synthetic(400, beta={}, cost_elasticity={"hands": 0.6, "ppo_epochs": 0.3})
    estimate = fit_estimate(found, AXES, COMPLEMENT)
    index = {axis: i for i, axis in enumerate(estimate.axes)}
    assert estimate.elasticity[index["hands"]] == pytest.approx(0.6, abs=0.12)
    assert estimate.elasticity[index["ppo_epochs"]] == pytest.approx(0.3, abs=0.12)
    assert abs(estimate.elasticity[index["lr"]]) < 0.1


def test_too_few_children_fit_nothing():
    assert fit_estimate(synthetic(MIN_FIT - 1, beta={}), AXES, COMPLEMENT) is None


def test_a_child_with_no_cpu_time_or_a_missing_axis_is_not_used():
    good = synthetic(MIN_FIT + 5, beta={})
    bad = [
        SweepObservation("z0", "p", 1500, 1500, 0.0, dict(BASE), dict(BASE)),
        SweepObservation("z1", "p", 1500, 1500, 5.0, {"lr": 1.0}, {"lr": 1.0}),
    ]
    assert fit_estimate(good + bad, AXES, COMPLEMENT).observations == len(good)


def test_the_gain_a_unit_of_cost_is_worth_never_goes_below_the_floor():
    """With children that no longer improve on average, `G <= 0` would make cost
    look like a benefit; the floor keeps the objective the right way up."""
    losing = synthetic(200, beta={})
    shifted = [
        SweepObservation(o.label, o.parent, o.parent_rating, o.parent_rating - 20, o.cpu_seconds,
                         o.settings, o.parent_settings, o.machine, o.time)
        for o in losing
    ]
    estimate = fit_estimate(shifted, AXES, COMPLEMENT, min_gain=1.5)
    assert estimate.mean_gain < 0
    assert estimate.gain_scale == 1.5


# ---- the draw ----------------------------------------------------------------


def frequencies(policy, axis: str, runs: int = 600) -> dict[float, float]:
    counts = {m: 0 for m in policy.multipliers}
    for seed in range(runs):
        counts[policy.draw(random.Random(seed))[axis]] += 1
    return {m: c / runs for m, c in counts.items()}


def test_until_warmed_up_the_draw_is_uniform_and_says_so():
    policy = build_policy(synthetic(50, beta={"lr": 40.0}), AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=150)
    assert not policy.guided
    assert all(0.25 < f < 0.42 for f in frequencies(policy, "lr").values())
    assert "riscaldamento" in report_lines(policy)[0] and "50/150" in report_lines(policy)[0]


def test_a_warming_policy_changes_nothing_about_the_plan():
    """The same seed gives the same plan with and without a policy that has not
    warmed up: switching the optimizer on must not move the sweep before it knows
    anything."""
    start = Hyperparameters(
        lr=3e-4, hands=512, opponent_probability=0.5, ppo_epochs=4, clip_epsilon=0.2,
        pool_top_share=0.5, pool_top_n=100,
    )
    cold = build_policy([], HP_AXES, _HP_COMPLEMENT_AXES, HP_MULTIPLIERS, warmup=150)
    plain = hyperparameter_plan(6, [None] * 6, start, rng=random.Random(5))
    steered = hyperparameter_plan(6, [None] * 6, start, rng=random.Random(5), policy=cold)
    assert plain == steered


def test_a_warmed_policy_follows_a_planted_gain_but_keeps_exploring():
    policy = build_policy(
        synthetic(600, beta={"lr": 30.0, "clip_epsilon": -30.0}), AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=150
    )
    assert policy.guided
    lr = frequencies(policy, "lr")
    clip = frequencies(policy, "clip_epsilon")
    assert lr[1.2] > 0.6 and lr[0.8] < 0.25
    assert clip[0.8] > 0.6 and clip[1.2] < 0.25
    # Exploration keeps every multiplier reachable on every axis.
    assert min(lr.values()) > 0.03 and min(clip.values()) > 0.03


def test_the_cost_is_charged_against_an_axis_that_gains_nothing():
    """Same gain for every axis, but `hands` is expensive: the objective is Elo per
    compute, so the draw should lean away from raising it."""
    found = synthetic(600, beta={"hands": 2.0, "lr": 2.0}, cost_elasticity={"hands": 1.0})
    policy = build_policy(found, AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=150, min_gain=5.0)
    hands = frequencies(policy, "hands")
    lr = frequencies(policy, "lr")
    assert hands[1.2] < hands[0.8]
    assert lr[1.2] > lr[0.8]


def test_the_draw_is_reproducible_from_its_rng():
    policy = build_policy(synthetic(300, beta={"lr": 20.0}), AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=100)
    assert policy.draw(random.Random(9)) == policy.draw(random.Random(9))


def test_the_report_names_every_axis_with_its_effect_and_cost():
    policy = build_policy(synthetic(300, beta={"lr": 20.0}, cost_elasticity={"hands": 0.6}),
                          AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=100)
    text = "\n".join(report_lines(policy))
    assert "ottimizzatore attivo su 300 figli" in text
    for axis in AXES:
        assert axis in text
    assert "costo" in text and "netto" in text


# ---- in the loop -------------------------------------------------------------


def sweep_args(tmp_path, **overrides) -> Namespace:
    values = {
        "sweep_optimizer": True, "sweep_explore": 0.35, "sweep_warmup": 150,
        "sweep_window": 1500, "sweep_min_gain": 1.0, "hp_multipliers": list(HP_MULTIPLIERS),
    }
    values.update(overrides)
    return Namespace(**values)


def test_switched_off_the_optimizer_does_not_steer_but_still_reports(tmp_path, capsys):
    found = synthetic(300, beta={"lr": 20.0})
    for item in found:
        write_observation(tmp_path, item)
    # Observations here use the test's own five axes, so the loop's twelve are not
    # all present: the fit is skipped, and the report says it is only watching.
    off = build_sweep_policy(sweep_args(tmp_path, sweep_optimizer=False, sweep_warmup=10), tmp_path)
    assert off is None
    assert "spento: solo osservazione" in capsys.readouterr().out


def test_the_report_says_when_it_is_only_watching():
    policy = build_policy(synthetic(300, beta={"lr": 20.0}), AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=100)
    assert "spento" not in report_lines(policy)[0]
    assert "spento: solo osservazione" in report_lines(policy, steering=False)[0]
    cold = build_policy([], AXES, COMPLEMENT, HP_MULTIPLIERS, warmup=100)
    assert "spento: solo osservazione" in report_lines(cold, steering=False)[0]


def test_an_empty_store_gives_a_warming_policy(tmp_path, capsys):
    policy = build_sweep_policy(sweep_args(tmp_path), tmp_path)
    assert policy is not None and not policy.guided
    assert "riscaldamento" in capsys.readouterr().out


def test_a_failure_reading_the_evidence_never_stops_the_generation(tmp_path, monkeypatch, capsys):
    def explode(*_args, **_kwargs):
        raise RuntimeError("volume gone")

    monkeypatch.setattr("pokerlab.rl.loop.read_observations", explode)
    assert build_sweep_policy(sweep_args(tmp_path), tmp_path) is None
    assert "saltato per errore" in capsys.readouterr().out


def test_the_plan_with_a_policy_still_has_every_axis_and_every_worker():
    found = [
        SweepObservation(
            f"m{i}", "p", 1500.0 + i % 7, 1510.0 + i % 11, 600.0 + i,
            {a: 1.0 for a in HP_AXES}, {a: 1.0 for a in HP_AXES},
        )
        for i in range(200)
    ]
    policy = build_policy(found, HP_AXES, _HP_COMPLEMENT_AXES, HP_MULTIPLIERS, warmup=10)
    start = starting_hyperparameters(
        Namespace(
            lr=3e-4, hands=512, opponent_probability=0.5, ppo_epochs=4, clip_epsilon=0.2,
            pool_top_share=0.5, pool_top_n=100, minibatch_size=1024, gae_lambda=0.95,
            value_coef=0.5, max_grad_norm=0.5, entropy_coef=0.0,
        )
    )
    plan = hyperparameter_plan(5, [None] * 5, start, rng=random.Random(1), policy=policy)
    assert len(plan) == 5
    assert all(hp.hands >= 1 and hp.ppo_epochs >= 1 for hp in plan)


# ---- in the worker -----------------------------------------------------------


def train_args(tmp_path, *extra: str) -> Namespace:
    pytest.importorskip("torch")
    from pokerlab.rl.train import build_parser

    return build_parser().parse_args(
        ["--config", "", "--global-dir", str(tmp_path), "--machine", "m", *extra]
    )


PARENT_SETTINGS = {"lr": 2.5e-4, "hands": 512, "ppo_epochs": 4}


def test_a_resumed_worker_leaves_one_observation_with_both_halves_of_its_step(tmp_path, capsys):
    from pokerlab.rl.train import record_sweep_observation

    args = train_args(tmp_path, "--resume", "--parent-label", "m-gen0001-w00-agent-x", "--lr", "0.0003")
    record_sweep_observation(
        args, label="m-gen0002-w03-agent-y", parent_rating=1600.0, parent_settings=PARENT_SETTINGS,
        rating=1624.0, cpu_seconds=1800.0,
    )
    [seen] = read_observations(tmp_path)
    assert (seen.parent, seen.gain, seen.cpu_seconds) == ("m-gen0001-w00-agent-x", 24.0, 1800.0)
    assert seen.settings["lr"] == 0.0003 and seen.parent_settings["lr"] == 2.5e-4
    assert "hp_arm" not in seen.settings  # only numbers are steps
    assert "+24 Elo" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra, parent_settings",
    [
        (["--parent-label", "p"], PARENT_SETTINGS),  # not resumed
        (["--resume"], PARENT_SETTINGS),  # a parent nobody named
        (["--resume", "--parent-label", "p"], {}),  # a parent with no settings on record
    ],
)
def test_a_worker_with_no_step_to_report_leaves_nothing(tmp_path, extra, parent_settings):
    from pokerlab.rl.train import record_sweep_observation

    record_sweep_observation(
        train_args(tmp_path, *extra), label="c", parent_rating=1500.0,
        parent_settings=parent_settings, rating=1510.0, cpu_seconds=10.0,
    )
    assert read_observations(tmp_path) == []


def test_the_parent_is_recorded_in_the_published_model(tmp_path):
    from pokerlab.rl.train import run_metadata

    args = train_args(tmp_path, "--resume", "--parent-label", "m-gen0001-w00-agent-x")
    assert run_metadata(args)["parent_label"] == "m-gen0001-w00-agent-x"
