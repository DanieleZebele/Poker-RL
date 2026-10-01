"""The monitoring page: what it reads, and that it only ever reads.

The HTML is not tested -- it is a static string with no logic worth pinning. What
matters is that the snapshot it serves stays JSON-serializable as `WorkerProgress`
grows fields, that an absent or half-written machine directory does not take the
page down, and that nothing here writes to the volume.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from pokerlab.rl.dashboard import State, read_fleet, read_history, read_machines
from pokerlab.rl.phases import ELO_PLAY, marker, progress_marker


def machine_dir(tmp_path, name, generation, workers=2, iterations=10):
    d = tmp_path / "machines" / name
    (d / "logs" / "loop").mkdir(parents=True, exist_ok=True)
    (d / "loop_state.json").write_text(
        json.dumps({"started": "oggi", "generation": generation, "phase": "training",
                    "workers": workers, "history": []}),
        encoding="utf-8",
    )
    for w in range(workers):
        lines = [f"iter {i:4}  reward   +1.00 bb  policy -0.01  value 1.0  "
                 f"entropy 0.500  kl 0.01  clip 0.1" for i in range(1, iterations + 1)]
        lines.append("        eval vs pool: +10.0 bb/100  rating 1523")
        (d / "logs" / "loop" / f"gen{generation:04d}-w{w:02d}.log").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    return d


def test_a_machine_reports_when_its_slowest_worker_should_be_done(tmp_path):
    """The end of a generation is the part that looks stuck: every bar reads
    100% and the workers then spend an hour in the final rounds. A generation
    ends when its slowest worker does, so the machine reports the longest ETA.
    """
    d = machine_dir(tmp_path, "host-a", 3, workers=2)
    logs = d / "logs" / "loop"
    for worker, eta in enumerate([600, 2400]):
        with (logs / f"gen0003-w{worker:02d}.log").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n" + marker(ELO_PLAY) + "\n"
                + progress_marker(ELO_PLAY, 1000, 2750, eta_seconds=eta) + "\n"
            )

    machine = read_machines(tmp_path / "machines", iterations=10)[0]

    assert machine["finish_eta"] == 2400
    # The page computes the percentage itself, so the payload carries the two
    # numbers rather than a third derived from them.
    progress = machine["workers"][0]["progress"]
    assert (progress["done"], progress["total"], progress["eta_seconds"]) == (1000, 2750, 600)
    assert machine["workers"][0]["stage_label"] == "elo: gioco"


def test_a_machine_that_is_still_training_has_no_finish_eta(tmp_path):
    machine_dir(tmp_path, "host-a", 3)
    machine = read_machines(tmp_path / "machines", iterations=10)[0]
    assert machine["finish_eta"] is None
    assert machine["workers"][0]["progress"] is None


def test_the_snapshot_is_json_serializable(tmp_path):
    """Served straight to the browser, so a field `WorkerProgress` gains later
    must not be something `json` refuses -- a dict of series scores, say."""
    machine_dir(tmp_path, "host-a", 3)
    state = State(SimpleNamespace(
        machines_dir=tmp_path / "machines", models_dir=tmp_path / "models",
        benchmark_dir=tmp_path / "benchmark", global_dir=tmp_path / "global",
        iterations=10,
    ))

    body = json.dumps(state.snapshot())

    reloaded = json.loads(body)
    assert reloaded["machines"][0]["machine"] == "host-a"
    assert len(reloaded["machines"][0]["workers"]) == 2
    assert reloaded["machines"][0]["workers"][0]["stage_label"]


def test_progress_is_the_share_of_the_generation_done(tmp_path):
    machine_dir(tmp_path, "host-a", 1, workers=2, iterations=5)
    machines = read_machines(tmp_path / "machines", iterations=10)
    assert machines[0]["progress"] == 0.5
    assert machines[0]["done"] == 0


def test_a_finished_generation_counts_its_workers_as_done(tmp_path):
    machine_dir(tmp_path, "host-a", 1, workers=3, iterations=10)
    machines = read_machines(tmp_path / "machines", iterations=10)
    assert machines[0]["done"] == 3
    assert machines[0]["progress"] == 1.0


def test_a_machine_with_no_logs_yet_is_listed_not_dropped(tmp_path):
    """A machine that has just started, or one whose logs were cleaned, still
    belongs on the page -- silently omitting it would read as "all fine"."""
    d = tmp_path / "machines" / "host-b"
    (d / "logs" / "loop").mkdir(parents=True)
    (d / "loop_state.json").write_text(json.dumps({"generation": 7, "phase": "idle"}))

    machines = read_machines(tmp_path / "machines", iterations=10)

    assert [m["machine"] for m in machines] == ["host-b"]
    assert machines[0]["workers"] == [] and machines[0]["progress"] == 0.0


def test_a_corrupt_state_file_does_not_take_the_page_down(tmp_path):
    d = tmp_path / "machines" / "host-c"
    (d / "logs" / "loop").mkdir(parents=True)
    (d / "loop_state.json").write_text("{ not json")

    machines = read_machines(tmp_path / "machines", iterations=10)

    assert machines[0]["generation"] == 0  # LoopState.load's fallback
    assert machines[0]["phase"] == "idle"


def test_missing_directories_are_empty_not_an_error(tmp_path):
    assert read_machines(tmp_path / "nope", iterations=10) == []
    fleet = read_fleet(tmp_path / "nope", tmp_path / "also-nope", tmp_path / "nor-this")
    assert fleet == {
        "models": 0, "benchmark_models": 0, "benchmark_series": [], "ratings": {},
    }


def test_the_fleet_numbers_are_cached_between_requests(tmp_path):
    """Counting the store walks ~9,000 files; the page polls every few seconds."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.pt").write_bytes(b"x")
    state = State(SimpleNamespace(
        machines_dir=tmp_path / "machines", models_dir=tmp_path / "models",
        benchmark_dir=tmp_path / "benchmark", global_dir=tmp_path / "global",
        iterations=10,
    ))

    assert state.snapshot()["fleet"]["models"] == 1
    (tmp_path / "models" / "b.pt").write_bytes(b"x")
    assert state.snapshot()["fleet"]["models"] == 1, "cached, not recounted"
    state._fleet_at = 0.0
    assert state.snapshot()["fleet"]["models"] == 2, "and refreshed once stale"


def test_the_dashboard_writes_nothing(tmp_path):
    """It is pointed at a live training volume and left open for hours."""
    machine_dir(tmp_path, "host-a", 1)
    (tmp_path / "models").mkdir()
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}

    State(SimpleNamespace(
        machines_dir=tmp_path / "machines", models_dir=tmp_path / "models",
        benchmark_dir=tmp_path / "benchmark", global_dir=tmp_path / "global",
        iterations=10,
    )).snapshot()

    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


# ---- the training curves ----------------------------------------------------


def test_history_returns_every_per_iteration_metric(tmp_path):
    machine_dir(tmp_path, "alpha", 3, workers=1, iterations=6)
    history = read_history(tmp_path / "machines", "alpha", "w00")
    assert history["total_iterations"] == 6
    assert history["iterations"] == [1, 2, 3, 4, 5, 6]
    for key in ("train_bb100", "policy", "value", "entropy", "kl", "clip"):
        assert len(history[key]) == 6, key
    # bb per 100 hands, converted from the bb-per-hand the log carries.
    assert history["train_bb100"] == [100.0] * 6
    # The sparse readings carry the iteration they were taken at, since they
    # happen every --eval-every iterations rather than every one.
    assert history["eval_bb100"] == [[6, 10.0]]
    assert history["eval_rating"] == [[6, 1523.0]]
    assert json.dumps(history)


def test_history_refuses_a_name_that_is_not_a_worker(tmp_path):
    machine_dir(tmp_path, "alpha", 3, workers=1)
    for worker in ("../../etc/passwd", "w00/../../x", "", "agent", "w"):
        assert read_history(tmp_path / "machines", "alpha", worker) is None


def test_history_refuses_a_machine_outside_the_machines_directory(tmp_path):
    machine_dir(tmp_path, "alpha", 3, workers=1)
    (tmp_path / "elsewhere" / "logs" / "loop").mkdir(parents=True)
    for machine in ("../elsewhere", "..", "/etc", "alpha/logs"):
        assert read_history(tmp_path / "machines", machine, "w00") is None


def test_history_of_a_generation_with_no_log_is_none_not_an_error(tmp_path):
    machine_dir(tmp_path, "alpha", 3, workers=1)
    assert read_history(tmp_path / "machines", "alpha", "w00", 999) is None


def test_history_is_thinned_to_the_point_budget(tmp_path):
    machine_dir(tmp_path, "alpha", 1, workers=1, iterations=500)
    history = read_history(tmp_path / "machines", "alpha", "w00", max_points=50)
    assert history["total_iterations"] == 500
    assert len(history["iterations"]) <= 51
    # The last iteration always survives: a chart that stopped short of the
    # present would be read as a stalled worker.
    assert history["iterations"][-1] == 500
    assert len(history["train_bb100"]) == len(history["iterations"])


def test_the_fleet_tiles_carry_the_store_wide_rating_scale(tmp_path):
    from pokerlab.rl.dashboard import store_ratings

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "registry.json").write_text(
        json.dumps({"members": [{"label": f"m{i}", "rating": 1400 + i} for i in range(200)]}),
        encoding="utf-8",
    )

    stats = store_ratings(global_dir)

    assert stats["rated"] == 200
    assert stats["mean"] == 1499.5
    assert stats["best"] == 1599
    # The top 1% is two models out of 200, and it is the number worth watching.
    assert stats["top1_count"] == 2
    assert stats["top1_mean"] == 1598.5


def test_an_unreadable_ranking_leaves_the_tiles_empty_not_broken(tmp_path):
    from pokerlab.rl.dashboard import store_ratings

    assert store_ratings(tmp_path / "missing") == {}
    (tmp_path / "registry.json").write_text("not json at all", encoding="utf-8")
    assert store_ratings(tmp_path) == {}


# ---- a client hanging up is not an error -------------------------------------


def test_a_client_disconnecting_is_not_reported(capsys):
    """Every open page polls every few seconds, so a tab closed mid-response is
    routine. Left alone, socketserver prints a 25-line traceback into the very
    terminal the dashboard runs in."""
    from pokerlab.rl.dashboard import QuietServer

    server = QuietServer.__new__(QuietServer)
    for error in (BrokenPipeError(32, "Broken pipe"), ConnectionResetError()):
        try:
            raise error
        except OSError:
            server.handle_error(None, ("127.0.0.1", 1234))
    assert capsys.readouterr().err == ""


def test_a_real_error_is_still_reported(capsys):
    """Only the connection-reset family is swallowed: a genuine bug in a handler
    has to stay visible."""
    from pokerlab.rl.dashboard import QuietServer

    server = QuietServer.__new__(QuietServer)
    try:
        raise ValueError("qualcosa di vero")
    except ValueError:
        server.handle_error(None, ("127.0.0.1", 1234))
    out = capsys.readouterr()
    assert "qualcosa di vero" in out.err + out.out


# ---- the hyperparameters each worker was given -----------------------------


def test_the_snapshot_carries_each_workers_settings(tmp_path):
    """Every worker of a generation now trains with its own, so "what is this
    machine running" no longer has one answer and the page has to be able to
    give the per-worker one."""
    from pokerlab.rl.phases import hyperparameters_marker

    d = machine_dir(tmp_path, "host-a", 4, workers=1)
    log = d / "logs" / "loop" / "gen0004-w00.log"
    log.write_text(
        log.read_text(encoding="utf-8")
        + "\n"
        + hyperparameters_marker(
            {"hp_arm": "inherited", "lr": 4.7e-4, "hands": 800, "ppo_epochs": 2}
        )
        + "\n",
        encoding="utf-8",
    )

    worker = read_machines(tmp_path / "machines", 10)[0]["workers"][0]

    assert worker["hyperparameters"]["hp_arm"] == "inherited"
    assert worker["hyperparameters"]["lr"] == "0.00047"
    assert worker["hyperparameters"]["hands"] == "800"
    # Still JSON, since this rides in the same payload the page polls.
    json.dumps(worker)


def test_a_worker_from_before_the_sweep_carries_an_empty_set(tmp_path):
    machine_dir(tmp_path, "host-a", 4, workers=1)

    worker = read_machines(tmp_path / "machines", 10)[0]["workers"][0]

    assert worker["hyperparameters"] == {}


def test_the_page_renders_the_settings_and_does_not_fetch_them_again(tmp_path):
    """They ride in the status payload the page already polls; a second request
    per open panel would be one more round trip for data already in hand."""
    from pokerlab.rl.dashboard import PAGE

    assert "hpByKey.set(key, w.hyperparameters || null)" in PAGE
    assert "function hpHtml(key)" in PAGE
    # Built in one place: three call sites used to repeat the panel's markup.
    assert PAGE.count("function panelHtml(key)") == 1
    assert PAGE.count("panel.innerHTML = panelHtml(key);") == 2


def test_the_page_counts_an_unassigned_arm_instead_of_hiding_it():
    """It is how a machine still running a supervisor from before the sweep
    identifies itself, so it has to be counted in the pill, not skipped."""
    from pokerlab.rl.dashboard import PAGE

    assert "'': 'senza sweep: supervisor da riavviare'" in PAGE
    assert "const arm = hp.hp_arm || '';" in PAGE


def test_the_snapshot_says_how_many_hands_the_training_rate_covers(tmp_path):
    """The column averages a fixed 100,000 hands, which for the first fifth of
    a run is not yet reachable; the page has to be able to tell a full window
    from one still filling."""
    d = machine_dir(tmp_path, "host-a", 4, workers=1, iterations=10)
    log = d / "logs" / "loop" / "gen0004-w00.log"
    log.write_text(
        "device cpu | 6 seats | 512 hands/iteration\n" + log.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    worker = read_machines(tmp_path / "machines", 10)[0]["workers"][0]

    assert worker["train_hands"] == 10 * 512
    json.dumps(worker)


def test_the_page_marks_a_window_that_is_still_filling():
    from pokerlab.rl.dashboard import PAGE

    assert "w.train_hands.toLocaleString('it') + ' mani'" in PAGE
    assert "w.train_hands < 100000" in PAGE
