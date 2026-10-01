"""The immediate kill: what it selects, what it spares, and in what order.

Two failure modes are worth real tests. Killing too little -- a worker that
outruns the scan and survives, so the next `start` finds the cores taken -- and
killing too much: another loop's processes, or the caller's own shell. The
selection tests run against a hand-built process table and signal nothing; the
two end-to-end tests spawn real `sleep` processes and kill them for real.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pokerlab.rl.killswitch import (
    STATE_FILENAME,
    KillReport,
    Proc,
    _root_variants,
    belongs_to,
    kill_everything,
    main,
    mark_killed,
    role_of,
    snapshot,
)


def proc(pid, argv, *, ppid=1, cwd="/work", state="S"):
    return Proc(pid=pid, ppid=ppid, argv=tuple(argv), cwd=Path(cwd), state=state)


# ---- who is one of ours ----------------------------------------------------


def test_the_supervisor_and_its_workers_are_recognised():
    assert role_of(proc(1, ["/venv/bin/poker-loop", "--workers", "30"])) == "supervisor"
    assert role_of(proc(2, ["/venv/bin/python", "-m", "pokerlab.rl.loop"])) == "supervisor"
    assert role_of(proc(3, ["/venv/bin/python", "-u", "-m", "pokerlab.rl.train"])) == "worker"
    assert role_of(proc(4, ["/venv/bin/poker-train", "--iterations", "100"])) == "worker"
    assert role_of(proc(5, ["/venv/bin/python", "-m", "pokerlab.rl.global_arena"])) == "shard"


def test_a_process_that_merely_mentions_the_loop_is_not_one():
    """The whole point of matching the entry point rather than the command line:
    a `tail -f` on a worker's log, or an editor with the file open, names
    poker-train in its arguments and must not be killed for it."""
    assert role_of(proc(1, ["tail", "-f", "checkpoints/logs/loop/gen0001-w00.log"])) is None
    assert role_of(proc(2, ["grep", "-r", "pokerlab.rl.train", "src"])) is None
    assert role_of(proc(3, ["vim", "src/pokerlab/rl/loop.py"])) is None
    assert role_of(proc(4, ["python", "-m", "pytest", "tests/unit/test_rl_loop.py"])) is None


def test_only_processes_naming_this_machines_directories_are_selected():
    """The store is shared over NFS but each machine runs its own loop. A worker
    of another machine's loop, seen here only because the path is visible, has to
    be left alone -- and in practice it is on another host entirely."""
    roots = _root_variants(Path("/vol/machines/host-a"), Path("/vol/machines/host-a/work"))

    mine = proc(1, ["poker-train", "--scratch-dir", "/vol/machines/host-a/work/gen0001-w00"])
    theirs = proc(2, ["poker-train", "--scratch-dir", "/vol/machines/host-b/work/gen0001-w00"])

    assert belongs_to(mine, roots)
    assert not belongs_to(theirs, roots)


def test_a_relative_directory_is_resolved_against_the_process_own_cwd():
    """`run.sh` passes `checkpoints/machines/<host>` relative to the project
    directory, so the string alone says nothing until it is joined to the cwd."""
    roots = _root_variants(Path("/project/checkpoints/machines/host-a"))

    inside = proc(1, ["poker-loop", "--state-dir", "checkpoints/machines/host-a"], cwd="/project")
    elsewhere = proc(2, ["poker-loop", "--state-dir", "checkpoints/machines/host-a"], cwd="/other")

    assert belongs_to(inside, roots)
    assert not belongs_to(elsewhere, roots)


def test_an_equals_spelled_flag_is_understood():
    roots = _root_variants(Path("/vol/host-a"))
    assert belongs_to(proc(1, ["poker-train", "--scratch-dir=/vol/host-a/work"]), roots)


# ---- the real thing --------------------------------------------------------


def spawn_fake(state_dir, name, *, seconds=60, module="pokerlab.rl.train"):
    """A process whose *command line* looks like a worker but which only sleeps.

    argv[0] is the interpreter and `-m <module>` is what `role_of` reads, so the
    selection logic treats it exactly as it treats a real worker -- while the
    code it runs is a sleep, which keeps the test fast and harmless.
    """
    script = f"import time; time.sleep({seconds})"
    return subprocess.Popen(
        [sys.executable, "-c", script, "-m", module,
         "--scratch-dir", str(Path(state_dir) / "work" / name)]
    )


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_it_kills_every_worker_of_this_machine_and_nothing_else(tmp_path):
    mine = [spawn_fake(tmp_path / "host-a", f"gen0001-w{i:02d}") for i in range(3)]
    theirs = spawn_fake(tmp_path / "host-b", "gen0001-w00")
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(0.5)
    try:
        report = kill_everything(tmp_path / "host-a", tmp_path / "host-a" / "work")

        assert report.count("worker") == 3
        assert sorted(pid for pid, *_ in report.stopped) == sorted(p.pid for p in mine)
        assert report.survivors == []
        for process in mine:
            assert process.wait(timeout=10) is not None
        assert theirs.poll() is None, "another machine's loop must be untouched"
        assert bystander.poll() is None, "an unrelated process must be untouched"
    finally:
        for process in (*mine, theirs, bystander):
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_children_of_a_worker_die_with_it(tmp_path):
    """A worker's global round shards it out to subprocesses; killing only the
    parent would orphan them and leave them running on the cores."""
    script = (
        "import subprocess, sys, time;"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "print(child.pid, flush=True); time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script, "-m", "pokerlab.rl.train",
         "--scratch-dir", str(tmp_path / "work" / "gen0001-w00")],
        stdout=subprocess.PIPE, text=True,
    )
    child_pid = int(parent.stdout.readline())
    try:
        report = kill_everything(tmp_path, tmp_path / "work")

        assert child_pid in [pid for pid, *_ in report.stopped]
        parent.wait(timeout=10)
        deadline = time.time() + 10
        while time.time() < deadline and Path(f"/proc/{child_pid}").exists():
            time.sleep(0.1)
        assert not Path(f"/proc/{child_pid}").exists(), "the child outlived its parent"
    finally:
        for pid in (parent.pid, child_pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.wait(timeout=10)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_a_dry_run_signals_nothing(tmp_path):
    process = spawn_fake(tmp_path, "gen0001-w00")
    time.sleep(0.5)
    try:
        report = kill_everything(tmp_path, tmp_path / "work", dry_run=True)

        assert [pid for pid, *_ in report.stopped] == [process.pid]
        assert process.poll() is None, "a dry run must leave the process running"
    finally:
        process.kill()
        process.wait(timeout=10)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_the_caller_never_kills_itself(tmp_path):
    """Every ancestor of this process is protected; without that, a kill run from
    a shell inside the work directory could take down the shell reading the report."""
    report = kill_everything(tmp_path, tmp_path / "work")
    assert os.getpid() not in [pid for pid, *_ in report.stopped]
    assert os.getppid() not in [pid for pid, *_ in report.stopped]


def test_nothing_running_is_not_an_error(tmp_path, capsys):
    assert main(["--state-dir", str(tmp_path), "--work-dir", str(tmp_path / "work")]) == 0
    assert "nessun processo" in capsys.readouterr().out


# ---- the state file --------------------------------------------------------


def test_the_state_records_that_the_loop_was_killed(tmp_path):
    """`--status` reads this file; left untouched it would go on reporting a
    generation that no longer has any processes behind it."""
    (tmp_path / STATE_FILENAME).write_text(
        json.dumps({"generation": 7, "phase": "training", "history": []}), encoding="utf-8"
    )

    assert mark_killed(tmp_path)

    state = json.loads((tmp_path / STATE_FILENAME).read_text())
    assert state["phase"] == "killed"
    assert state["generation"] == 7, "nothing else may be rewritten"


def test_marking_a_missing_state_file_is_harmless(tmp_path):
    assert not mark_killed(tmp_path)
    assert not (tmp_path / STATE_FILENAME).exists()


def test_the_state_filename_matches_the_loops_own():
    """The killswitch deliberately does not import loop.py (that would pull in
    torch), so the one constant they share is pinned by a test instead."""
    from pokerlab.rl.loop import STATE_FILENAME as loop_name

    assert STATE_FILENAME == loop_name


# ---- the process table -----------------------------------------------------


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_the_snapshot_reads_this_process():
    table = snapshot()
    assert os.getpid() in table
    assert table[os.getpid()].ppid == os.getppid()


def test_a_command_name_with_spaces_does_not_break_the_stat_parser(tmp_path):
    """The process name in /proc/<pid>/stat is in parentheses and may itself
    contain spaces and parentheses; splitting on whitespace reads the wrong field."""
    fake = tmp_path / "42"
    fake.mkdir()
    (fake / "cmdline").write_bytes(b"poker-train\0--scratch-dir\0/x\0")
    (fake / "stat").write_text("42 (weird name (x)) R 7 42 42 0 -1 0")

    table = snapshot(tmp_path)

    assert table[42].state == "R"
    assert table[42].ppid == 7


def test_the_report_counts_by_role():
    report = KillReport(stopped=[(1, "supervisor", ""), (2, "worker", "gen0001-w00")])
    assert report.count("worker") == 1
    assert report.count("shard") == 0
