from __future__ import annotations

import os
import random

import pytest

from pokerlab.rl.forking import run_in_copies, torch_seeded


def square_after_seeding(job: int) -> tuple[int, float]:
    # What a real job does: everything it draws comes from its own number.
    return os.getpid(), random.Random(job).random()


def test_the_results_come_back_in_job_order_whatever_the_number_of_copies():
    serial = [value for _pid, value in run_in_copies(square_after_seeding, 12, 1)]
    forked = [value for _pid, value in run_in_copies(square_after_seeding, 12, 4)]
    assert forked == serial


def test_a_job_really_runs_in_another_process():
    pids = {pid for pid, _value in run_in_copies(square_after_seeding, 12, 3)}
    assert os.getpid() not in pids and len(pids) >= 1


def test_without_copies_everything_runs_here():
    assert {pid for pid, _ in run_in_copies(square_after_seeding, 5, 1)} == {os.getpid()}
    assert {pid for pid, _ in run_in_copies(square_after_seeding, 1, 8)} == {os.getpid()}


def test_every_result_is_reported_as_it_arrives():
    seen = []
    run_in_copies(lambda job: job * 2, 9, 3, on_result=lambda job, result: seen.append((job, result)))
    assert sorted(seen) == [(job, job * 2) for job in range(9)]


def test_a_failing_job_raises_with_its_traceback():
    def work(job):
        if job == 3:
            raise ValueError("boom in job three")
        return job

    with pytest.raises(RuntimeError, match="boom in job three"):
        run_in_copies(work, 8, 3)


def test_a_copy_that_dies_without_reporting_is_noticed():
    def work(job):
        if job == 2:
            os._exit(1)
        return job

    with pytest.raises(RuntimeError, match="exited without reporting|returned"):
        run_in_copies(work, 6, 2)


def test_zero_jobs_is_an_empty_list():
    assert run_in_copies(lambda job: job, 0, 4) == []


def test_torch_seeding_gives_the_same_draws_and_leaves_the_outer_stream_alone():
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    before = torch.get_rng_state()
    with torch_seeded(11):
        first = torch.rand(3)
    with torch_seeded(11):
        second = torch.rand(3)
    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), before)
