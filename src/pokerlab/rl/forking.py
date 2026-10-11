"""Independent jobs, played by forked copies of this process.

Every phase that plays hands for a rating -- the validation pass, the pass against the
frozen anchors, the cross-machine Elo passes -- is a list of sessions that do not depend
on each other: who sits where and the seed of each is decided up front, and only the
*rating* that follows needs them in order. So the playing can be spread over copies of
the process, and the parent rates what comes back.

A copy is made with `fork`: it sees the parent's memory as it is at that moment (the
loaded models, the closures that play) and shares it copy-on-write, so it costs a fraction
of a second process and nothing has to be pickled on the way in. Only the result of each job
comes back, through a pipe, as soon as it is done. The copies take jobs off a shared counter
rather than being handed a fixed share: a session's cost depends on the size of its table,
and a fixed share leaves everyone waiting for the copy that drew the big ones.

`SelfPlayCollector.collect` forks too (`rl/rollout.py`), for the same reason, but it hands
tables out in its own way and does not go through here.

Pure Python: torch is touched only if it is already imported.
"""

from __future__ import annotations

import multiprocessing
import sys
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing.connection import wait
from typing import TypeVar

T = TypeVar("T")


@contextmanager
def torch_seeded(seed: int) -> Iterator[None]:
    """Seed torch's global stream for the block and put it back after.

    What decides a model's action is torch's stream, so a job that is to play the same hands
    wherever it runs must seed it. Putting it back keeps a job run in the parent from
    disturbing the stream the parent itself draws from. Does nothing if torch is not loaded."""
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    state = torch.get_rng_state()
    torch.manual_seed(seed % (2**63))
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _copy_main(work: Callable[[int], T], count: int, taken, sender) -> None:
    """The body of one copy: take job numbers off the counter until none is left, send each
    result back as it is done, and report a failure instead of dying silently."""
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.set_num_threads(1)  # a copy must not open a thread pool of its own
        while True:
            with taken.get_lock():
                job = taken.value
                taken.value += 1
            if job >= count:
                break
            sender.send(("result", job, work(job)))
    except BaseException:  # noqa: BLE001 - handed to the parent, which raises
        sender.send(("error", traceback.format_exc(), None))
    finally:
        sender.send(("done", None, None))
        sender.close()


def run_in_copies(
    work: Callable[[int], T],
    count: int,
    workers: int,
    *,
    on_result: Callable[[int, T], None] | None = None,
) -> list[T]:
    """`[work(0), ..., work(count - 1)]`, played by up to `workers` forked copies.

    `work` is inherited by the copies, not pickled, so it may be any closure; its *result*
    goes through a pipe and must be picklable. It has to be a pure function of its job
    number (seed whatever it draws from that number): then the list is the same whatever
    the number of copies, and with `workers <= 1` -- or a single job -- everything runs here,
    in order, with no fork at all. `on_result(job, result)` is called in the parent as each
    result arrives, in the order of arrival.

    A job that raises makes the whole call raise `RuntimeError` with its traceback, after
    the other copies have finished or been reaped."""
    workers = min(workers, count)
    if workers <= 1:
        results = []
        for job in range(count):
            result = work(job)
            results.append(result)
            if on_result is not None:
                on_result(job, result)
        return results

    context = multiprocessing.get_context("fork")
    taken = context.Value("i", 0)
    live = {}
    processes = []
    for _ in range(workers):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_copy_main, args=(work, count, taken, sender))
        process.start()
        sender.close()  # the copy holds the other end; EOF here means it died
        live[receiver] = process
        processes.append(process)

    results: list = [None] * count
    arrived = 0
    failure: str | None = None
    while live:
        for connection in wait(list(live)):
            try:
                kind, first, second = connection.recv()
            except EOFError:
                failure = failure or "a copy exited without reporting"
                live.pop(connection)
                continue
            if kind == "result":
                results[first] = second
                arrived += 1
                if on_result is not None and failure is None:
                    on_result(first, second)
            elif kind == "error":
                failure = failure or first
            else:  # done
                live.pop(connection)
    for process in processes:
        process.join()
    if failure is not None:
        raise RuntimeError(f"a forked copy failed:\n{failure}")
    if arrived != count:
        raise RuntimeError(f"forked copies returned {arrived} of {count} jobs")
    return results
