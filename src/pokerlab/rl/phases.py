"""The stages of one `poker-train` run, as the worker's log announces them.

`poker-loop --status` cannot ask a worker what it is doing: the supervisor is
blocked in `process.wait()` and learns nothing until the process exits. The log
is the only channel, so a worker writes a `phase: <name>` line as it enters each
stage and the watcher reads the last one. This module is the whole vocabulary,
shared by the three places that must agree on it (the worker that prints, the
population pass that announces its own stages, and the watcher that parses),
so a name cannot drift between them.

Pure Python, no torch: the watcher and the tests import it freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MARKER_PREFIX = "phase: "

EVALUATION = "evaluation"  # the learner plays the drawn pool to earn a rating
SERIES = "series"  # the published model is rated against the frozen anchors
ELO_PLAY = "elo_play"  # end-of-run population pass: playing the games
ELO_MERGE = "elo_merge"  # ... folding the results into the shared ratings
PRUNING = "pruning"  # ... and deleting the weakest models, when it is due
# Extra population passes a worker that finished early plays while the rest of
# its generation is still training. A separate name from ELO_PLAY on purpose:
# both are the same machinery, but one is a worker doing its own run's last
# stage and the other is a worker *waiting*, and `--status` has to be able to
# tell "behind" from "filling time" -- without it, a machine whose fast workers
# are all filling in looks identical to one whose slow workers are stuck.
ELO_FILL = "elo_fill"
DONE = "done"  # the very last line of a run that got to the end


# ---- the hyperparameters a worker was given ---------------------------------
#
# Every worker of a generation now trains with its own settings, drawn by
# `loop.hyperparameter_plan`. They live in three places already -- the worker's
# command line, the supervisor's log, and the metadata inside the model it
# publishes -- and **none of the three is what a watcher reads**: `--status` and
# the dashboard parse the worker's own log, and nothing else. So the worker
# prints them, once, as it starts.
#
# It is also what makes an archived log self-describing. A log that says a run
# reached entropy 0.4 and -30 bb/100 is worth much less if what that run was
# configured with has to be reconstructed from a supervisor log that has since
# rotated.
#
# Deliberately `key=value` pairs and not a fixed-shape line: `parse_hyperparameters`
# hardcodes no key, so adding an axis to the sweep needs no change here and no
# change in the watcher. Values never contain spaces (they are numbers and one
# single-word arm name), which is what makes splitting on whitespace safe.
HYPERPARAMETERS_PREFIX = "iperparametri: "


def hyperparameters_marker(values) -> str:
    """The log line that records what this worker was configured with."""
    rendered = []
    for key, value in values.items():
        rendered.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return HYPERPARAMETERS_PREFIX + " ".join(rendered)


def parse_hyperparameters(line: str) -> dict[str, str] | None:
    """What a log line records, or None if it is not one of these lines.

    Values stay strings: this is read to be displayed, and parsing `0.00038`
    back into a float only to format it again would lose the spelling the run
    actually used. Tolerant of a truncated tail for the same reason as
    `parse_progress` -- the logs are read while the worker is still writing.
    """
    stripped = line.strip()
    if not stripped.startswith(HYPERPARAMETERS_PREFIX):
        return None
    values: dict[str, str] = {}
    for token in stripped[len(HYPERPARAMETERS_PREFIX) :].split():
        key, separator, value = token.partition("=")
        if separator and key and value:
            values[key] = value
    return values or None


# ---- the two file names the fill-in phase is coordinated by -----------------
#
# Here for the same reason the stage names are: a worker writes one and its
# supervisor writes the other, and a name that drifted between the two modules
# would not fail -- it would leave every worker filling in until its deadline
# while the supervisor waited for workers it thought were still training.
#
# The handshake is deliberately two one-way files rather than anything cleverer.
# The supervisor cannot simply wait for its workers, because a worker in this
# phase does not exit: it is waiting for the supervisor. So the worker announces
# "I am only filling time now" by creating `FILL_DRAINING_FILENAME` in its own
# scratch directory, the supervisor polls for one per worker, and when every
# worker is either gone or draining it creates `FILL_STOP_FILENAME` in its state
# directory, which every worker polls between passes. Both are created and
# deleted, never written to, so there is no partial-read problem and nothing to
# parse; both live on the machine's *own* disk, never on the shared volume, so
# two machines cannot stop each other's workers.
FILL_DRAINING_FILENAME = "draining"
FILL_STOP_FILENAME = "FILL_STOP"


def marker(name: str) -> str:
    """The log line that announces stage `name`."""
    return f"{MARKER_PREFIX}{name}"


def parse_marker(line: str) -> str | None:
    """The stage a log line announces, or None if it is not a marker."""
    stripped = line.strip()
    if not stripped.startswith(MARKER_PREFIX):
        return None
    name = stripped[len(MARKER_PREFIX) :].strip()
    return name or None


# ---- progress inside a stage ----------------------------------------------
#
# A stage marker says *what* a worker is doing; these lines say *how far in* it
# is. They exist for the two stages that take the best part of an hour each and
# print, between them, four lines: the population pass (`ELO_PLAY`) and the
# benchmark pass (`SERIES`). Without them `--status` can only show
# "elo: gioco" for ninety minutes -- long enough that the 10-minute stale-log
# warning fires routinely on a worker that is perfectly healthy -- and cannot
# answer the one question actually being asked, which is how much is left.
#
# Same contract as `marker`/`parse_marker`, and here for the same reason: the
# format is written in one place and parsed in one place, and both are this one.

PROGRESS_PREFIX = "avanzamento "

# The ETA is emitted in whole minutes. A stage that lasts an hour does not need
# seconds, and minutes keep the line readable by a human tailing the log while
# staying trivial to parse back.
_PROGRESS_HEAD = re.compile(r"^avanzamento\s+(\S+):\s*(\d+)\s*/\s*(\d+)")
_PROGRESS_ETA = re.compile(r"~(\d+)m rimasti")


@dataclass(frozen=True)
class Progress:
    """How far into `stage` a worker is, as its log reports it.

    `done`/`total` are whatever unit the stage counts (sessions for the benchmark
    pass, and for the population pass); `detail` is free text for the
    reader and nothing decides anything from it.
    """

    stage: str
    done: int
    total: int
    detail: str = ""
    eta_seconds: float | None = None

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.done / self.total)

    @property
    def percent(self) -> int:
        return round(100 * self.fraction)


def progress_marker(
    stage: str,
    done: int,
    total: int,
    *,
    detail: str = "",
    eta_seconds: float | None = None,
) -> str:
    """The log line that reports progress inside stage `stage`."""
    reported = Progress(stage=stage, done=done, total=total)
    line = f"{PROGRESS_PREFIX}{stage}: {done}/{total} ({reported.percent}%)"
    if detail:
        line += f", {detail}"
    if eta_seconds is not None:
        line += f", ~{max(1, round(eta_seconds / 60))}m rimasti"
    return line


def parse_progress(line: str) -> Progress | None:
    """The progress a log line reports, or None if it is not one of these lines.

    Tolerant of a truncated line by construction: the logs are read while the
    worker is still appending to them, so a half-written tail must read as "no
    progress line here" rather than raise.
    """
    stripped = line.strip()
    if not stripped.startswith(PROGRESS_PREFIX):
        return None
    matched = _PROGRESS_HEAD.match(stripped)
    if matched is None:
        return None
    tail = stripped[matched.end() :]
    eta: float | None = None
    found = _PROGRESS_ETA.search(tail)
    if found is not None:
        eta = int(found.group(1)) * 60.0
        tail = tail[: found.start()]
    # Whatever is left after the fraction, its percentage and the ETA is the
    # free-text detail.
    detail = tail.split(")", 1)[-1].strip().strip(",").strip()
    return Progress(
        stage=matched.group(1),
        done=int(matched.group(2)),
        total=int(matched.group(3)),
        detail=detail,
        eta_seconds=eta,
    )
