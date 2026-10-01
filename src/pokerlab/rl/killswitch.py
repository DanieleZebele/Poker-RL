"""An immediate, unconditional stop of everything this machine's loop is running.

`./run.sh stop` is the polite way out: it lets the generation in flight finish.
This is the other one, for when that is too slow or the loop is wedged. It does
not ask any process to wind down and it does not care what stage one is in --
training, benchmark, the Elo round, a pruning pass, or hung on a read: it freezes
the whole tree and then kills it.

Two properties matter, and both are why this is not just `pkill -f poker-`:

  * **Nothing can escape.** The supervisor starts a new generation's workers
    between one look at the process table and the next, and a naive kill loop
    races it. So every process is first stopped (SIGSTOP), which cannot be
    caught and leaves a process unable to fork or to clean up; the table is
    rescanned until no new member turns up; and only then does SIGKILL go out to
    all of them. A frozen supervisor cannot spawn the worker that would survive.
  * **Only this machine's loop.** Processes are matched by what they run (the
    supervisor, `poker-train`, the global-round shards) *and* by naming this
    machine's state or work directory on their command line, plus everything
    that descends from them. Another loop, a hand-started `poker-train` in a
    different directory, or an editor with one of these words in its arguments is
    left alone. `/proc` only shows this host, which is exactly the scope wanted:
    the project directory is shared over NFS but every machine runs its own loop.

It does not clean up. A killed worker leaves its scratch directory and maybe a
per-model lock: the next start's sweep publishes any unpublished archive and
removes the rest, and locks expire on their own (2 minutes; 30 for a pruning
pass). Deliberately no torch and no import of `loop.py`: it has to start in a
fraction of a second on a machine whose cores are all busy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# What the processes to stop are running. Matched against the entry point of a
# command line (the script or the `-m` module), never against any argument, so a
# `grep` or `tail` that merely mentions these names is not mistaken for one.
SUPERVISOR = frozenset({"pokerlab.rl.loop", "poker-loop"})
WORKER = frozenset({"pokerlab.rl.train", "poker-train"})
SHARD = frozenset({"pokerlab.rl.global_arena"})

# Kept equal to `loop.STATE_FILENAME`: importing it from there would pull in
# torch (a test pins the two together).
STATE_FILENAME = "loop_state.json"
DEFAULT_STATE_DIR = Path("checkpoints/state")
DEFAULT_WORK_DIR = Path("checkpoints/work")

# The most times the table is rescanned for newcomers before giving up. A frozen
# tree stops growing after the first pass or two; this is only a backstop.
MAX_PASSES = 50
_WORKER_NAME = re.compile(r"gen\d{4}-w\d{2}")


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    argv: tuple[str, ...]
    cwd: Path | None
    state: str


@dataclass
class KillReport:
    """What one call found and did. `stopped` lists every process frozen, as
    `(pid, role, name)`; `survivors` is what was still alive after SIGKILL."""

    stopped: list[tuple[int, str, str]] = field(default_factory=list)
    survivors: list[int] = field(default_factory=list)
    denied: list[int] = field(default_factory=list)

    def count(self, role: str) -> int:
        return sum(1 for _pid, r, _name in self.stopped if r == role)


def snapshot(proc_root: Path = Path("/proc")) -> dict[int, Proc]:
    """Every live process on this host. A process that exits mid-scan is skipped."""
    table: dict[int, Proc] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name in `stat` may itself contain spaces and parentheses,
        # so the fields that follow it start after the *last* closing one.
        try:
            fields = stat.rsplit(")", 1)[1].split()
            state, ppid = fields[0], int(fields[1])
        except (IndexError, ValueError):
            continue
        try:
            cwd: Path | None = Path(os.readlink(entry / "cwd"))
        except OSError:
            cwd = None
        argv = tuple(part for part in raw.decode(errors="replace").split("\0") if part)
        table[int(entry.name)] = Proc(int(entry.name), ppid, argv, cwd, state)
    return table


def _entrypoints(argv: tuple[str, ...]) -> set[str]:
    """What a command line actually runs: the script (`poker-train`, possibly
    behind the interpreter) or the module after `-m`."""
    found: set[str] = set()
    if argv:
        found.add(Path(argv[0]).name)
    if len(argv) > 1:
        found.add(Path(argv[1]).name)
    if "-m" in argv[:6]:
        index = argv.index("-m")
        if index + 1 < len(argv):
            found.add(argv[index + 1])
    return found


def role_of(proc: Proc) -> str | None:
    entry = _entrypoints(proc.argv)
    if entry & SUPERVISOR:
        return "supervisor"
    if entry & WORKER:
        return "worker"
    if entry & SHARD:
        return "shard"
    return None


def _root_variants(*roots: Path) -> tuple[Path, ...]:
    """Each directory as given, made absolute, and with symlinks resolved: a
    process may have been started with any of the three spellings."""
    variants: dict[Path, None] = {}
    for root in roots:
        variants[Path(os.path.normpath(root))] = None
        variants[Path(os.path.abspath(root))] = None
        variants[Path(os.path.realpath(root))] = None
    return tuple(variants)


def _under(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def belongs_to(proc: Proc, roots: tuple[Path, ...]) -> bool:
    """Whether the process names one of this machine's directories: the supervisor
    through `--state-dir`/`--work-dir`, a worker through its scratch directory
    and live checkpoint, both of which live under the work directory."""
    candidates: list[str] = []
    for token in proc.argv[1:]:
        candidates.append(token.split("=", 1)[1] if token.startswith("--") and "=" in token else token)
    base = proc.cwd or Path("/")
    # Cheap pass first: pure string arithmetic, no filesystem access.
    for token in candidates:
        if token.startswith("-") or not token:
            continue
        if _under(Path(os.path.normpath(base / token)), roots):
            return True
    # Slower pass, only for a process the first one did not claim: an absolute
    # path spelled through a symlink resolves to the same place as the root.
    for token in candidates:
        if token.startswith("/") and _under(Path(os.path.realpath(token)), roots):
            return True
    return False


def _descendants(table: dict[int, Proc], seeds: set[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for proc in table.values():
        children.setdefault(proc.ppid, []).append(proc.pid)
    found: set[int] = set()
    queue = list(seeds)
    while queue:
        for child in children.get(queue.pop(), ()):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def _protected(table: dict[int, Proc]) -> set[int]:
    """This process and every ancestor of it (the shell that ran the command):
    whatever happens, the caller must live to print its report."""
    protected = {1}
    pid = os.getpid()
    while pid and pid not in protected and pid in table:
        protected.add(pid)
        pid = table[pid].ppid
    protected.add(os.getpid())
    return protected


def _send(pid: int, signum: int, report: KillReport) -> None:
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass  # already gone: exactly what was wanted
    except PermissionError:
        if pid not in report.denied:
            report.denied.append(pid)


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # A killed process lingers as a zombie until its parent collects it; it is
    # dead for every purpose here.
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


_PRIORITY = {"supervisor": 0, "worker": 1, "shard": 2}


def kill_everything(
    state_dir: Path,
    work_dir: Path,
    *,
    dry_run: bool = False,
    settle_seconds: float = 10.0,
) -> KillReport:
    """Freeze, then kill, every process of this machine's loop. See the module
    docstring for why in that order. With `dry_run` nothing is signalled and the
    report lists what would be."""
    roots = _root_variants(state_dir, work_dir)
    report = KillReport()
    frozen: dict[int, Proc] = {}
    for _ in range(MAX_PASSES):
        table = snapshot()
        protected = _protected(table)
        seeds = {
            pid
            for pid, proc in table.items()
            if pid not in protected
            and proc.state != "Z"
            and role_of(proc) is not None
            and belongs_to(proc, roots)
        }
        wanted = seeds | _descendants(table, seeds | set(frozen))
        fresh = [
            table[pid]
            for pid in wanted
            if pid not in frozen and pid not in protected and table[pid].state != "Z"
        ]
        if not fresh:
            break
        # Supervisor first: the one process that could start more.
        fresh.sort(key=lambda proc: (_PRIORITY.get(role_of(proc) or "", 3), proc.pid))
        for proc in fresh:
            frozen[proc.pid] = proc
            role = role_of(proc) or "figlio"
            found = _WORKER_NAME.search(" ".join(proc.argv))
            report.stopped.append((proc.pid, role, found.group(0) if found else ""))
            if not dry_run:
                _send(proc.pid, signal.SIGSTOP, report)
        if dry_run:
            break

    if dry_run:
        return report

    for pid in frozen:
        _send(pid, signal.SIGKILL, report)
    deadline = time.monotonic() + settle_seconds
    remaining = [pid for pid in frozen if _alive(pid)]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = [pid for pid in remaining if _alive(pid)]
    report.survivors = remaining
    return report


def mark_killed(state_dir: Path) -> bool:
    """Record in `loop_state.json` that the loop was killed, so the status does
    not go on reporting a generation that no longer exists. Written like the
    supervisor writes it (temporary file, then rename), and left alone if there
    is no state to update."""
    path = state_dir / STATE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    data["phase"] = "killed"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Immediately kill this machine's poker-loop, its workers and their children."
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR,
                        help="the loop's state directory (what poker-loop was started with)")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                        help="the loop's work directory (what poker-loop was started with)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be killed, and kill nothing")
    args = parser.parse_args(argv)

    machine = socket.gethostname().split(".")[0]
    report = kill_everything(args.state_dir, args.work_dir, dry_run=args.dry_run)
    if not report.stopped:
        print(f"nessun processo del loop trovato su {machine} "
              f"(state-dir {args.state_dir}, work-dir {args.work_dir})")
        return 0

    verb = "TROVATI (dry-run, nessuno toccato)" if args.dry_run else "fermati"
    print(f"{machine}: {verb} {report.count('supervisor')} supervisor, "
          f"{report.count('worker')} worker, "
          f"{report.count('shard') + report.count('figlio')} processi figli")
    for pid, role, name in report.stopped:
        print(f"  {pid:>8}  {role:<10}{name}")
    if args.dry_run:
        return 0

    # Only mark the state once the supervisor really is gone.
    supervisors_alive = any(
        pid in report.survivors for pid, role, _ in report.stopped if role == "supervisor"
    )
    if not supervisors_alive:
        mark_killed(args.state_dir)
    if report.survivors:
        print(f"ANCORA VIVI dopo SIGKILL: {report.survivors} "
              "(probabilmente bloccati in I/O su NFS: spariranno quando l'I/O finisce)")
    if report.denied:
        print(f"NON AUTORIZZATO a segnalare: {report.denied} (processi di un altro utente?)")
    if not report.survivors and not report.denied:
        print("tutto fermo. Nessuna pulizia fatta: il prossimo `./run.sh start` recupera "
              "e pubblica gli archivi non pubblicati e rimuove gli scratch; i lock "
              "scadono da soli (2 minuti, 30 se era in pruning).")
    return 1 if (report.survivors or report.denied) else 0


if __name__ == "__main__":
    sys.exit(main())
