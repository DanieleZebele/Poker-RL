"""Run only the tests a change can affect.

    python tests/affected.py src/pokerlab/rl/loop.py config.toml
    python tests/affected.py            # the files `git diff HEAD` lists

A change to GUI code has no business re-running the engine's fuzz tests, and the
other way round is the point of the areas: the engine feeds everything, the GUI
feeds nothing. The area of a test file comes from its name (see
`tests/conftest.py`). Anything this does not recognise runs the whole suite --
the safe direction. Extra arguments after `--` go to pytest (`-x`, `-k ...`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# source prefix -> the areas whose tests can see a change there. Order matters:
# the first match wins.
AREAS = (
    ("src/pokerlab/gui/", {"gui"}),
    ("src/pokerlab/vision/", {"vision", "gui"}),  # the spot screen reads the screen
    ("src/pokerlab/rl/", {"rl"}),
    ("src/pokerlab/config.py", {"config", "rl"}),
    ("src/pokerlab/cli/", {"engine", "gui"}),
    # The engine, cards, evaluator and players are what everything else runs on.
    ("src/pokerlab/", {"engine", "rl", "gui"}),
    ("run.sh", {"rl"}),
    ("config.toml", {"rl", "config"}),
)


def select(changed: list[str]) -> tuple[set[str], list[str]] | None:
    """`(areas, test files)` for `changed`, or None when everything must run."""
    areas: set[str] = set()
    files: list[str] = []
    for path in changed:
        if path.startswith("tests/") and path.endswith(".py"):
            if Path(path).name not in {"conftest.py", "support.py", "affected.py"}:
                files.append(path)
                continue
            return None  # shared test plumbing: anything may depend on it
        match = next((found for prefix, found in AREAS if path.startswith(prefix)), None)
        if match is None:
            if path.endswith((".md", ".txt")):
                continue
            return None
        areas |= match
    return areas, files


def main(argv: list[str]) -> int:
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    changed = argv or subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    chosen = select(changed)
    if chosen is None:
        print("affected: unrecognised change, running everything")
        command = [sys.executable, "-m", "pytest", *extra]
    else:
        areas, files = chosen
        if not areas and not files:
            print("affected: nothing that tests cover changed")
            return 0
        command = [sys.executable, "-m", "pytest", *files]
        if areas:
            command += ["-m", " or ".join(sorted(areas))]
        command += extra
        print(f"affected: areas {sorted(areas) or '-'}, files {files or '-'}")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
