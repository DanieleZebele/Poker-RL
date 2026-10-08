# Present so pytest's rootless import mode puts `tests/` on sys.path,
# making `tests/support.py` importable as a bare `support` module from both
# `tests/unit/` and `tests/integration/` without a `tests/__init__.py`
# (which would change how pytest names and collects every test module).

import gc
import sys
from pathlib import Path

import pytest

# The studies are scripts, not part of the package (`src/` never imports them), run with
# their own folder on the path: the tests import them the same way.
STUDY_DIRS = (Path(__file__).resolve().parents[1] / "studies" / "agents",)
for _folder in STUDY_DIRS:
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

# Every test file belongs to one area, named by its file name, so a change can be
# checked by running only its area (`pytest -m rl`, or `tests/affected.py`).
# Applied here rather than decorated by hand so a new file is covered by default:
# anything that is not gui/vision/rl/config/study is the engine.
AREA_PREFIXES = (
    ("test_gui_", "gui"),
    ("test_vision_", "vision"),
    ("test_rl_", "rl"),
    ("test_config", "config"),
    ("test_study_", "study"),
)
# The one test that launches real `poker-loop` and `poker-train` subprocesses.
SLOW_FILES = {"test_rl_loop_e2e.py"}


def pytest_collection_modifyitems(items):
    for item in items:
        name = item.path.name
        area = next((area for prefix, area in AREA_PREFIXES if name.startswith(prefix)), "engine")
        item.add_marker(getattr(pytest.mark, area))
        if name in SLOW_FILES:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def app():
    """The one Tk root of the whole test run, shared by every GUI test file.

    Creating more than one `tk.Tk()` in a process is unstable with this
    project's Tcl/Tk install on Windows: once four GUI test files each held a
    module-scoped root, a run failed one time in three at setup with
    `invalid command name "tcl_findLibrary"`. Each test parents its frames to
    this root and destroys only those frames, never the root."""
    pytest.importorskip("tkinter")
    from pokerlab.gui.app import PokerGuiApp

    application = PokerGuiApp()
    application.withdraw()
    yield application
    # Collect the frames' StringVars here, on the main thread, while the Tk
    # interpreter is still alive. Left to chance they are collected later by
    # whichever thread happens to trigger a GC -- a test file's worker -- and
    # `Variable.__del__` then raises "main thread is not in main loop" as an
    # unraisable exception, failing an unrelated test.
    gc.collect()
    application.destroy()
    gc.collect()
