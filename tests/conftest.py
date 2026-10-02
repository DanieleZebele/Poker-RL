# Present so pytest's rootless import mode puts `tests/` on sys.path,
# making `tests/support.py` importable as a bare `support` module from both
# `tests/unit/` and `tests/integration/` without a `tests/__init__.py`
# (which would change how pytest names and collects every test module).

import gc

import pytest


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
