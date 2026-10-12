"""Giving back the memory Python has already freed.

A training iteration builds, and then drops, a lot of small and medium blocks: every
decision carries a list of ~660 Python floats (~20 KB), a hand's trajectory holds a few of
them, and a batch holds thousands. When they are freed, glibc's `malloc` keeps most of the
pages in its free lists instead of handing them back to the system, and fragmentation
keeps the biggest blocks from merging, so the resident size of the process climbs with
each iteration (about 20 MB an iteration, 300-500 MB held at any moment, in a measured run)
although nothing is in use. On a machine running many workers that adds up to a full disk
of swap. `malloc_trim` returns those pages.

Linux/glibc only; anywhere else this does nothing.
"""

from __future__ import annotations

import ctypes
import gc
import sys

_libc = None


def release_free_memory() -> None:
    """Collect cycles, then ask `malloc` to return its free pages to the system."""
    global _libc
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
