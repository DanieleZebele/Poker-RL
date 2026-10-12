from __future__ import annotations

from pokerlab.rl.memory import release_free_memory


def test_giving_memory_back_is_harmless_and_repeatable():
    junk = [[float(i) for i in range(660)] for _ in range(2000)]
    del junk
    release_free_memory()
    release_free_memory()


def test_it_works_where_there_is_no_malloc_trim(monkeypatch):
    import pokerlab.rl.memory as memory

    monkeypatch.setattr(memory.sys, "platform", "win32")
    memory.release_free_memory()  # a no-op, not an error
