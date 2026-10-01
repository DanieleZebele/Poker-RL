# Present so pytest's rootless import mode puts `tests/` on sys.path,
# making `tests/support.py` importable as a bare `support` module from both
# `tests/unit/` and `tests/integration/` without a `tests/__init__.py`
# (which would change how pytest names and collects every test module).
