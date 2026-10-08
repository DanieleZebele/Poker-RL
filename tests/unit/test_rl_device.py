from __future__ import annotations

import subprocess
import sys

import pytest

from pokerlab.rl.device import resolve_device


def test_an_explicit_device_is_left_alone_and_never_imports_torch():
    code = (
        "import sys; from pokerlab.rl.device import resolve_device;"
        "assert resolve_device('cpu') == 'cpu' and resolve_device('cuda:1') == 'cuda:1';"
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_auto_picks_the_gpu_only_when_torch_sees_one():
    torch = pytest.importorskip("torch")
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
