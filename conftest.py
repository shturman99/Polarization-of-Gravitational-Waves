"""Make the test suite importable with a bare ``pytest`` from the repo root.

Adds the repo root (for ``import src.gw_turbulence`` used by the kernel/plotting
tests), ``src/`` (for ``import gw_turbulence`` used by the paper-claim tests), and
``Notebooks/`` (for the figure-script modules those tests import) to ``sys.path``,
so the suite runs whether or not the package has been ``pip install -e .``'d and
regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "src", ROOT / "Notebooks"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
