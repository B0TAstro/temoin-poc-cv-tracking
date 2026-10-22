"""Put the repo root on the path so `temoin_cv` imports like a package.

The apps are run with `python -m` from the root, which does this already; pytest
is launched from the root too, but this keeps the tests working whatever the
invocation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
