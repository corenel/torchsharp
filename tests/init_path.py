"""Init module path for scripts."""

import os
import sys


def add_path(path):
    """Add path to system PYTHONPATH."""
    if path not in sys.path:
        sys.path.insert(0, path)


# Add packaage to PYTHONPATH
cwd = os.path.dirname(__file__)
add_path(os.path.join(cwd, "..", "torchsharp"))
