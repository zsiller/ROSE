"""Shared sys.path setup for scripts run as `python path/to/script.py`."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root(__file__: str) -> str:
    """Put the project root on ``sys.path`` and drop the script directory so package imports resolve."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    return root
