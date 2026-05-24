"""Shared pytest fixtures and sys.path setup."""

import os
import sys

# Ensure project root is on sys.path so `train.*` imports work from tests/.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
