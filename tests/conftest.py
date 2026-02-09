"""Shared pytest configuration and fixtures."""
import sys
from pathlib import Path

# Ensure project root and src are on path (for schemas, configs, components)
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
