"""Tests for the source-tree import smoke utility."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_smoke_cli_runs() -> None:
    """The import-smoke tool should succeed from the repository root."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/import_smoke.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Imported bsm_rfm" in result.stdout
