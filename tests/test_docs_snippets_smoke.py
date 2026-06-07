"""Smoke tests for documented CLI entry points and key imports.

Catches drift between documentation/configuration that promises a CLI
entry point and the package that's supposed to provide it (e.g. the
round-21 ``ModuleNotFoundError: tools`` regression where
``rfm-hpc-reduce`` worked from source tree but not from a pip install).
"""

from __future__ import annotations

import subprocess
import sys


def test_rfm_hpc_reduce_help_runs() -> None:
    """``rfm-hpc-reduce --help`` must succeed via the installed entry point."""
    result = subprocess.run(
        ["rfm-hpc-reduce", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"rfm-hpc-reduce --help failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "--manifest" in result.stdout
    assert "--stage" in result.stdout


def test_manuscript_pipeline_helpers_import_cleanly() -> None:
    """Helpers used by ``hpc_reduce`` must import without sys.path hacks."""
    cmd = [
        sys.executable,
        "-c",
        (
            "from rfm_pipeline.manuscript_pipeline_helpers import ("
            "_load_empirical_null_screening_result,"
            "_load_interaction_discovery_result,"
            "_load_nonlinear_discovery_result,"
            "_load_output_conditioning_result,"
            "_load_sparse_selection_result,"
            "_load_tables,"
            "config_to_legacy_case_study)"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"helper import failed: stderr={result.stderr!r}"


def test_manuscript_runtime_context_import() -> None:
    """``load_manuscript_case_study_config`` is the canonical runtime loader."""
    cmd = [
        sys.executable,
        "-c",
        "from rfm_pipeline.manuscript_runtime import load_manuscript_case_study_config",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"manuscript_runtime import failed: stderr={result.stderr!r}"


def test_chrome_resolver_imports() -> None:
    """Chrome resolver helper must be importable from the installed package."""
    cmd = [
        sys.executable,
        "-c",
        "from rfm_pipeline._chrome import find_chrome; assert callable(find_chrome)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"chrome resolver import failed: stderr={result.stderr!r}"
