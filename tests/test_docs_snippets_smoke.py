"""Smoke tests for documented CLI entry points and key imports.

Catches drift between documentation/configuration that promises a CLI
entry point and the package that's supposed to provide it (e.g. the
round-21 ``ModuleNotFoundError: tools`` regression where
``rfm-hpc-reduce`` worked from source tree but not from a pip install).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DOCS_SNIPPET_SMOKE_FILES = (
    "docs/setup_and_first_run.md",
    "docs/configuration_reference.md",
    "docs/troubleshooting.md",
    "docs/quickstart.md",
    "docs/reproducibility_example.md",
    "docs/PARALLEL_RUN_VALIDATION_CHECKLIST.md",
    "docs/SESSION_SUMMARY_2026_05_09.md",
)

PROHIBITED_COMMAND_PATTERNS = (
    (r"\bpixi shell\b", "Use `pixi run ...` instead of `pixi shell`."),
    (r"PYTHONPATH=src", "Do not require PYTHONPATH overrides in docs commands."),
    (
        r"^\s*(?:\$\s*)?(?:env\s+\S+=\S+\s+)*python3?\b",
        "Run project Python commands via `pixi run python ...`.",
    ),
    (
        r"(?:\||&&|;)\s*python3?\b",
        "Do not chain bare `python`; use `pixi run python ...`.",
    ),
)


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


def test_docs_snippets_follow_pixi_command_policy() -> None:
    """Critical docs snippets must not drift to non-Pixi command patterns."""
    violations: list[str] = []
    for rel_path in DOCS_SNIPPET_SMOKE_FILES:
        path = Path(rel_path)
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, message in PROHIBITED_COMMAND_PATTERNS:
                if re.search(pattern, line):
                    violations.append(f"{rel_path}:{line_no}: {message} -> {line.strip()}")

    assert not violations, "Command-policy snippet drift:\n" + "\n".join(violations)
