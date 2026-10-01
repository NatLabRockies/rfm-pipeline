"""Sphinx configuration for the rfm-pipeline documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _read_project_version(pyproject_path: Path) -> str:
    """Return the project version declared in ``pyproject.toml``."""
    match = re.search(
        r'^version = "([^"]+)"$',
        pyproject_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Could not find project version in {pyproject_path}")
    return match.group(1)


project = "rfm-pipeline"
author = "Dylan Hettinger"
release = _read_project_version(ROOT / "pyproject.toml")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render NumPy ``Attributes`` sections as ivar fields so dataclass field
# members and attribute descriptions do not register duplicate object
# descriptions during autodoc indexing.
napoleon_use_ivar = True
autodoc_typehints = "description"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "MEMORY.md",
    "AGENT_SYNC.md",
    "WORKFLOW_MANUSCRIPT_ALIGNMENT_PLAN.md",
    "final_scripts_from_hpc/*",
]
include_patterns = [
    "index.md",
    "overview.md",
    "setup_and_first_run.md",
    "quickstart.md",
    "reproducibility_example.md",
    "configuration_reference.md",
    "artifact_reference.md",
    "interpreting_results.md",
    "export_bundle.md",
    "HPC_DISTRIBUTED_EXECUTION.md",
    "troubleshooting.md",
    "scope_boundary.md",
    "api.rst",
]
html_theme = "alabaster"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
myst_heading_anchors = 3
