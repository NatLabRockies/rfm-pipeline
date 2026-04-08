"""Sphinx configuration for the BSM reduced-form refactor docs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "BSM reduced-form model refactor"
author = "Dylan Hettinger"
release = "0.1.0"

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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "MEMORY.md"]
html_theme = "alabaster"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
