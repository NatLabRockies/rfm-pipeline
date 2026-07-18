"""G1-S05: AFSC/UAEORO absent from src/rfm_pipeline and config.py docstrings are generic.

Acceptance criteria:
- Neither ``AFSC`` nor ``UAEORO`` appears anywhere under src/rfm_pipeline/.
- config.py class-level docstrings contain no ``BSM`` or ``JDS`` literals.
"""

from __future__ import annotations

from pathlib import Path

_SRC_ROOT = Path(__file__).parents[2] / "src" / "rfm_pipeline"
_CONFIG_FILE = _SRC_ROOT / "config.py"

_SCENARIO_TOKENS: frozenset[str] = frozenset(["AFSC", "UAEORO"])
_CONFIG_DENYLIST: frozenset[str] = frozenset(["BSM", "JDS"])


def test_scenario_tokens_absent_from_src() -> None:
    """AFSC and UAEORO must not appear anywhere in src/rfm_pipeline."""
    hits: list[str] = []
    for py_file in sorted(_SRC_ROOT.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        for token in sorted(_SCENARIO_TOKENS):
            if token in source:
                hits.append(f"{py_file.relative_to(_SRC_ROOT.parents[1])}:{token!r}")

    assert not hits, "Scenario-specific tokens found in src/rfm_pipeline/:\n" + "\n".join(
        f"  {h}" for h in hits
    )


def test_config_docstrings_are_generic() -> None:
    """config.py must not reference BSM or JDS in its docstrings."""
    source = _CONFIG_FILE.read_text(encoding="utf-8")
    hits = [token for token in sorted(_CONFIG_DENYLIST) if token in source]
    assert not hits, f"Publication-specific tokens still in config.py docstrings: {hits}"
