"""G1-S03: AFSC/UAEORO no longer return 'Scenario' from canonical_module_from_factor_name."""

from __future__ import annotations

import pytest

from rfm_pipeline.features import canonical_module_from_factor_name


@pytest.mark.parametrize("name", ["AFSC", "UAEORO"])
def test_literal_names_not_scenario(name: str) -> None:
    result = canonical_module_from_factor_name(name)
    assert result != "Scenario", f"{name!r} should not map to 'Scenario'"
    assert result == "Unscoped", f"unscoped bare name {name!r} should map to 'Unscoped'"


def test_scoped_names_still_use_prefix() -> None:
    assert canonical_module_from_factor_name("OI.Use AEO Reference Oil") == "OI"
    assert canonical_module_from_factor_name("FM.Use Agnostic FS Conversion") == "FM"
    assert canonical_module_from_factor_name("X.y") == "X"


def test_generic_unscoped_still_unscoped() -> None:
    assert canonical_module_from_factor_name("SomeBareToken") == "Unscoped"
