"""RS-S04: the recovery-study driver must control interaction-FWER under the
null-interaction scenarios.

The two prespecified scenarios ``global_null`` and ``interaction_null`` contain
main effects (and, for ``interaction_null``, nonlinear transforms) but **no true
interactions**.  A faithful hierarchical interaction-discovery stage must
therefore control the family-wise error rate for interaction selection at the
configured ``alpha`` — i.e. it must not let main-effect signal leak into the
interaction scores.

This is a keystone gate for the method-evidence claim in the manuscript: the
empirical interaction-FWER measured on the composed driver pipeline must stay at
or below ``alpha`` (within Monte-Carlo error), not merely for the isolated maxT
rule (covered by RS-S01).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "run_recovery_study.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("_rrs_driver", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rrs_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


def _small_scale(driver, alpha: float = 0.2):
    return driver.StudyScale(
        n_inputs=10,
        n_outputs=12,
        n_runs=500,
        B=99,
        B_screen=99,
        fwer_reps=40,
        alt_reps=1,
        alpha=alpha,
        description="RS-S04 test scale",
    )


@pytest.mark.parametrize("scenario_name", ["global_null", "interaction_null"])
def test_interaction_null_fwer_controlled(driver, scenario_name):
    from rfm_pipeline import empirical_interaction_fwer, prespecified_recovery_scenarios

    scenarios = {s.name: s for s in prespecified_recovery_scenarios()}
    scenario = scenarios[scenario_name]
    alpha = 0.1
    scale = _small_scale(driver, alpha=alpha)

    master_rng = np.random.default_rng(20240117)
    flags = driver._run_fwer_replicates(scenario, scale, master_rng)
    stats = empirical_interaction_fwer(flags)

    # FWER must be controlled at alpha (allow Monte-Carlo slack: at fwer_reps=40
    # and alpha=0.1 the binomial SE is ~0.047, so alpha + ~3 SE ~= 0.24).
    assert stats["fwer_proportion"] <= alpha + 0.16, (
        f"{scenario_name}: empirical interaction-FWER "
        f"{stats['fwer_proportion']:.3f} exceeds alpha={alpha} "
        f"(main-effect leakage into interaction scores?); "
        f"flags={flags}"
    )
