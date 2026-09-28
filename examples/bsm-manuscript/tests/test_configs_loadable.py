"""Smoke test: every standalone HPC YAML must load against the pinned rfm-pipeline.

Catches drift between this repo's configs and renames/removals in the pinned
rfm-pipeline (e.g. round-3 ``lasso_alpha_percentile -> lasso_alpha_grid_size``
and round-4 ``transform_families`` rejection). Partial / template configs that
are merged at runtime are intentionally excluded.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from rfm_pipeline.config import load_config
from rfm_pipeline.hpc_workflow_config import load_hpc_workflow_config

REPO_ROOT = Path(__file__).resolve().parent.parent

STANDALONE_CONFIGS = [
    "configs/hpc/kestrel_publication_full_dataset.yml",
    "configs/hpc/kestrel_publication_full_dataset_distributed_base.yml",
    *sorted(
        p
        for p in glob.glob(str(REPO_ROOT / "configs/hpc/dev/kestrel_*.yml"))
        # ``workflow_*`` YAMLs are orchestration payloads consumed by
        # ``load_hpc_workflow_config`` (see ORCHESTRATION_CONFIGS below).
        if "workflow_" not in Path(p).name
    ),
]

ORCHESTRATION_CONFIGS = [
    "configs/hpc/kestrel_publication_orchestration.yml",
    *sorted(
        glob.glob(str(REPO_ROOT / "configs/hpc/dev/kestrel_workflow_*.yml"))
    ),
]

# Optional gitignored local override; loaded only when present so the test
# suite still passes on a fresh clone.
_LOCAL_ORCH = REPO_ROOT / "configs/hpc/kestrel_publication_orchestration.local.yml"
if _LOCAL_ORCH.exists():
    ORCHESTRATION_CONFIGS.append(str(_LOCAL_ORCH))


@pytest.mark.parametrize("path", STANDALONE_CONFIGS)
def test_config_loads(path: str) -> None:
    load_config(str(REPO_ROOT / path) if not Path(path).is_absolute() else path)


@pytest.mark.parametrize("path", ORCHESTRATION_CONFIGS)
def test_orchestration_config_loads(path: str) -> None:
    load_hpc_workflow_config(
        str(REPO_ROOT / path) if not Path(path).is_absolute() else path
    )
