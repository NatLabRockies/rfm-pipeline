"""Focused tests for distributed stage_loaders dataset-root resolution.

The distributed shard worker loads data via
``rfm_pipeline.distributed.stage_loaders._resolve_data_root``. This resolver
must honor an explicit ``config.dataset.path`` (the value the HPC controller
injects into each generated stage config) so that datasets living outside the
repository ``artifacts/`` tree (e.g. on cluster scratch) are found. When no
explicit path is set it falls back to a ``dataset.type``-derived layout under
``<study_root>/artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline.config import DatasetConfig, WorkflowConfig
from rfm_pipeline.distributed.stage_loaders import _resolve_data_root


def _config(path: str | None, dtype: str = "real_full_dataset") -> WorkflowConfig:
    return WorkflowConfig(dataset=DatasetConfig(type=dtype, path=path))


def test_absolute_dataset_path_is_honored(monkeypatch, tmp_path):
    monkeypatch.delenv("RFM_STUDY_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)  # ensure CWD fallback would differ from the path
    explicit = tmp_path / "elsewhere" / "preprocessed_real_data_30k"
    resolved = _resolve_data_root(_config(str(explicit)))
    assert resolved == explicit


def test_relative_dataset_path_is_resolved_against_study_root(monkeypatch, tmp_path):
    study_root = tmp_path / "study"
    monkeypatch.setenv("RFM_STUDY_ROOT", str(study_root))
    resolved = _resolve_data_root(_config("data/my_dataset"))
    assert resolved == study_root / "data" / "my_dataset"


def test_type_based_fallback_when_no_path(monkeypatch, tmp_path):
    study_root = tmp_path / "study"
    monkeypatch.setenv("RFM_STUDY_ROOT", str(study_root))
    resolved = _resolve_data_root(_config(None, dtype="real_full_dataset"))
    assert resolved == study_root / "artifacts" / "preprocessed_real_data_30k"


def test_type_based_fallback_uses_cwd_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RFM_STUDY_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_data_root(_config(None, dtype="real_full_dataset"))
    assert resolved == Path(tmp_path) / "artifacts" / "preprocessed_real_data_30k"
