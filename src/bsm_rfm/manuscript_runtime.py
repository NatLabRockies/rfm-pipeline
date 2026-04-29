"""Runtime helpers for manuscript-reproduction notebooks and data intake."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .manuscript_data_contract import (
    manuscript_notebook_order,
    manuscript_placeholder_path_policy,
    required_manuscript_artifacts,
)

_ARTIFACT_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "input_metadata": ("input_name", "input_kind", "units"),
    "output_metadata": ("output_name", "year", "units"),
    "case_study_input_matrix": ("sample_id", "x1", "x2", "AFSC", "UAEORO"),
    "case_study_output_matrix": ("sample_id", "y1", "y2", "y3"),
    "manuscript_feature_catalog": ("feature_name", "feature_type", "origin"),
    "fixed_holdout_assignments": ("sample_id", "split"),
}


@dataclass(frozen=True)
class ManuscriptRuntimeContext:
    """Resolved runtime context for manuscript notebooks."""

    mode: str
    repo_root: Path
    artifact_paths: dict[str, Path]
    output_root: Path
    unresolved_placeholders: tuple[str, ...]
    local_override_used: bool
    runtime_dir: Path | None


@dataclass(frozen=True)
class ManuscriptNotebookContext:
    """Resolved runtime and loaded tables for a manuscript notebook."""

    notebook_name: str
    runtime: ManuscriptRuntimeContext
    tables: dict[str, pd.DataFrame]
    case_study_config: dict[str, Any]
    runtime_manifest: dict[str, Any]


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def normalize_manuscript_repo_root(candidate: Path) -> Path:
    """Resolve the repository root from a repo root or any nested repo path."""
    path = candidate.expanduser().resolve()
    search_roots = [path, *path.parents]
    for root in search_roots:
        if (root / "pyproject.toml").exists() and (root / "configs").exists():
            return root
    raise FileNotFoundError(f"Could not resolve repository root from candidate path: {candidate}")


def load_manuscript_runtime_manifest(repo_root: Path) -> dict[str, Any]:
    """Load the tracked notebook/runtime manifest."""
    repo_root = normalize_manuscript_repo_root(repo_root)
    return _read_yaml_mapping(repo_root / "configs" / "manuscript_runtime.yml")


def load_manuscript_case_study_config(repo_root: Path) -> dict[str, Any]:
    """Load the tracked manuscript case-study contract."""
    repo_root = normalize_manuscript_repo_root(repo_root)
    return _read_yaml_mapping(repo_root / "configs" / "manuscript_case_study.yml")


def load_manuscript_paths_template(repo_root: Path) -> dict[str, str]:
    """Load the tracked placeholder-path template."""
    repo_root = normalize_manuscript_repo_root(repo_root)
    template_file = manuscript_placeholder_path_policy()["template_file"]
    data = _read_yaml_mapping(repo_root / template_file)
    return {str(key): str(value) for key, value in data.items()}


def load_manuscript_local_override(repo_root: Path) -> dict[str, str]:
    """Load the optional local real-data path override."""
    repo_root = normalize_manuscript_repo_root(repo_root)
    override_file = repo_root / manuscript_placeholder_path_policy()["local_override_file"]
    if not override_file.exists():
        return {}
    data = _read_yaml_mapping(override_file)
    return {str(key): str(value) for key, value in data.items()}


def unresolved_manuscript_placeholders(paths: dict[str, str]) -> tuple[str, ...]:
    """Return artifact names whose configured paths still use the placeholder prefix."""
    prefix = manuscript_placeholder_path_policy()["placeholder_prefix"]
    unresolved = [
        name
        for name in required_manuscript_artifacts()
        if str(paths.get(name, "")).startswith(prefix)
    ]
    return tuple(unresolved)


def write_demo_manuscript_artifacts(root: Path) -> dict[str, Path]:
    """Write a deterministic toy dataset matching the manuscript artifact contract."""
    root.mkdir(parents=True, exist_ok=True)
    sample_ids = list(range(1, 81))
    x1 = pd.Series([(value - 39.5) / 10.0 for value in range(80)], name="x1")
    x2 = pd.Series(
        [0.1 + float((value * 7) % 17) / 10.0 for value in range(80)],
        name="x2",
    )
    afsc = pd.Series([value % 2 for value in range(80)], name="AFSC")
    uaeoro = pd.Series([(value // 2) % 2 for value in range(80)], name="UAEORO")
    input_matrix = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x1": x1,
            "x2": x2,
            "AFSC": afsc,
            "UAEORO": uaeoro,
        }
    )
    output_matrix = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y1": 1.0 + 12.0 * x1 + 0.10 * x2,
            "y2": -0.5 - 10.0 * x1 + 0.25 * afsc,
            "y3": 2.0 + 8.0 * x1 + 0.25 * uaeoro,
        }
    )
    input_metadata = pd.DataFrame(
        {
            "input_name": ["x1", "x2", "AFSC", "UAEORO"],
            "input_kind": ["scalar", "scalar", "boolean", "boolean"],
            "units": ["unitless", "unitless", "flag", "flag"],
        }
    )
    output_metadata = pd.DataFrame(
        {
            "output_name": ["y1", "y2", "y3"],
            "year": [2015, 2015, 2015],
            "units": ["unitless", "unitless", "unitless"],
        }
    )
    feature_catalog = pd.DataFrame(
        {
            "feature_name": [
                "x1",
                "x2",
                "AFSC",
                "UAEORO",
                "x1:x2",
                "x1_squared",
                "log1p_x2",
            ],
            "feature_type": [
                "first_order",
                "first_order",
                "first_order",
                "first_order",
                "interaction",
                "transformation",
                "transformation",
            ],
            "origin": ["toy_demo"] * 7,
        }
    )
    holdout = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": ["train"] * 64 + ["holdout"] * 16,
        }
    )

    tables: dict[str, pd.DataFrame] = {
        "input_metadata": input_metadata,
        "output_metadata": output_metadata,
        "case_study_input_matrix": input_matrix,
        "case_study_output_matrix": output_matrix,
        "manuscript_feature_catalog": feature_catalog,
        "fixed_holdout_assignments": holdout,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def _read_tabular_artifact(path: Path) -> pd.DataFrame:
    """Load a tabular artifact from CSV or Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported artifact extension for {path}")


def validate_manuscript_artifact_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Return validation issues for loaded manuscript artifact tables."""
    issues: list[str] = []
    for artifact_name in required_manuscript_artifacts():
        if artifact_name not in tables:
            issues.append(f"Missing required artifact table: {artifact_name}")
            continue
        frame = tables[artifact_name]
        required_columns = _ARTIFACT_REQUIRED_COLUMNS[artifact_name]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            issues.append(
                f"Artifact {artifact_name} is missing required columns: {', '.join(missing)}"
            )
    return issues


def load_manuscript_artifact_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load manuscript artifact tables from resolved runtime paths."""
    tables = {name: _read_tabular_artifact(path) for name, path in paths.items()}
    issues = validate_manuscript_artifact_tables(tables)
    if issues:
        raise ValueError("; ".join(issues))
    return tables


def resolve_manuscript_runtime(repo_root: Path) -> ManuscriptRuntimeContext:
    """Resolve real-data paths when available, otherwise fall back to deterministic demo data."""
    repo_root = normalize_manuscript_repo_root(repo_root)
    template = load_manuscript_paths_template(repo_root)
    local_override = load_manuscript_local_override(repo_root)
    merged = dict(template)
    merged.update(local_override)
    unresolved = unresolved_manuscript_placeholders(merged)
    output_root_value = merged.get("output_root")
    local_override_used = bool(local_override)

    all_real_files_exist = not unresolved and all(
        Path(merged[name]).expanduser().exists() for name in required_manuscript_artifacts()
    )
    if all_real_files_exist:
        artifact_paths = {
            name: Path(merged[name]).expanduser().resolve()
            for name in required_manuscript_artifacts()
        }
        if output_root_value is None or str(output_root_value).startswith(
            manuscript_placeholder_path_policy()["placeholder_prefix"]
        ):
            output_root = (repo_root / "artifacts" / "manuscript-case-study").resolve()
        else:
            output_root = Path(output_root_value).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        mode = "real"
        runtime_dir = None
    else:
        runtime_dir = Path(tempfile.mkdtemp(prefix="bsm_manuscript_demo_"))
        artifact_paths = write_demo_manuscript_artifacts(runtime_dir / "data")
        output_root = runtime_dir / "artifacts"
        output_root.mkdir(parents=True, exist_ok=True)
        mode = "demo"

    return ManuscriptRuntimeContext(
        mode=mode,
        repo_root=repo_root,
        artifact_paths=artifact_paths,
        output_root=output_root,
        unresolved_placeholders=unresolved,
        local_override_used=local_override_used,
        runtime_dir=runtime_dir,
    )


def build_manuscript_notebook_context(
    repo_root: Path,
    notebook_name: str,
) -> ManuscriptNotebookContext:
    """Resolve runtime inputs and loaded tables for one manuscript notebook."""
    repo_root = normalize_manuscript_repo_root(repo_root)
    runtime = resolve_manuscript_runtime(repo_root)
    if notebook_name not in manuscript_notebook_order():
        raise ValueError(f"Unknown manuscript notebook: {notebook_name}")
    tables = load_manuscript_artifact_tables(runtime.artifact_paths)
    return ManuscriptNotebookContext(
        notebook_name=notebook_name,
        runtime=runtime,
        tables=tables,
        case_study_config=load_manuscript_case_study_config(repo_root),
        runtime_manifest=load_manuscript_runtime_manifest(repo_root),
    )


def manuscript_runtime_summary_table(context: ManuscriptNotebookContext) -> pd.DataFrame:
    """Summarize the resolved runtime context for notebook display."""
    rows = [
        {"field": "notebook_name", "value": context.notebook_name},
        {"field": "mode", "value": context.runtime.mode},
        {"field": "local_override_used", "value": str(context.runtime.local_override_used)},
        {"field": "output_root", "value": str(context.runtime.output_root)},
        {
            "field": "unresolved_placeholders",
            "value": ", ".join(context.runtime.unresolved_placeholders) or "none",
        },
    ]
    return pd.DataFrame(rows)
