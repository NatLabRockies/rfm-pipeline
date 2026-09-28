#!/usr/bin/env python
"""Create a content-addressed train/holdout layout for the G11 applied run.

This is a data-boundary operation, not an analysis.  It reads the existing
frozen split once, writes adaptive-stage inputs and holdout inputs to disjoint
directories, hashes every file, and removes read permissions from the holdout
response until the post-freeze prediction job explicitly authorizes access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BINARY_INPUT_NAMES = (
    "FM.Use Agnostic FS Conversion",
    "OI.Use AEO Reference Oil",
)
SOURCE_BINARY_INPUT_NAMES = ("AFSC", "UAEORO")
STRUCTURAL_INPUT_COLUMNS = ("scenario", "run_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_partitioned_parquet(
    source: Path,
    *,
    split_by_id: dict[Any, str],
    train_path: Path,
    holdout_path: Path,
    source_to_canonical: dict[str, str] | None = None,
    excluded_columns: tuple[str, ...] = (),
) -> tuple[int, int, int]:
    parquet = pq.ParquetFile(source)
    if "sample_id" not in parquet.schema_arrow.names:
        raise ValueError(f"{source.name} lacks sample_id")
    train_ids = pa.array(
        [key for key, split in split_by_id.items() if split == "train"]
    )
    holdout_ids = pa.array(
        [key for key, split in split_by_id.items() if split == "holdout"]
    )
    train_writer: pq.ParquetWriter | None = None
    holdout_writer: pq.ParquetWriter | None = None
    output_feature_count: int | None = None
    train_rows = 0
    holdout_rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=64):
            table = pa.Table.from_batches([batch])
            if excluded_columns:
                retained = [
                    name for name in table.column_names if name not in excluded_columns
                ]
                table = table.select(retained)
            if source_to_canonical:
                table = table.rename_columns(
                    [source_to_canonical.get(name, name) for name in table.column_names]
                )
            train_table = table.filter(
                pc.is_in(table["sample_id"], value_set=train_ids)
            )
            holdout_table = table.filter(
                pc.is_in(table["sample_id"], value_set=holdout_ids)
            )
            if train_writer is None:
                output_feature_count = len(table.column_names) - 1
                train_writer = pq.ParquetWriter(
                    train_path, table.schema, compression="zstd"
                )
                holdout_writer = pq.ParquetWriter(
                    holdout_path, table.schema, compression="zstd"
                )
            if train_table.num_rows:
                train_writer.write_table(train_table)
                train_rows += train_table.num_rows
            if holdout_table.num_rows:
                assert holdout_writer is not None
                holdout_writer.write_table(holdout_table)
                holdout_rows += holdout_table.num_rows
    finally:
        if train_writer is not None:
            train_writer.close()
        if holdout_writer is not None:
            holdout_writer.close()
    if train_writer is None or holdout_writer is None or output_feature_count is None:
        raise ValueError(f"{source.name} contained no rows")
    return train_rows, holdout_rows, output_feature_count


def prepare_applied_data_layout(
    source_root: Path,
    output_root: Path,
    *,
    expected_rows: int,
    expected_inputs: int,
    expected_outputs: int,
    expected_train_rows: int,
    expected_holdout_rows: int,
) -> dict[str, Any]:
    """Split immutable source parquets and seal holdout responses."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    required = {
        "X.parquet": source_root / "X.parquet",
        "Y.parquet": source_root / "Y.parquet",
        "fixed_holdout_assignments.parquet": source_root
        / "holdout_assignments.parquet",
        "manuscript_feature_catalog.parquet": source_root
        / "actual_input_feature_catalog.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse applied-data root: {output_root}")

    assignments = pq.read_table(required["fixed_holdout_assignments.parquet"])
    if assignments.column_names != ["sample_id", "split"]:
        raise ValueError("holdout assignments must contain exactly sample_id and split")
    ids = assignments["sample_id"].to_pylist()
    source_splits = [
        str(value).strip().lower() for value in assignments["split"].to_pylist()
    ]
    split_aliases = {"train": "train", "test": "holdout"}
    if any(value not in split_aliases for value in source_splits):
        raise ValueError("holdout assignments contain an unknown split label")
    splits = [split_aliases[value] for value in source_splits]
    if len(ids) != len(set(ids)) or set(splits) != {"train", "holdout"}:
        raise ValueError(
            "holdout assignments are duplicate or do not define both splits"
        )
    split_by_id = dict(zip(ids, splits, strict=True))
    if len(ids) != expected_rows:
        raise ValueError("assignment row count differs from the frozen applied scale")
    if (
        splits.count("train") != expected_train_rows
        or splits.count("holdout") != expected_holdout_rows
    ):
        raise ValueError("assignment split counts differ from the frozen applied scale")
    assignment_ids = set(ids)
    for stem in ("X", "Y"):
        source_ids = pq.read_table(required[f"{stem}.parquet"], columns=["sample_id"])[
            "sample_id"
        ].to_pylist()
        if len(source_ids) != expected_rows or len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{stem} sample IDs are duplicate or incomplete")
        if set(source_ids) != assignment_ids:
            raise ValueError(f"{stem} sample IDs differ from the frozen assignments")
        if source_ids != ids:
            raise ValueError(
                f"{stem} sample ID row order differs from the frozen assignments"
            )

    source_x_names = [
        name
        for name in pq.ParquetFile(required["X.parquet"]).schema_arrow.names
        if name not in {"sample_id", *STRUCTURAL_INPUT_COLUMNS}
    ]
    if len(source_x_names) != expected_inputs or not set(
        SOURCE_BINARY_INPUT_NAMES
    ) <= set(source_x_names):
        raise ValueError("X does not contain the frozen continuous/binary input schema")
    rename_inputs = dict(
        zip(SOURCE_BINARY_INPUT_NAMES, BINARY_INPUT_NAMES, strict=True)
    )
    x_names = [rename_inputs.get(name, name) for name in source_x_names]
    catalog = pq.read_table(required["manuscript_feature_catalog.parquet"])
    if not {"feature_name", "feature_type"} <= set(catalog.column_names):
        raise ValueError("feature catalog lacks feature_name or feature_type")
    catalog_names = [str(value) for value in catalog["feature_name"].to_pylist()]
    catalog_types = [str(value) for value in catalog["feature_type"].to_pylist()]
    first_order = {
        name
        for name, feature_type in zip(catalog_names, catalog_types, strict=True)
        if feature_type in {"first_order", "numeric"}
    }
    canonical_first_order = {rename_inputs.get(name, name) for name in first_order}
    if (
        len(catalog_names) != len(set(catalog_names))
        or canonical_first_order != set(x_names)
        or len(catalog_names) != expected_inputs
    ):
        raise ValueError("feature catalog does not exactly cover the applied inputs")

    train_root = output_root / "adaptive_train"
    holdout_root = output_root / "sealed_holdout"
    metadata_root = output_root / "metadata"
    train_root.mkdir(parents=True)
    holdout_root.mkdir()
    metadata_root.mkdir()
    try:
        counts: dict[str, tuple[int, int, int]] = {}
        for stem in ("X", "Y"):
            counts[stem] = _write_partitioned_parquet(
                required[f"{stem}.parquet"],
                split_by_id=split_by_id,
                train_path=train_root / f"{stem}.parquet",
                holdout_path=holdout_root / f"{stem}.parquet",
                source_to_canonical=rename_inputs if stem == "X" else None,
                excluded_columns=STRUCTURAL_INPUT_COLUMNS if stem == "X" else (),
            )
        if counts["X"][:2] != counts["Y"][:2]:
            raise ValueError("X/Y split row counts differ")
        if counts["X"] != (expected_train_rows, expected_holdout_rows, expected_inputs):
            raise ValueError("applied X dimensions differ from the frozen contract")
        if counts["Y"] != (
            expected_train_rows,
            expected_holdout_rows,
            expected_outputs,
        ):
            raise ValueError("applied Y dimensions differ from the frozen contract")

        x_train = pq.read_table(
            train_root / "X.parquet",
            columns=["sample_id", *BINARY_INPUT_NAMES],
        )
        x_holdout = pq.read_table(
            holdout_root / "X.parquet",
            columns=["sample_id", *BINARY_INPUT_NAMES],
        )
        expected_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        for split_name, table in (("train", x_train), ("holdout", x_holdout)):
            binary_columns = [table[name].to_pylist() for name in BINARY_INPUT_NAMES]
            if any(
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) not in {0.0, 1.0}
                for column in binary_columns
                for value in column
            ):
                raise ValueError(
                    f"{split_name} scenario inputs must be exactly binary 0/1"
                )
            binary_cells = {
                tuple(int(value) for value in row)
                for row in zip(
                    *binary_columns,
                    strict=True,
                )
            }
            if binary_cells != expected_cells:
                raise ValueError(
                    f"{split_name} split does not cover all four binary cells"
                )

        for split, root in (("train", train_root), ("holdout", holdout_root)):
            split_ids = [key for key, value in split_by_id.items() if value == split]
            pq.write_table(
                pa.table({"sample_id": split_ids, "split": [split] * len(split_ids)}),
                root / "assignments.parquet",
                compression="zstd",
            )
        canonical_catalog = catalog.set_column(
            catalog.schema.get_field_index("feature_name"),
            "feature_name",
            pa.array([rename_inputs.get(name, name) for name in catalog_names]),
        )
        canonical_catalog = canonical_catalog.set_column(
            canonical_catalog.schema.get_field_index("feature_type"),
            "feature_type",
            pa.array(["first_order"] * canonical_catalog.num_rows),
        )
        pq.write_table(
            canonical_catalog,
            metadata_root / "manuscript_feature_catalog.parquet",
            compression="zstd",
        )

        generated = sorted(
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.name != "dataset_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "status": "PREPARED_HOLDOUT_SEALED",
            "preparer_sha256": _sha256(Path(__file__).resolve()),
            "source_root": str(source_root),
            "source_sha256": {name: _sha256(path) for name, path in required.items()},
            "dimensions": {
                "rows": expected_rows,
                "train_rows": expected_train_rows,
                "holdout_rows": expected_holdout_rows,
                "inputs": expected_inputs,
                "continuous_inputs": expected_inputs - len(BINARY_INPUT_NAMES),
                "binary_inputs": len(BINARY_INPUT_NAMES),
                "outputs": expected_outputs,
            },
            "binary_input_names": list(BINARY_INPUT_NAMES),
            "source_binary_input_names": list(SOURCE_BINARY_INPUT_NAMES),
            "source_split_labels": {"train": "train", "test": "holdout"},
            "excluded_structural_input_columns": list(STRUCTURAL_INPUT_COLUMNS),
            "generated_sha256": {
                str(path.relative_to(output_root)): _sha256(path) for path in generated
            },
            "adaptive_stage_root": str(train_root),
            "holdout_stage_root": str(holdout_root),
            "holdout_response_mode": "000",
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["dataset_manifest_sha256"] = hashlib.sha256(
            canonical.encode()
        ).hexdigest()
        (output_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(holdout_root / "Y.parquet", 0o000)
        return manifest
    except Exception:
        # Preserve the partial root for forensic diagnosis; never silently reuse it.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_applied_data_layout(
        args.source_root,
        args.output_root,
        expected_rows=30_000,
        expected_inputs=160,
        expected_outputs=23_495,
        expected_train_rows=28_500,
        expected_holdout_rows=1_500,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
