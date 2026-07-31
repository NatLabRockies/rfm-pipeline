"""Focused tests for stage_loaders._read_parquet_with_mode out-of-core paths.

The distributed shard worker reads dataset parquet files through
``_read_parquet_with_mode``. When out-of-core mode is enabled with
spill-to-disk, the function must drive :class:`SpillToDiskBuffer` through its
real API and return the fully reconstructed DataFrame.
"""

from __future__ import annotations

import pandas as pd

from rfm_pipeline.config import OutOfCoreConfig
from rfm_pipeline.distributed.stage_loaders import _read_parquet_with_mode


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": list(range(200)),
            "a": [float(i) for i in range(200)],
            "b": [float(i) * 2 for i in range(200)],
        }
    )


def test_direct_read_when_out_of_core_disabled(tmp_path):
    df = _frame()
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    out = _read_parquet_with_mode(p, out_of_core=OutOfCoreConfig(enabled=False))
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df)


def test_chunked_read_without_spill(tmp_path):
    df = _frame()
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    cfg = OutOfCoreConfig(enabled=True, enable_spill_to_disk=False, chunk_size_mb=1)
    out = _read_parquet_with_mode(p, out_of_core=cfg)
    pd.testing.assert_frame_equal(out.sort_values("sample_id").reset_index(drop=True), df)


def test_chunked_read_with_spill_to_disk(tmp_path):
    df = _frame()
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    cfg = OutOfCoreConfig(
        enabled=True,
        enable_spill_to_disk=True,
        chunk_size_mb=1,
        max_memory_budget_mb=1,
        temp_dir=str(tmp_path / "spill"),
    )
    out = _read_parquet_with_mode(p, out_of_core=cfg)
    pd.testing.assert_frame_equal(out.sort_values("sample_id").reset_index(drop=True), df)


def test_spill_read_respects_column_subset(tmp_path):
    df = _frame()
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    cfg = OutOfCoreConfig(
        enabled=True,
        enable_spill_to_disk=True,
        chunk_size_mb=1,
        max_memory_budget_mb=1,
        temp_dir=str(tmp_path / "spill"),
    )
    out = _read_parquet_with_mode(p, out_of_core=cfg, columns=["sample_id", "a"])
    assert sorted(out.columns) == ["a", "sample_id"]
    assert len(out) == len(df)
