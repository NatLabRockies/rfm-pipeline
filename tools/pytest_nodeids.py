"""Minimal pytest plugin that emits machine-readable collected node IDs."""

from __future__ import annotations

NODEID_PREFIX = "RFM_CI_NODEID="


def pytest_collection_finish(session) -> None:
    """Print every collected test item for the CI shard launcher."""
    for item in session.items:
        print(f"{NODEID_PREFIX}{item.nodeid}")
