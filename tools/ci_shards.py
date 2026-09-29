"""Deterministic workload partitioning for parallel CI jobs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def select_shard(items: Sequence[T], *, index: int, count: int) -> list[T]:
    """Return one stable round-robin shard from an ordered sequence."""
    if count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= index < count:
        raise ValueError(f"shard index must be between 0 and {count - 1}")
    return list(items[index::count])


def select_weighted_shard(
    items: Sequence[T],
    *,
    index: int,
    count: int,
    weight: Callable[[T], int],
) -> list[T]:
    """Return one deterministic shard balanced by estimated item weight."""
    if count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= index < count:
        raise ValueError(f"shard index must be between 0 and {count - 1}")

    weighted_items: list[tuple[int, int, T]] = []
    for position, item in enumerate(items):
        item_weight = weight(item)
        if item_weight < 1:
            raise ValueError("shard item weights must be positive")
        weighted_items.append((item_weight, position, item))

    shards: list[list[T]] = [[] for _ in range(count)]
    shard_weights = [0] * count
    for item_weight, _, item in sorted(weighted_items, key=lambda value: (-value[0], value[1])):
        target = min(
            range(count),
            key=lambda shard_index: (
                shard_weights[shard_index],
                len(shards[shard_index]),
                shard_index,
            ),
        )
        shards[target].append(item)
        shard_weights[target] += item_weight
    return shards[index]
