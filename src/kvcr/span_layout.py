# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unambiguous projections of stored objects for partial delivery."""

from functools import lru_cache

from .types import MemDescriptor


@lru_cache(maxsize=512)
def delivery_indices(
    source: tuple[str, ...], destination: tuple[str, ...]
) -> tuple[int, ...] | None:
    """Keep exact layouts compatible; subsets need unique, nonempty names.

    A projection preserves stored span order. Reordering or an ambiguous
    repeated/unnamed layout must not silently change the meaning of a block.
    The bounded cache stores layout metadata only, never addresses or claims.
    """
    if source == destination:
        return tuple(range(len(source)))
    if (
        not destination
        or len(destination) >= len(source)
        or any(not name for name in source + destination)
        or len(set(source)) != len(source)
        or len(set(destination)) != len(destination)
    ):
        return None
    by_name = {name: index for index, name in enumerate(source)}
    try:
        indices = tuple(by_name[name] for name in destination)
    except KeyError:
        return None
    return indices if tuple(sorted(indices)) == indices else None


def select_delivery_spans(
    sources: list[MemDescriptor] | tuple[MemDescriptor, ...],
    destinations: list[MemDescriptor] | tuple[MemDescriptor, ...],
    *,
    allow_subset: bool,
) -> tuple[MemDescriptor, ...] | None:
    source = tuple(span.info for span in sources)
    destination = tuple(span.info for span in destinations)
    if not allow_subset and source != destination:
        return None
    indices = delivery_indices(source, destination)
    if indices is None:
        return None
    selected = tuple(sources[index] for index in indices)
    if any(src.size != dst.size for src, dst in zip(selected, destinations)):
        return None
    return selected
