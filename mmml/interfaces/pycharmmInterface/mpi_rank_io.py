"""Rank-0 I/O and logging helpers for ``np>1`` PyCHARMM / MLpot workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def mpi_rank_size() -> tuple[int, int]:
    """Return ``(rank, size)`` from mpi4py or OpenMPI env vars."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge import mpi_rank_size as _rank_size

    return _rank_size()


def is_mpi_rank_zero() -> bool:
    rank, _size = mpi_rank_size()
    return int(rank) == 0


def rank0_only(value: T, *, default: T | None = None) -> T | None:
    """Return ``value`` on rank 0, else ``default``."""
    if is_mpi_rank_zero():
        return value
    return default


def rank0_print(*args: Any, quiet: bool = False, **kwargs: Any) -> None:
    """``print`` on rank 0 when ``size>1`` (disable gating with ``MMML_MPI_RANK0_PRINT=0``)."""
    if quiet:
        return
    rank, size = mpi_rank_size()
    gate = os.environ.get("MMML_MPI_RANK0_PRINT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if gate and size > 1 and rank != 0:
        return
    print(*args, **kwargs)


def rank0_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> bool:
    """Write a text file on rank 0 only; return True when written."""
    if not is_mpi_rank_zero():
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)
    return True


def rank0_write_json(path: Path | str, payload: Any, *, indent: int = 2) -> bool:
    """Write JSON on rank 0 only."""
    return rank0_write_text(path, json.dumps(payload, indent=indent) + "\n")


def rank0_call(fn: Callable[[], T], *, default: T | None = None) -> T | None:
    """Run ``fn`` on rank 0 only."""
    if not is_mpi_rank_zero():
        return default
    return fn()
