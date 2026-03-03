"""MPI helpers for distributing grid evaluations across ranks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


try:
    from mpi4py import MPI  # type: ignore
except ImportError:  # pragma: no cover - depends on local MPI installation
    MPI = None


@dataclass(frozen=True)
class MPIContext:
    comm: object
    rank: int
    size: int


def get_mpi_context(enabled: bool) -> MPIContext | None:
    if not enabled:
        return None
    if MPI is None:
        raise RuntimeError("MPI requested but mpi4py is not installed.")
    comm = MPI.COMM_WORLD
    return MPIContext(comm=comm, rank=comm.Get_rank(), size=comm.Get_size())


def mpi_is_active() -> bool:
    if MPI is None:
        return False
    return MPI.COMM_WORLD.Get_size() > 1


def split_row_indices(num_rows: int, rank: int, size: int) -> np.ndarray:
    return np.arange(rank, num_rows, size, dtype=int)


def gather_grid(local_rows: dict[int, np.ndarray], shape: tuple[int, int], context: MPIContext) -> np.ndarray:
    gathered = context.comm.gather(local_rows, root=0)
    if context.rank != 0:
        return np.zeros(shape)
    grid = np.zeros(shape)
    for chunk in gathered:
        for row_index, row_values in chunk.items():
            grid[row_index] = row_values
    return grid
