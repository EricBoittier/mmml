"""
Fast HDF5 reporter for JAX MD simulation data.

Buffers frames in memory and flushes to disk in bulk to minimise I/O
overhead.  JAX arrays are transparently moved to host via a single
``jax.device_get`` call per flush.

Typical usage
-------------

.. code-block:: python

    from mmml.utils.hdf5_reporter import HDF5Reporter, DatasetSpec

    reporter = HDF5Reporter(
        "trajectory.h5",
        datasets={
            "potential_energy": DatasetSpec(shape=(), dtype="float64"),
            "kinetic_energy":   DatasetSpec(shape=(), dtype="float64"),
            "temperature":      DatasetSpec(shape=(), dtype="float64"),
            "invariant":        DatasetSpec(shape=(), dtype="float64"),
            "positions":        DatasetSpec(shape=(n_atoms, 3), dtype="float32"),
            "velocities":       DatasetSpec(shape=(n_atoms, 3), dtype="float32"),
        },
        buffer_size=100,
    )

    for step in range(n_steps):
        state = sim(state)
        if step % report_interval == 0:
            reporter.report(
                potential_energy=e_pot,
                kinetic_energy=e_kin,
                temperature=temp,
                invariant=H,
                positions=state.position,
                velocities=state.momentum / state.mass,
            )

    reporter.close()

Or as a context manager::

    with HDF5Reporter("traj.h5", datasets=...) as reporter:
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise ImportError(
        "h5py is required for HDF5Reporter. Install it with: pip install h5py"
    ) from exc

# Optional JAX import – the reporter works with plain numpy arrays too.
try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Schema for a single HDF5 dataset (one entry per recorded frame).

    Parameters
    ----------
    shape : tuple[int, ...]
        Per-frame shape.  Use ``()`` for scalars, ``(n_atoms, 3)`` for
        positions, etc.
    dtype : str | np.dtype
        NumPy-compatible dtype string (e.g. ``"float64"``, ``"float32"``).
    compression : str | None
        HDF5 compression filter.  ``"gzip"`` is a safe, portable default.
        Set to ``None`` to disable compression (fastest writes).
    compression_opts : int | None
        Compression level (0-9 for gzip).  ``1`` is fast with decent ratio.
    chunk_frames : int | None
        Number of frames per HDF5 chunk along axis 0.  ``None`` lets the
        reporter pick a sensible default (equal to ``buffer_size``).
    """

    shape: Tuple[int, ...] = ()
    dtype: str = "float64"
    compression: Optional[str] = "gzip"
    compression_opts: Optional[int] = 1
    chunk_frames: Optional[int] = None


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class HDF5Reporter:
    """Buffered HDF5 reporter for MD simulation observables.

    The reporter keeps the file handle open for the lifetime of the object
    (or until :meth:`close` is called) to avoid repeated open/close
    overhead.  Data is accumulated in NumPy buffers and flushed to disk
    every *buffer_size* frames in a single, contiguous write per dataset.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the output HDF5 file.
    datasets : dict[str, DatasetSpec]
        Mapping of dataset name -> specification.
    buffer_size : int
        Number of frames to buffer before flushing.  Larger values reduce
        I/O calls but use more memory.
    mode : str
        File open mode.  ``"w"`` truncates, ``"w-"`` fails if the file
        exists, ``"a"`` appends (useful for restarts).
    attrs : dict | None
        Optional top-level HDF5 attributes (metadata) written once at
        creation.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        datasets: Dict[str, DatasetSpec],
        buffer_size: int = 100,
        mode: str = "w",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._path = str(path)
        self._specs = dict(datasets)
        self._buffer_size = int(buffer_size)
        self._buf_idx = 0  # next free slot in the current buffer
        self._total_frames = 0  # frames already flushed to disk

        # Open the file and create the datasets.
        self._h5: h5py.File = h5py.File(self._path, mode=mode)

        if attrs is not None:
            for k, v in attrs.items():
                self._h5.attrs[k] = v

        # Per-dataset in-memory buffers (plain numpy).
        self._buffers: Dict[str, np.ndarray] = {}

        for name, spec in self._specs.items():
            chunk_frames = spec.chunk_frames or self._buffer_size
            per_frame_shape = spec.shape if spec.shape else ()
            full_chunk = (chunk_frames,) + per_frame_shape

            if name not in self._h5:
                create_kwargs: Dict[str, Any] = dict(
                    shape=(0,) + per_frame_shape,
                    maxshape=(None,) + per_frame_shape,
                    dtype=spec.dtype,
                    chunks=full_chunk,
                )
                if spec.compression is not None:
                    create_kwargs["compression"] = spec.compression
                    if spec.compression_opts is not None:
                        create_kwargs["compression_opts"] = spec.compression_opts

                self._h5.create_dataset(name, **create_kwargs)
            else:
                # Append mode – dataset already exists; validate shape.
                existing = self._h5[name]
                if existing.shape[1:] != per_frame_shape:
                    raise ValueError(
                        f"Dataset '{name}' shape mismatch: existing "
                        f"{existing.shape[1:]} vs spec {per_frame_shape}"
                    )
                self._total_frames = max(self._total_frames, existing.shape[0])

            # Allocate the in-memory buffer.
            self._buffers[name] = np.zeros(
                (self._buffer_size,) + per_frame_shape,
                dtype=spec.dtype,
            )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def report(self, **data: Any) -> None:
        """Record one frame of data.

        Keyword arguments must match the dataset names supplied at init.
        Values can be scalars, NumPy arrays, or JAX arrays.  Any JAX array
        is moved to the host lazily; the actual ``device_get`` is batched
        at flush time for efficiency.

        Raises
        ------
        ValueError
            If a key is not in the declared datasets.
        """
        for key, value in data.items():
            if key not in self._buffers:
                raise ValueError(
                    f"Unknown dataset '{key}'. "
                    f"Declared datasets: {list(self._specs)}"
                )
            arr = _to_numpy(value)
            expected_shape = self._specs[key].shape or ()
            # Allow scalar values for scalar datasets.
            if expected_shape == ():
                arr = np.asarray(arr, dtype=self._specs[key].dtype).reshape(())
            else:
                arr = np.asarray(arr, dtype=self._specs[key].dtype)
                if arr.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': "
                        f"got {arr.shape}, expected {expected_shape}"
                    )
            self._buffers[key][self._buf_idx] = arr

        self._buf_idx += 1
        if self._buf_idx >= self._buffer_size:
            self.flush()

    def report_batch(self, **data: Any) -> None:
        """Record a batch of frames at once (e.g. from ``lax.fori_loop`` output).

        Each value should have shape ``(n_frames,) + per_frame_shape``.
        The batch is converted to numpy in one shot, then written directly
        (bypassing the internal buffer) for maximum throughput.
        """
        if not data:
            return

        # Convert everything to numpy in one tree_map call if JAX is around.
        np_data = _tree_to_numpy(data)

        n_frames = None
        for key, arr in np_data.items():
            if key not in self._specs:
                raise ValueError(
                    f"Unknown dataset '{key}'. "
                    f"Declared datasets: {list(self._specs)}"
                )
            if n_frames is None:
                n_frames = arr.shape[0]
            elif arr.shape[0] != n_frames:
                raise ValueError(
                    f"Inconsistent batch size: '{key}' has {arr.shape[0]} "
                    f"frames but expected {n_frames}"
                )

        if n_frames is None or n_frames == 0:
            return

        # Flush any partial buffer first.
        if self._buf_idx > 0:
            self.flush()

        # Write directly to disk.
        start = self._total_frames
        end = start + n_frames
        for key, arr in np_data.items():
            dset = self._h5[key]
            new_len = end
            if dset.shape[0] < new_len:
                dset.resize(new_len, axis=0)
            dset[start:end] = arr

        self._total_frames = end
        self._h5.flush()

    def flush(self) -> None:
        """Flush the in-memory buffer to disk."""
        if self._buf_idx == 0:
            return

        n = self._buf_idx
        start = self._total_frames
        end = start + n

        for name, buf in self._buffers.items():
            dset = self._h5[name]
            new_len = end
            if dset.shape[0] < new_len:
                dset.resize(new_len, axis=0)
            dset[start:end] = buf[:n]

        self._total_frames = end
        self._buf_idx = 0
        self._h5.flush()

    def close(self) -> None:
        """Flush remaining data and close the file."""
        if self._h5 is not None:
            self.flush()
            self._h5.close()
            self._h5 = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Total number of frames written (flushed) so far."""
        return self._total_frames + self._buf_idx

    @property
    def n_flushed(self) -> int:
        """Number of frames already written to disk."""
        return self._total_frames

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_open(self) -> bool:
        return self._h5 is not None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HDF5Reporter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup.
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return (
            f"HDF5Reporter(path={self._path!r}, datasets={list(self._specs)}, "
            f"buffer_size={self._buffer_size}, frames={self.n_frames}, {status})"
        )


# ---------------------------------------------------------------------------
# Factory helpers for common simulation setups
# ---------------------------------------------------------------------------

def make_jaxmd_reporter(
    path: Union[str, os.PathLike],
    n_atoms: int,
    *,
    buffer_size: int = 100,
    include_positions: bool = True,
    include_velocities: bool = False,
    include_forces: bool = False,
    include_box: bool = False,
    box_shape: Tuple[int, ...] = (3, 3),
    scalar_quantities: Optional[Sequence[str]] = None,
    extra_datasets: Optional[Dict[str, DatasetSpec]] = None,
    pos_dtype: str = "float32",
    scalar_dtype: str = "float64",
    compression: Optional[str] = "gzip",
    mode: str = "w",
    attrs: Optional[Dict[str, Any]] = None,
) -> HDF5Reporter:
    """Create an :class:`HDF5Reporter` pre-configured for JAX MD simulations.

    By default the reporter records ``potential_energy``, ``kinetic_energy``,
    ``temperature``, and ``invariant`` as scalars, plus ``positions`` as a
    per-atom array.  Additional scalar quantities can be listed via
    *scalar_quantities*, and arbitrary extra datasets via *extra_datasets*.

    Parameters
    ----------
    path : str | PathLike
        Output file path.
    n_atoms : int
        Number of atoms in the system.
    buffer_size : int
        Frames to buffer before flushing.
    include_positions : bool
        Record atomic positions ``(n_atoms, 3)``.
    include_velocities : bool
        Record velocities ``(n_atoms, 3)``.
    include_forces : bool
        Record forces ``(n_atoms, 3)``.
    include_box : bool
        Record simulation box.
    box_shape : tuple
        Shape of box array (default ``(3, 3)`` for full cell matrix,
        use ``(3,)`` for orthorhombic lengths, etc.).
    scalar_quantities : sequence of str | None
        Extra scalar datasets to create beyond the four defaults.
    extra_datasets : dict | None
        Fully custom :class:`DatasetSpec` entries.
    pos_dtype : str
        Dtype for per-atom arrays.
    scalar_dtype : str
        Dtype for scalar quantities.
    compression : str | None
        Compression for all datasets.
    mode : str
        HDF5 file mode.
    attrs : dict | None
        Top-level HDF5 attributes.

    Returns
    -------
    HDF5Reporter
    """
    spec_kwargs = dict(compression=compression, compression_opts=1)
    datasets: Dict[str, DatasetSpec] = {
        "potential_energy": DatasetSpec(shape=(), dtype=scalar_dtype, **spec_kwargs),
        "kinetic_energy": DatasetSpec(shape=(), dtype=scalar_dtype, **spec_kwargs),
        "temperature": DatasetSpec(shape=(), dtype=scalar_dtype, **spec_kwargs),
        "invariant": DatasetSpec(shape=(), dtype=scalar_dtype, **spec_kwargs),
    }

    if include_positions:
        datasets["positions"] = DatasetSpec(
            shape=(n_atoms, 3), dtype=pos_dtype, **spec_kwargs
        )
    if include_velocities:
        datasets["velocities"] = DatasetSpec(
            shape=(n_atoms, 3), dtype=pos_dtype, **spec_kwargs
        )
    if include_forces:
        datasets["forces"] = DatasetSpec(
            shape=(n_atoms, 3), dtype=pos_dtype, **spec_kwargs
        )
    if include_box:
        datasets["box"] = DatasetSpec(
            shape=box_shape, dtype=pos_dtype, **spec_kwargs
        )

    if scalar_quantities:
        for name in scalar_quantities:
            datasets[name] = DatasetSpec(shape=(), dtype=scalar_dtype, **spec_kwargs)

    if extra_datasets:
        datasets.update(extra_datasets)

    return HDF5Reporter(
        path=path,
        datasets=datasets,
        buffer_size=buffer_size,
        mode=mode,
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    """Convert a value to a NumPy array on the host."""
    if _HAS_JAX and isinstance(x, jax.Array):
        return np.asarray(jax.device_get(x))
    return np.asarray(x)


def _tree_to_numpy(tree: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Batch-convert a dict of values (possibly JAX arrays) to numpy."""
    if _HAS_JAX:
        # A single device_get over the whole tree is faster than per-leaf.
        host_tree = jax.device_get(tree)
        return {k: np.asarray(v) for k, v in host_tree.items()}
    return {k: np.asarray(v) for k, v in tree.items()}
