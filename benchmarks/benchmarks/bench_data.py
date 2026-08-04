"""Training-data batching and trajectory I/O.

Neither of these is a kernel, and both routinely dominate wall time anyway: a
training epoch that spends longer assembling batches than it does on gradients
is a real failure mode, and DCD writing runs on the host while the device waits.

``physnetjax.data.batches`` ships two implementations of the same contract —
``prepare_batches_jit`` and ``prepare_batches_fast``, which claims "same outputs"
via NumPy fancy indexing and a reusable pair cache. :class:`BatchPreparation`
puts a number on that claim's payoff, and on what the ``pair_cache`` argument is
worth on the second and later epochs.
"""

from __future__ import annotations

import numpy as np

from ._common import block, require_jax, skip


def _synthetic_dataset(n_samples: int, n_atoms: int, *, seed: int = 0) -> dict:
    """A PhysNet-shaped dataset dict (``R``/``Z``/``N``/``E``/``F``/``D``)."""
    rng = np.random.default_rng(seed)
    n_real = max(4, int(n_atoms * 0.6))
    z = np.zeros((n_samples, n_atoms), dtype=np.int32)
    z[:, :n_real] = rng.choice(np.array([1, 6, 7, 8]), size=(n_samples, n_real))
    return {
        "R": rng.uniform(-6.0, 6.0, size=(n_samples, n_atoms, 3)),
        "Z": z,
        "N": np.full((n_samples,), n_real, dtype=np.int32),
        "E": rng.normal(size=(n_samples, 1)),
        "F": rng.normal(scale=0.5, size=(n_samples, n_atoms, 3)),
        "D": rng.normal(size=(n_samples, 3)),
    }


class BatchPreparation:
    """``prepare_batches_jit`` vs. ``prepare_batches_fast`` for one epoch."""

    params = [(2048, 32), (2048, 64), (8192, 32)]
    param_names = ["n_samples_n_atoms"]
    timeout = 1200.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 10, 60.0)

    batch_size = 32

    def setup(self, n_samples_n_atoms):
        jax = require_jax()
        try:
            from mmml.models.physnetjax.physnetjax.data.batches import (
                _pair_indices,
                prepare_batches_fast,
                prepare_batches_jit,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"physnetjax.data.batches unavailable: {exc}") from exc

        n_samples, n_atoms = n_samples_n_atoms
        self.n_atoms = int(n_atoms)
        self.data = _synthetic_dataset(int(n_samples), self.n_atoms)
        self.key = jax.random.PRNGKey(0)
        self._jit = prepare_batches_jit
        self._fast = prepare_batches_fast
        # Second-and-later-epoch conditions: the pair layout is fixed by
        # (num_atoms, batch_size), so a real training loop builds it once.
        self.pair_cache = _pair_indices(self.n_atoms, self.batch_size)

    def time_prepare_batches_jit(self, n_samples_n_atoms):
        block(
            self._jit(
                self.key,
                self.data,
                self.batch_size,
                num_atoms=self.n_atoms,
            )
        )

    def time_prepare_batches_fast(self, n_samples_n_atoms):
        block(
            self._fast(
                self.key,
                self.data,
                self.batch_size,
                num_atoms=self.n_atoms,
            )
        )

    def time_prepare_batches_fast_cached_pairs(self, n_samples_n_atoms):
        block(
            self._fast(
                self.key,
                self.data,
                self.batch_size,
                num_atoms=self.n_atoms,
                pair_cache=self.pair_cache,
            )
        )


class BatchRotationAugmentation:
    """``rot_augment=True`` — the random-rotation augmentation used in training."""

    params = [False, True]
    param_names = ["rot_augment"]
    timeout = 1200.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 10, 60.0)

    n_samples, n_atoms, batch_size = 2048, 32, 32

    def setup(self, rot_augment):
        jax = require_jax()
        try:
            from mmml.models.physnetjax.physnetjax.data.batches import (
                prepare_batches_fast,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"physnetjax.data.batches unavailable: {exc}") from exc

        self.data = _synthetic_dataset(self.n_samples, self.n_atoms)
        self.key = jax.random.PRNGKey(0)
        self._fast = prepare_batches_fast
        self.rot_augment = bool(rot_augment)

    def time_prepare_epoch(self, rot_augment):
        block(
            self._fast(
                self.key,
                self.data,
                self.batch_size,
                num_atoms=self.n_atoms,
                rot_augment=self.rot_augment,
            )
        )


class PairIndexCache:
    """``_pair_indices`` — the ``(batch_size * n_atoms)^2`` edge layout itself."""

    params = [(32, 32), (64, 32), (32, 128)]
    param_names = ["n_atoms_batch_size"]
    timeout = 600.0
    number = 1
    repeat = (3, 10, 20.0)

    def setup(self, n_atoms_batch_size):
        require_jax()
        try:
            from mmml.models.physnetjax.physnetjax.data.batches import _pair_indices
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"physnetjax.data.batches unavailable: {exc}") from exc

        self.fn = _pair_indices
        self.n_atoms, self.batch_size = (int(v) for v in n_atoms_batch_size)

    def time_pair_indices(self, n_atoms_batch_size):
        block(self.fn(self.n_atoms, self.batch_size))


class DCDTrajectoryIO:
    """``save_trajectory_dcd`` / ``read_dcd_trajectory`` — the pure-Python DCD path.

    No MDAnalysis: this is the implementation the MD drivers actually write
    with, so its cost lands directly on every recorded frame.
    """

    params = [(200, 1500), (1000, 1500), (200, 8000)]
    param_names = ["n_frames_n_atoms"]
    timeout = 900.0
    number = 1
    repeat = (3, 10, 60.0)

    def setup(self, n_frames_n_atoms):
        import tempfile
        from pathlib import Path

        try:
            import ase

            from mmml.utils.dcd_reader import read_dcd_trajectory
            from mmml.utils.dcd_writer import save_trajectory_dcd
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"DCD I/O unavailable: {exc}") from exc

        n_frames, n_atoms = (int(v) for v in n_frames_n_atoms)
        rng = np.random.default_rng(0)
        self.positions = rng.uniform(0.0, 30.0, size=(n_frames, n_atoms, 3)).astype(
            np.float32
        )
        self.atoms = ase.Atoms(numbers=np.ones(n_atoms, dtype=int))
        self._save = save_trajectory_dcd
        self._read = read_dcd_trajectory

        self._tmpdir = tempfile.TemporaryDirectory(prefix="mmml-bench-dcd-")
        self.path = Path(self._tmpdir.name) / "traj.dcd"
        self._save(self.path, self.positions, self.atoms, dt_ps=0.0005)

    def teardown(self, n_frames_n_atoms):
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()

    def time_write_dcd(self, n_frames_n_atoms):
        self._save(self.path, self.positions, self.atoms, dt_ps=0.0005)

    def time_read_dcd(self, n_frames_n_atoms):
        self._read(self.path)
