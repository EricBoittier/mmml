"""The production ML calculator path, with the bundled checkpoint.

Everything else in this suite measures a kernel; this measures what a user
actually calls. ``setup_calculator`` resolves the checkpoint, rebuilds the
PhysNet module from its stored config, and returns both an ASE calculator and
the jitted ``spherical_fn`` the jax-md path uses.

The two are benchmarked side by side on purpose. They evaluate the same model on
the same geometry, so the gap between ``ASEMLEnergy`` and ``JaxMLEnergy`` is
pure calculator overhead — ASE array marshalling, unit conversion, and the
per-call neighbour setup that the jitted path hoists out. That gap is the
argument for running MD through jax-md rather than through ASE, and it should be
a measured number rather than folklore.

``track_setup_seconds`` covers the other half of the user experience: checkpoint
load plus first compile, which dominates short jobs.
"""

from __future__ import annotations

import time

import numpy as np

from ._common import aco_cluster, block, default_checkpoint, require_jax, skip


def _build_calculator(n_monomers: int):
    """``(calc, spherical_fn, geometry)`` for an ACO cluster from the bundled JSON."""
    try:
        from mmml.interfaces.pycharmmInterface.calculator_utils import (
            unpack_factory_result,
        )
        from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise skip(f"mmml calculator unavailable: {exc}") from exc

    ckpt = default_checkpoint()
    geom = aco_cluster(int(n_monomers))
    n_atoms = int(geom["R"].shape[0])

    factory = setup_calculator(
        ATOMS_PER_MONOMER=geom["atoms_per_monomer"],
        N_MONOMERS=int(n_monomers),
        doML=True,
        doMM=False,
        model_restart_path=str(ckpt),
        MAX_ATOMS_PER_SYSTEM=n_atoms,
        defer_xla_gpu_warmup=True,
        verbose=False,
    )
    calc, spherical_fn, _ = unpack_factory_result(
        factory(
            atomic_numbers=geom["Z"],
            atomic_positions=geom["R"],
            n_monomers=int(n_monomers),
        )
    )
    return calc, spherical_fn, geom


class ASEMLEnergy:
    """``atoms.get_potential_energy()`` / ``get_forces()`` through the ASE calculator."""

    params = [2, 3, 4]
    param_names = ["n_monomers"]
    timeout = 1200.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 30.0)

    def setup(self, n_monomers):
        require_jax()
        try:
            import ase
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"ase unavailable: {exc}") from exc

        calc, _spherical, geom = _build_calculator(n_monomers)
        self.atoms = ase.Atoms(numbers=geom["Z"], positions=geom["R"])
        self.atoms.calc = calc
        # First call compiles; keep it out of the samples.
        self.atoms.get_potential_energy()
        self.atoms.get_forces()

    def _dirty(self):
        """Invalidate ASE's result cache so the next call really recomputes.

        Without this, ``get_forces`` after ``get_potential_energy`` on an
        unchanged geometry returns a cached array and the benchmark measures a
        dictionary lookup.
        """
        self.atoms.positions += 0.0
        self.atoms.calc.results.clear()

    def time_energy(self, n_monomers):
        self._dirty()
        self.atoms.get_potential_energy()

    def time_forces(self, n_monomers):
        self._dirty()
        block(self.atoms.get_forces())

    def track_setup_seconds(self, n_monomers):
        """Checkpoint load + module rebuild + first compile, from cold."""
        start = time.perf_counter()
        calc, _spherical, geom = _build_calculator(n_monomers)
        import ase

        atoms = ase.Atoms(numbers=geom["Z"], positions=geom["R"])
        atoms.calc = calc
        atoms.get_potential_energy()
        return time.perf_counter() - start

    track_setup_seconds.unit = "seconds"


class JaxMLEnergy:
    """The jitted ``spherical_fn`` the jax-md path drives, plus its force gradient."""

    params = [2, 3, 4]
    param_names = ["n_monomers"]
    timeout = 1200.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 30.0)

    def setup(self, n_monomers):
        jax = require_jax()
        try:
            import e3x

            from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"e3x / cutoffs unavailable: {exc}") from exc

        import jax.numpy as jnp

        _calc, spherical_fn, geom = _build_calculator(n_monomers)
        n_atoms = int(geom["R"].shape[0])

        dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
        pair_idx = jnp.stack([dst, src], axis=1)
        pair_mask = jnp.ones(pair_idx.shape[0], dtype=jnp.float32)
        cutoff_params = CutoffParameters()
        z = jnp.asarray(geom["Z"])

        def energy(positions):
            out = spherical_fn(
                atomic_numbers=z,
                positions=positions,
                n_monomers=int(n_monomers),
                cutoff_params=cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=True,
                debug=False,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
            )
            return out.energy.reshape(-1)[0]

        self.R = jnp.asarray(geom["R"])
        self.energy = jax.jit(energy)
        self.forces = jax.jit(jax.grad(energy))
        block(self.energy(self.R))
        block(self.forces(self.R))

    def time_energy(self, n_monomers):
        block(self.energy(self.R))

    def time_forces(self, n_monomers):
        block(self.forces(self.R))


class CheckpointLoad:
    """Just the restart path: read the JSON and rebuild the module + params."""

    timeout = 600.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 10, 30.0)

    def setup(self):
        require_jax()
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                resolve_checkpoint,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"mlpot.cli_common unavailable: {exc}") from exc

        self.resolve = resolve_checkpoint
        self.path = default_checkpoint()
        self.resolve(self.path)

    def time_resolve_checkpoint(self):
        self.resolve(self.path)

    def track_checkpoint_kb(self):
        return self.path.stat().st_size / 1024.0

    track_checkpoint_kb.unit = "KiB"


def _dimer_scan_positions(n_points: int) -> np.ndarray:
    """ACO dimer separations along x — the geometry every scan script sweeps."""
    geom = aco_cluster(2, spacing=3.0)
    per = geom["atoms_per_monomer"]
    base = geom["R"]
    frames = np.repeat(base[None, ...], n_points, axis=0)
    offsets = np.linspace(0.0, 6.0, n_points)
    frames[:, per:, 0] += offsets[:, None]
    return frames


class DimerScanThroughput:
    """Batched dimer-scan evaluation — the inner loop of the validation scripts.

    ``vmap`` over scan points is the difference between a scan that finishes in
    seconds and one that finishes in minutes, and it is the shape most of the
    repo's analysis code wants.
    """

    params = [16, 64, 256]
    param_names = ["n_points"]
    timeout = 1200.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 10, 30.0)

    def setup(self, n_points):
        jax = require_jax()
        try:
            import e3x

            from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"e3x / cutoffs unavailable: {exc}") from exc

        import jax.numpy as jnp

        _calc, spherical_fn, geom = _build_calculator(2)
        n_atoms = int(geom["R"].shape[0])
        dst, src = e3x.ops.sparse_pairwise_indices(n_atoms)
        pair_idx = jnp.stack([dst, src], axis=1)
        pair_mask = jnp.ones(pair_idx.shape[0], dtype=jnp.float32)
        cutoff_params = CutoffParameters()
        z = jnp.asarray(geom["Z"])

        def energy(positions):
            out = spherical_fn(
                atomic_numbers=z,
                positions=positions,
                n_monomers=2,
                cutoff_params=cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=True,
                debug=False,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
            )
            return out.energy.reshape(-1)[0]

        self.frames = jnp.asarray(_dimer_scan_positions(int(n_points)))
        self.scan = jax.jit(jax.vmap(energy))
        self.serial = jax.jit(energy)
        block(self.scan(self.frames))
        block(self.serial(self.frames[0]))

    def time_vmapped_scan(self, n_points):
        block(self.scan(self.frames))

    def time_serial_scan(self, n_points):
        block([self.serial(frame) for frame in self.frames])
