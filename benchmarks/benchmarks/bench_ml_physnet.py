"""PhysNet / SpookyNet forward and force evaluation (``mmml.models.physnetjax``).

The architecture defaults mirror the bundled ``DESdimers_params.json``
checkpoint (features=32, max_degree=1, 2 message-passing iterations, 16 radial
basis functions, 6 Å cutoff, ZBL on), so the scaling classes vary one dimension
away from a configuration that is actually in use rather than from an
arbitrary toy.

``time_energy`` vs ``time_energy_and_forces`` is the number that matters for MD:
forces come from ``jax.value_and_grad`` over the whole message-passing stack, and
the backward pass is what an MD step actually pays for.
"""

from __future__ import annotations

import numpy as np

from ._common import block, precision_tag, require_jax, skip

# Bundled-checkpoint architecture; every class below starts here.
BASE_ARCH = dict(
    features=32,
    max_degree=1,
    num_iterations=2,
    num_basis_functions=16,
    cutoff=6.0,
    max_atomic_number=118,
    charges=False,
    n_refinement_blocks=3,
    total_charge=0.0,
    zbl=True,
)


def _dense_inputs(n_atoms: int, *, n_real: int | None = None, seed: int = 0):
    """Padded single-molecule inputs with a dense (all-pairs) edge list.

    This is the layout the MD path feeds the model: ``max_padded_atoms`` slots,
    real atoms first, padding at the origin, and a ``batch_mask`` that kills every
    edge touching padding. Building it here (not in the timed region) keeps the
    benchmark on the model rather than on index bookkeeping.
    """
    import jax.numpy as jnp

    n_real = int(n_atoms if n_real is None else n_real)
    rng = np.random.default_rng(seed)

    # ~10 atoms per acetone-sized fragment, spread so the 6 Å cutoff sees a
    # realistic number of neighbours instead of one dense blob.
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    n_frag = max(1, n_real // 10)
    for f in range(n_frag):
        lo, hi = f * 10, min((f + 1) * 10, n_real)
        centre = rng.uniform(0.0, 4.0, size=3) + np.array([5.0 * f, 0.0, 0.0])
        positions[lo:hi] = centre + rng.uniform(-1.6, 1.6, size=(hi - lo, 3))

    z = np.zeros(n_atoms, dtype=np.int32)
    z[:n_real] = rng.choice(np.array([1, 6, 7, 8]), size=n_real)
    atom_mask = (z > 0).astype(np.float64)

    idx = np.arange(n_atoms)
    dst, src = np.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = dst != src
    dst, src = dst[keep], src[keep]
    batch_mask = (atom_mask[dst] > 0) & (atom_mask[src] > 0)

    return dict(
        atomic_numbers=jnp.asarray(z),
        positions=jnp.asarray(positions),
        dst_idx=jnp.asarray(dst),
        src_idx=jnp.asarray(src),
        batch_segments=jnp.zeros(n_atoms, dtype=jnp.int32),
        batch_size=1,
        batch_mask=jnp.asarray(batch_mask.astype(np.float64)),
        atom_mask=jnp.asarray(atom_mask),
    )


class _PhysNetBase:
    """Builds a jitted ``apply`` and warms it up; subclasses pick the axis to vary."""

    timeout = 600.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 20.0)

    def _build(self, n_atoms: int, **arch_overrides):
        jax = require_jax()
        try:
            from mmml.models.physnetjax.physnetjax.models.model import PhysNet
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"physnetjax unavailable: {exc}") from exc

        arch = dict(BASE_ARCH)
        arch.update(arch_overrides)
        model = PhysNet(max_padded_atoms=int(n_atoms), **arch)
        self.inputs = _dense_inputs(int(n_atoms))
        self.params = model.init(jax.random.PRNGKey(0), **self.inputs)

        static = {"batch_size", "compute_forces"}
        traced = {k: v for k, v in self.inputs.items() if k not in static}

        def _apply(params, traced_inputs, compute_forces: bool):
            return model.apply(
                params,
                batch_size=1,
                compute_forces=compute_forces,
                **traced_inputs,
            )

        self.model = model
        self.traced = traced
        self.energy_fn = jax.jit(lambda p, t: _apply(p, t, False))
        self.forces_fn = jax.jit(lambda p, t: _apply(p, t, True))

        # Compile outside the timed region; asv would otherwise fold a one-off
        # XLA compilation into the first sample of every benchmark.
        block(self.energy_fn(self.params, self.traced))
        block(self.forces_fn(self.params, self.traced))

    def _time_energy(self):
        block(self.energy_fn(self.params, self.traced))

    def _time_forces(self):
        block(self.forces_fn(self.params, self.traced))

    def _compile_seconds(self):
        """Wall time for a cold XLA compilation of the energy+forces graph."""
        import time

        import jax

        jax.clear_caches()
        fn = jax.jit(
            lambda p, t: self.model.apply(p, batch_size=1, compute_forces=True, **t)
        )
        start = time.perf_counter()
        block(fn(self.params, self.traced))
        return time.perf_counter() - start


class PhysNetSystemSize(_PhysNetBase):
    """Cost vs. number of (padded) atoms — the dominant MD scaling axis."""

    params = [20, 40, 80, 160, 320]
    param_names = ["n_atoms"]

    def setup(self, n_atoms):
        self._build(n_atoms)

    def time_energy(self, n_atoms):
        self._time_energy()

    def time_energy_and_forces(self, n_atoms):
        self._time_forces()

    def track_compile_energy_and_forces_s(self, n_atoms):
        return self._compile_seconds()

    track_compile_energy_and_forces_s.unit = "seconds"


class PhysNetWidth(_PhysNetBase):
    """Cost vs. feature width at a fixed 60-atom system."""

    params = [16, 32, 64, 128]
    param_names = ["features"]

    def setup(self, features):
        self._build(60, features=int(features))

    def time_energy(self, features):
        self._time_energy()

    def time_energy_and_forces(self, features):
        self._time_forces()


class PhysNetDepth(_PhysNetBase):
    """Cost vs. message-passing iterations at a fixed 60-atom system."""

    params = [1, 2, 3, 5]
    param_names = ["num_iterations"]

    def setup(self, num_iterations):
        self._build(60, num_iterations=int(num_iterations))

    def time_energy(self, num_iterations):
        self._time_energy()

    def time_energy_and_forces(self, num_iterations):
        self._time_forces()


class PhysNetAngularResolution(_PhysNetBase):
    """Cost vs. ``max_degree`` — the e3x spherical-harmonic order.

    Degree 0 is a plain invariant model; each extra degree adds ``2L+1``
    components to every equivariant feature, so this is where the accuracy /
    throughput trade-off gets decided.
    """

    params = [0, 1, 2, 3]
    param_names = ["max_degree"]

    def setup(self, max_degree):
        self._build(60, max_degree=int(max_degree))

    def time_energy(self, max_degree):
        self._time_energy()

    def time_energy_and_forces(self, max_degree):
        self._time_forces()


class PhysNetElectrostatics(_PhysNetBase):
    """Charge prediction + damped Coulomb on top of the base architecture.

    ``charges=True`` turns on the atomic-charge head, the total-charge
    constraint, and the electrostatic energy with its switching functions — a
    distinct chunk of graph that every latent-charge model in this repo pays for.
    """

    params = [False, True]
    param_names = ["charges"]

    def setup(self, charges):
        self._build(60, charges=bool(charges))

    def time_energy_and_forces(self, charges):
        self._time_forces()


class PhysNetZBL(_PhysNetBase):
    """ZBL short-range nuclear repulsion on/off (``physnetjax.models.zbl``)."""

    params = [False, True]
    param_names = ["zbl"]

    def setup(self, zbl):
        self._build(60, zbl=bool(zbl))

    def time_energy_and_forces(self, zbl):
        self._time_forces()


class SpookyNetSystemSize:
    """SpookyPhysNet forward+backward — the distillation teacher's architecture."""

    params = [20, 60, 160]
    param_names = ["n_atoms"]
    timeout = 600.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 20.0)

    def setup(self, n_atoms):
        jax = require_jax()
        try:
            from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"spooky_model unavailable: {exc}") from exc

        import jax.numpy as jnp

        model = SpookyPhysNet(max_padded_atoms=int(n_atoms), **BASE_ARCH)
        inputs = _dense_inputs(int(n_atoms))
        inputs["charges"] = jnp.zeros(int(n_atoms))
        inputs["spins"] = jnp.zeros(int(n_atoms))
        params = model.init(jax.random.PRNGKey(0), **inputs)

        traced = {k: v for k, v in inputs.items() if k != "batch_size"}
        fn = jax.jit(
            lambda p, t: model.apply(p, batch_size=1, compute_forces=True, **t)
        )
        self.params, self.traced, self.fn = params, traced, fn
        block(self.fn(self.params, self.traced))

    def time_energy_and_forces(self, n_atoms):
        block(self.fn(self.params, self.traced))


class ZBLRepulsionKernel:
    """The standalone ZBL pair kernel (``physnetjax.models.zbl.ZBLRepulsion``)."""

    params = [1_000, 10_000, 100_000]
    param_names = ["n_pairs"]
    timeout = 300.0
    warmup_time = 0.0
    number = 1
    repeat = (3, 20, 10.0)

    def setup(self, n_pairs):
        jax = require_jax()
        try:
            from mmml.models.physnetjax.physnetjax.models.zbl import ZBLRepulsion
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise skip(f"zbl module unavailable: {exc}") from exc

        import jax.numpy as jnp

        n_pairs = int(n_pairs)
        n_atoms = max(8, int(np.sqrt(2 * n_pairs)) + 1)
        rng = np.random.default_rng(0)

        z = jnp.asarray(rng.choice(np.array([1, 6, 7, 8]), size=n_atoms).astype(np.int32))
        idx_i = jnp.asarray(rng.integers(0, n_atoms, size=n_pairs).astype(np.int32))
        idx_j = jnp.asarray(rng.integers(0, n_atoms, size=n_pairs).astype(np.int32))
        # ZBL only bites inside ~0.6 Å; sample across the switch so the benchmark
        # exercises the switching branch rather than a uniformly-zero region.
        distances = jnp.asarray(rng.uniform(0.05, 1.2, size=n_pairs))
        atom_mask = jnp.ones(n_atoms)
        batch_mask = jnp.ones(n_pairs)
        batch_segments = jnp.zeros(n_atoms, dtype=jnp.int32)

        model = ZBLRepulsion(cutoff=0.6, cuton=0.1, trainable=False)
        call_args = (z, distances, None, None, idx_i, idx_j, atom_mask, batch_mask, batch_segments, 1)
        variables = model.init(jax.random.PRNGKey(0), *call_args)

        self.fn = jax.jit(
            lambda v, d: model.apply(
                v, z, d, None, None, idx_i, idx_j, atom_mask, batch_mask, batch_segments, 1
            )
        )
        self.variables, self.distances = variables, distances
        block(self.fn(self.variables, self.distances))

    def time_repulsion(self, n_pairs):
        block(self.fn(self.variables, self.distances))


def track_precision(*_args):
    """Records the process-global float precision the run used (32 or 64)."""
    return 64 if precision_tag() == "x64" else 32


track_precision.unit = "bits"
