"""Parity tests for the ``mm_nonbonded`` term vs. ``nonbonded_energy_and_forces``.

The reference is pure jax (given arrays), so these run without CHARMM. The term
reuses the same switching kernels, so agreement should be to ~machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mmml.data.units import KCAL_MOL_TO_EV  # noqa: E402
from mmml.md.energy import EnergyContext, available_terms  # noqa: E402
from mmml.md.energy.terms import MMNonbondedTerm  # noqa: E402
from mmml.md.system import FFParams, MolecularSystem  # noqa: E402


def _system_and_ref():
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        CharmmNbondSettings,
        NonbondedSystemData,
    )

    rng = np.random.default_rng(3)
    box = np.diag([20.0, 20.0, 20.0])
    # two 3-atom molecules, a few Å apart so intermolecular pairs fall in-cutoff
    pos = np.concatenate([
        rng.uniform(0.0, 2.0, size=(3, 3)),
        rng.uniform(5.0, 7.0, size=(3, 3)),
    ])
    n = 6
    mol_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    charges = rng.uniform(-0.6, 0.6, size=n)
    charges -= charges.mean()  # roughly neutral
    epsilon = rng.uniform(0.05, 0.2, size=n)
    rmin_half = rng.uniform(1.2, 2.0, size=n)
    at_codes = np.arange(n, dtype=np.int32)

    ff = FFParams(
        charges=charges,
        epsilon=epsilon,
        rmin_half=rmin_half,
        at_codes=at_codes,
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    system = MolecularSystem(R=pos, Z=np.ones(n, int), box=box, mol_id=mol_id, ff_params=ff)
    settings = CharmmNbondSettings(cutnb=12.0, ctonnb=10.0, ctofnb=12.0)
    nbdata = NonbondedSystemData(
        charges=charges, at_codes=at_codes, epsilon=epsilon, rmin=rmin_half,
        excluded_pairs=frozenset(), e14_pairs=frozenset(),
    )
    return system, settings, nbdata


def _reference_energy_eV(system, settings, nbdata):
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        nonbonded_energy_and_forces,
    )

    terms, forces = nonbonded_energy_and_forces(
        np.asarray(system.R), nbdata, np.asarray(system.box), settings,
        molecule_id=np.asarray(system.mol_id),
    )
    return float(terms["total"]) * KCAL_MOL_TO_EV, np.asarray(forces) * KCAL_MOL_TO_EV


def test_registered():
    assert "mm_nonbonded" in available_terms()


def test_jax_hostpair_matches_reference():
    system, settings, nbdata = _system_and_ref()
    fn = MMNonbondedTerm(settings).make(system, EnergyContext()).jax_energy_fn
    got = float(fn(jnp.asarray(system.R)))  # host-build path, all pairs
    ref, _ = _reference_energy_eV(system, settings, nbdata)
    assert got == pytest.approx(ref, rel=1e-9)
    assert got != 0.0  # sanity: there ARE intermolecular interactions


def test_padded_mask_matches_unpadded():
    system, settings, nbdata = _system_and_ref()
    term = MMNonbondedTerm(settings)
    fns = term.make(system, EnergyContext())
    pi, pj, e14, vdw14 = term._host_pairs(np.asarray(system.R), settings, system.ff_params, system.mol_id)

    R = jnp.asarray(system.R)
    e_unpadded = float(fns.jax_energy_fn(R, pair_i=pi, pair_j=pj, e14_scale=e14, vdw14_scale=vdw14))

    # pad with dummy (0,0) pairs masked off
    pad = 5
    pi_p = np.concatenate([pi, np.zeros(pad, dtype=np.int32)])
    pj_p = np.concatenate([pj, np.zeros(pad, dtype=np.int32)])
    e14_p = np.concatenate([e14, np.ones(pad)])
    vdw14_p = np.concatenate([vdw14, np.ones(pad)])
    mask = np.concatenate([np.ones(len(pi), dtype=np.int8), np.zeros(pad, dtype=np.int8)])
    e_padded = float(fns.jax_energy_fn(
        R, pair_i=pi_p, pair_j=pj_p, e14_scale=e14_p, vdw14_scale=vdw14_p, pair_mask=mask
    ))
    assert e_padded == pytest.approx(e_unpadded, rel=0, abs=1e-9)


def test_padded_path_is_jittable():
    system, settings, nbdata = _system_and_ref()
    term = MMNonbondedTerm(settings)
    fn = term.make(system, EnergyContext()).jax_energy_fn
    pi, pj, e14, vdw14 = term._host_pairs(np.asarray(system.R), settings, system.ff_params, system.mol_id)
    jfn = jax.jit(lambda R, pi, pj, e14, vdw14: fn(R, pair_i=pi, pair_j=pj, e14_scale=e14, vdw14_scale=vdw14))
    got = float(jfn(jnp.asarray(system.R), jnp.asarray(pi), jnp.asarray(pj), jnp.asarray(e14), jnp.asarray(vdw14)))
    ref, _ = _reference_energy_eV(system, settings, nbdata)
    assert got == pytest.approx(ref, rel=1e-9)


def test_ase_face_matches_reference():
    system, settings, nbdata = _system_and_ref()
    contribution = MMNonbondedTerm(settings).make(system, EnergyContext()).ase_contribution

    class _Stub:
        def __init__(self, system):
            self._pos = np.asarray(system.R)
            self._cell = np.asarray(system.box)

        def get_positions(self):
            return self._pos

        @property
        def cell(self):
            class _C:
                array = self._cell
            return _C()

        def __len__(self):
            return len(self._pos)

    energy, forces = contribution(_Stub(system))
    ref_e, ref_f = _reference_energy_eV(system, settings, nbdata)
    assert energy == pytest.approx(ref_e, rel=1e-9)
    assert np.allclose(forces, ref_f, atol=1e-8)


# --- lr_solver: mic (jax face, jit-compatible) vs. everything else (ASE only) ---


class _PosStub:
    def __init__(self, system):
        self._pos = np.asarray(system.R)
        self._cell = np.asarray(system.box)

    def get_positions(self):
        return self._pos

    @property
    def cell(self):
        class _C:
            array = self._cell

        return _C()

    def __len__(self):
        return len(self._pos)


def test_jax_face_rejects_non_mic_lr_solver():
    """jax_pme/etc. are host-orchestrated (ASE Atoms + host neighbor list in
    jax-pme's own evaluator) and cannot be traced inside jax.jit; calling the
    jax face with a non-mic solver must fail loudly, not silently use mic."""
    system, settings, _ = _system_and_ref()
    fn = MMNonbondedTerm(settings, lr_solver="jax_pme").make(system, EnergyContext()).jax_energy_fn
    with pytest.raises(NotImplementedError, match="lr_solver"):
        fn(jnp.asarray(system.R))


def test_ase_face_jax_pme_differs_from_mic_and_is_finite():
    pytest.importorskip("jaxpme")
    system, settings, _ = _system_and_ref()

    mic_contribution = MMNonbondedTerm(settings, lr_solver="mic").make(system, EnergyContext()).ase_contribution
    pme_contribution = MMNonbondedTerm(
        settings, lr_solver="jax_pme", jax_pme_sr_cutoff_A=6.0,
    ).make(system, EnergyContext()).ase_contribution

    e_mic, f_mic = mic_contribution(_PosStub(system))
    e_pme, f_pme = pme_contribution(_PosStub(system))

    assert np.isfinite(e_mic)
    assert np.isfinite(e_pme)
    assert np.all(np.isfinite(f_pme))
    # different electrostatics treatment (MIC truncated switch vs. full
    # periodic Ewald/PME) must give a different energy -- if these matched,
    # the lr_solver kwarg would not actually be reaching the reference call.
    assert e_pme != pytest.approx(e_mic, rel=1e-6)
