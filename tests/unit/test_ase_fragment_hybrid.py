"""Unit tests for calculator-neutral fragment ML/MM (dummy ASE, no torch)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from mmml.interfaces.calculators.ase_fragment_hybrid import (
    AseFragmentHybridCalculator,
    evaluate_fragment_hybrid,
    evaluate_whole_system,
    numpy_ml_switch_scale,
    wrap_dimer_monomer_b_numpy,
)
from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
)


class PairwiseDistanceCalculator(Calculator):
    """E = sum_{i<j} |r_j - r_i|; analytic forces. Isolated atoms have E=0."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Atoms object is required")
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        n = pos.shape[0]
        energy = 0.0
        forces = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[j] - pos[i]
                r = float(np.linalg.norm(delta))
                energy += r
                if r > 1.0e-12:
                    rhat = delta / r
                    forces[i] += rhat
                    forces[j] -= rhat
        self.results = {"energy": float(energy), "forces": forces}


class ConstantEnergyCalculator(Calculator):
    """E = n_atoms eV, zero forces (degenerate interaction)."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        self.results = {
            "energy": float(n),
            "forces": np.zeros((n, 3), dtype=np.float64),
        }


def _two_atom_dimer(separation: float) -> tuple[np.ndarray, np.ndarray]:
    numbers = np.array([1, 1], dtype=int)
    positions = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=np.float64)
    return numbers, positions


def test_numpy_ml_switch_scale_matches_jax() -> None:
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale

    mm_on = DEFAULT_MM_SWITCH_ON
    width = DEFAULT_ML_SWITCH_WIDTH
    radii = np.linspace(0.0, mm_on + 2.0, 41)
    jax_vals = np.asarray(
        ml_switch_scale(
            jnp.asarray(radii),
            mm_switch_on=mm_on,
            ml_switch_width=width,
        )
    )
    numpy_vals = np.array(
        [numpy_ml_switch_scale(float(r), mm_switch_on=mm_on, ml_switch_width=width) for r in radii]
    )
    np.testing.assert_allclose(numpy_vals, jax_vals, atol=1e-12, rtol=1e-12)
    assert numpy_ml_switch_scale(0.0) == pytest.approx(1.0)
    assert numpy_ml_switch_scale(mm_on + 0.1) == pytest.approx(0.0)


def test_wrap_dimer_monomer_b_numpy_matches_jax() -> None:
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.pbc_utils_jax import wrap_dimer_monomer_b

    cell = np.diag([30.0, 30.0, 30.0])
    pos_a = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    pos_b = np.array([[29.0, 0.0, 0.0], [29.1, 0.0, 0.0]], dtype=np.float64)
    wrapped_b = wrap_dimer_monomer_b_numpy(pos_a, pos_b, cell)
    pos = np.vstack([pos_a, pos_b])
    jax_wrapped = np.asarray(wrap_dimer_monomer_b(jnp.asarray(pos), 2, 2, jnp.asarray(cell)))
    np.testing.assert_allclose(wrapped_b, jax_wrapped[2:], atol=1e-10)
    sep = float(np.linalg.norm(wrapped_b.mean(axis=0) - pos_a.mean(axis=0)))
    assert sep == pytest.approx(1.0, abs=5e-3)


def test_evaluate_fragment_hybrid_monomer_sum_plus_switched_interaction() -> None:
    calc = PairwiseDistanceCalculator()
    # Well inside ML region: s=1, E = 0 + 0 + 1*(r - 0 - 0) = r
    r_in = 3.0
    numbers, positions = _two_atom_dimer(r_in)
    inside = evaluate_fragment_hybrid(
        calc, numbers, positions, [1, 1], do_ml=True, do_ml_dimer=True
    )
    assert inside.energy_ev == pytest.approx(r_in)
    assert inside.n_monomers_evaluated == 2
    assert inside.n_dimers_evaluated == 1
    np.testing.assert_allclose(inside.forces_ev_per_angstrom[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(inside.forces_ev_per_angstrom[1], [-1.0, 0.0, 0.0])

    r_out = DEFAULT_MM_SWITCH_ON + 1.0
    _, positions_out = _two_atom_dimer(r_out)
    outside = evaluate_fragment_hybrid(
        calc, numbers, positions_out, [1, 1], do_ml=True, do_ml_dimer=True
    )
    assert outside.energy_ev == pytest.approx(0.0)
    assert outside.n_dimers_evaluated == 0
    np.testing.assert_allclose(outside.forces_ev_per_angstrom, 0.0)


def test_fragment_hybrid_switch_forces_match_finite_difference() -> None:
    calc = PairwiseDistanceCalculator()
    r0 = DEFAULT_MM_SWITCH_ON - 0.5 * DEFAULT_ML_SWITCH_WIDTH
    numbers, positions = _two_atom_dimer(r0)
    result = evaluate_fragment_hybrid(calc, numbers, positions, [1, 1])
    delta = 1.0e-5
    e_plus = evaluate_fragment_hybrid(
        calc, numbers, _two_atom_dimer(r0 + delta)[1], [1, 1]
    ).energy_ev
    e_minus = evaluate_fragment_hybrid(
        calc, numbers, _two_atom_dimer(r0 - delta)[1], [1, 1]
    ).energy_ev
    # Move atom B along +x: dE/dx_B ≈ -F_Bx
    numeric = (e_plus - e_minus) / (2.0 * delta)
    assert result.forces_ev_per_angstrom[1, 0] == pytest.approx(-numeric, rel=1e-4, abs=1e-5)


def test_evaluate_whole_system_is_one_call() -> None:
    calc = ConstantEnergyCalculator()
    numbers = np.array([1, 8, 1], dtype=int)
    positions = np.zeros((3, 3), dtype=np.float64)
    result = evaluate_whole_system(calc, numbers, positions)
    assert result.eval_mode == "whole_system"
    assert result.energy_ev == pytest.approx(3.0)
    assert result.n_monomers_evaluated == 1
    assert result.n_dimers_evaluated == 0


def test_ase_fragment_hybrid_calculator_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="eval_mode"):
        AseFragmentHybridCalculator(PairwiseDistanceCalculator, [1, 1], eval_mode="nope")


def test_ase_fragment_hybrid_calculator_fragments_property() -> None:
    from ase import Atoms

    calc = AseFragmentHybridCalculator(
        PairwiseDistanceCalculator, [1, 1], eval_mode="fragments"
    )
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    atoms.calc = calc
    assert atoms.get_potential_energy() == pytest.approx(3.0)
