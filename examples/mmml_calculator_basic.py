#!/usr/bin/env python3

import numpy as np

def main() -> int:
    try:
        from mmml.pycharmmInterface.mmml_calculator import setup_calculator, CutoffParameters
    except Exception as exc:
        print("This example requires optional dependencies (PyCHARMM, JAX, e3x).\n"
              f"Failed to import calculator: {exc}")
        return 0

    # Simple toy system: two monomers with ATOMS_PER_MONOMER atoms each
    ATOMS_PER_MONOMER = 5
    N_MONOMERS = 2
    MAX_ATOMS = 100

    # Create fake atomic numbers and positions for demonstration
    Z = np.zeros((N_MONOMERS * ATOMS_PER_MONOMER,), dtype=np.int32)
    Z[:ATOMS_PER_MONOMER] = 6  # carbon-like
    Z[ATOMS_PER_MONOMER:2*ATOMS_PER_MONOMER] = 1  # hydrogen-like

    R = np.zeros((N_MONOMERS * ATOMS_PER_MONOMER, 3), dtype=np.float32)
    R[0:ATOMS_PER_MONOMER, 0] = np.linspace(0.0, 2.0, ATOMS_PER_MONOMER)
    R[ATOMS_PER_MONOMER:2*ATOMS_PER_MONOMER, 0] = np.linspace(5.0, 7.0, ATOMS_PER_MONOMER)

    # Build calculator factory
    get_calc_factory = setup_calculator(
        ATOMS_PER_MONOMER=ATOMS_PER_MONOMER,
        N_MONOMERS=N_MONOMERS,
        ml_cutoff_distance=2.0,
        mm_switch_on=5.0,
        mm_cutoff=1.0,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=None,
        MAX_ATOMS_PER_SYSTEM=MAX_ATOMS,
    )

    cutoff = CutoffParameters(ml_cutoff=2.0, mm_switch_on=5.0, mm_cutoff=1.0)

    # Create calculator and core function
    calculator, core_fn = get_calc_factory(
        atomic_numbers=R[:, 0].astype(np.int32),  # placeholder, not used directly by factory
        atomic_positions=R,
        n_monomers=N_MONOMERS,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=False,
        debug=False,
    )

    # Use core function directly
    try:
        out = core_fn(
            positions=R,
            atomic_numbers=Z,
            n_monomers=N_MONOMERS,
            cutoff_params=cutoff,
            doML=True,
            doMM=True,
            doML_dimer=True,
            debug=False,
        )
        print("Energy:", np.asarray(out.energy))
        print("Forces shape:", np.asarray(out.forces).shape)
    except Exception as exc:
        print("Computation failed (likely missing runtime deps).", exc)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


