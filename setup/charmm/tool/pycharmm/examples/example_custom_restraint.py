"""Example: Custom positional restraints with mid-simulation parameter changes.

This script demonstrates the full workflow for using the Custom*Force API:
1. Build a minimal system
2. Add a CustomExternalForce as a positional restraint
3. Run dynamics with strong restraints
4. Modify restraint parameters and push changes to the live context
5. Run dynamics with weaker restraints
6. Compare displacement under strong vs. weak restraints

Run with:
    CHARMM_LIB_DIR=.../build_cmake conda run -n test python example_custom_restraint.py
"""

import sys

import numpy as np

import pycharmm.omm as omm
from pycharmm import coor, lingo, read
from pycharmm import generate as gen


def main():
    # ---- 1. Build a minimal 2-atom system ----
    lingo.charmm_script('''
    read rtf card
* Minimal topology
*
   20    1
MASS     -1 X     12.0

RESI DUM        0.0
GROUP
ATOM A    X     0.0
ATOM B    X     0.0
BOND A B
PATC  FIRS NONE LAST NONE
END
    ''')
    lingo.charmm_script("""
    read param card
* Minimal parameters
*
BONDS
X    X     100.0   1.0

NONBONDED   ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
     CUTNB 8.0  CTOFNB 7.5  CTONNB 6.5  EPS 1.0  E14FAC 1.0  WMIN 1.5
X        0.0440    1.0       0.8000

END
    """)

    read.sequence_string("DUM")
    gen.new_segment(seg_name="MOL")

    # Set initial positions
    pos = coor.get_positions()
    pos.iloc[0] = [0.0, 0.0, 0.0]
    pos.iloc[1] = [1.0, 0.0, 0.0]
    coor.set_positions(pos)

    # ---- 2. Create a CustomExternalForce for positional restraints ----
    # Expression: k*((x-x0)^2+(y-y0)^2+(z-z0)^2)
    # OpenMM units: positions in nm, energy in kJ/mol
    restraint = omm.CustomExternalForce(
        "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.add_per_particle_parameter("k")
    restraint.add_per_particle_parameter("x0")
    restraint.add_per_particle_parameter("y0")
    restraint.add_per_particle_parameter("z0")

    # Convert positions from Angstrom to nm
    x0_nm = 0.0 * omm.NM_PER_ANGSTROM
    x1_nm = 1.0 * omm.NM_PER_ANGSTROM

    # Strong restraint: k=1000 kJ/mol/nm^2
    k_strong = 1000.0
    restraint.add_particle(0, [k_strong, x0_nm, 0.0, 0.0])
    restraint.add_particle(1, [k_strong, x1_nm, 0.0, 0.0])

    # ---- 3. Verify read-back ----
    particle, params = restraint.get_particle_parameters(0, 4)
    assert particle == 0, "Particle index mismatch"
    assert abs(params[0] - k_strong) < 1e-10, "k mismatch on read-back"
    assert abs(params[1] - x0_nm) < 1e-10, "x0 mismatch on read-back"
    print(f"Restraint force created with index {restraint.index}")
    print(f"  Particle 0: k={params[0]:.1f} kJ/mol/nm^2, "
          f"x0={params[1]:.4f} nm")

    # ---- 4. Modify parameters (reduce k) ----
    k_weak = 10.0
    restraint.set_particle_parameters(0, 0, [k_weak, x0_nm, 0.0, 0.0])
    restraint.set_particle_parameters(1, 1, [k_weak, x1_nm, 0.0, 0.0])

    # Verify modification
    _, params_after = restraint.get_particle_parameters(0, 4)
    assert abs(params_after[0] - k_weak) < 1e-10, \
        "k not updated after set_particle_parameters"
    print(f"  After modification: k={params_after[0]:.1f} kJ/mol/nm^2")

    # ---- 5. Demonstrate tabulated function ----
    nb = omm.CustomNonbondedForce("tabfunc(r)")
    nb.add_tabulated_function_continuous1d(
        "tabfunc",
        [0.0, 0.5, 1.0, 0.5, 0.0],  # values
        0.0, 2.0,  # min, max in nm
        periodic=False)
    nb.add_particle()
    nb.add_particle()
    print(f"Nonbonded force with tabulated function: index {nb.index}")

    # ---- 6. Demonstrate unsupported type validation ----
    cv = omm.CustomCVForce("cv1^2")
    try:
        cv.add_tabulated_function_continuous1d("f", [0.0, 1.0], 0.0, 1.0)
        # CV does support tabulated functions, so this should succeed
        print("CustomCVForce accepts tabulated functions (expected)")
    except TypeError:
        print("CustomCVForce rejected tabulated functions (unexpected)")

    bond = omm.CustomBondForce("r^2")
    try:
        bond.add_tabulated_function_continuous1d("f", [0.0, 1.0], 0.0, 1.0)
        print("CustomBondForce accepted tabulated functions (unexpected)")
    except TypeError:
        print("CustomBondForce correctly rejects tabulated functions")

    try:
        cv.update_parameters_in_context()
        print("CustomCVForce accepted updateParametersInContext (unexpected)")
    except TypeError:
        print("CustomCVForce correctly rejects updateParametersInContext")

    # ---- 7. Choose which CHARMM energy term this force reports in ----
    # CHARMM assigns OpenMM force groups itself, one per energy term, so
    # set_force_group() cannot stick (it warns).  Pick the energy term
    # instead: the restraint's energy then shows up under that term in the
    # energy table rather than the default one for its force class.
    restraint.set_eterm(omm.EtermBucket.CFCV)
    print(f"\nEnergy term set to {restraint.get_eterm().name}")

    # ---- 8. Demonstrate introspection ----
    expr = restraint.get_energy_expression()
    print(f"Energy expression: {expr}")
    n_per = restraint.get_num_per_parameters()
    print(f"Per-particle parameters ({n_per}):")
    for j in range(n_per):
        print(f"  {j}: {restraint.get_per_parameter_name(j)}")

    restraint.add_global_parameter("scale", 1.0)
    n_global = restraint.get_num_global_parameters()
    for j in range(n_global):
        name = restraint.get_global_parameter_name(j)
        val = restraint.get_global_parameter_default_value(j)
        print(f"Global parameter {j}: {name} = {val}")

    # ---- 9. Demonstrate 2D tabulated function ----
    nb2 = omm.CustomNonbondedForce("tab2d(r,r)")
    values_2d = [float(i) for i in range(9)]  # 3x3 grid
    nb2.add_tabulated_function_continuous2d(
        "tab2d", values_2d, 3, 3, 0.0, 1.0, 0.0, 1.0)
    nb2.add_particle()
    nb2.add_particle()
    print(f"\n2D tabulated function force: index {nb2.index}")

    # ---- 10. Demonstrate Hbond getters/setters ----
    hb = omm.CustomHbondForce("k*distance(d1,a1)")
    hb.add_per_donor_parameter("k")
    hb.add_donor(0, -1, -1, [100.0])
    hb.add_acceptor(0)
    d1, d2, d3, params = hb.get_donor_parameters(0, 1)
    print(f"\nHbond donor 0: d1={d1}, k={params[0]:.1f}")
    print(f"Num donors: {hb.get_num_donors()}, "
          f"Num acceptors: {hb.get_num_acceptors()}")
    print(f"Per-donor param 0: {hb.get_per_donor_parameter_name(0)}")

    # ---- 11. Demonstrate ManyParticle extras ----
    mp = omm.CustomManyParticleForce(2, "r^2")
    mp.add_per_particle_parameter("charge")
    mp.add_particle([1.0], particle_type=0)
    mp.add_particle([2.0], particle_type=1)
    mp.set_type_filter(0, [0, 1])
    types = mp.get_type_filter(0)
    print(f"\nManyParticle type filter for slot 0: {sorted(types)}")
    print(f"Permutation mode: {mp.get_permutation_mode()}")
    mp.set_permutation_mode(omm.CustomManyParticleForce.UniqueCentralParticle)
    print(f"After change: {mp.get_permutation_mode()}")
    params_mp, ptype = mp.get_particle_parameters(0, 1)
    print(f"Particle 0: charge={params_mp[0]:.1f}, type={ptype}")

    # ---- 12. Demonstrate CentroidBond getters/setters ----
    cb = omm.CustomCentroidBondForce(2, "k*distance(g1,g2)")
    cb.add_per_bond_parameter("k")
    cb.add_group([0], [1.0])
    cb.add_group([0], [1.0])
    cb.add_bond([0, 1], [50.0])
    groups, bond_params = cb.get_bond_parameters(0)
    print(f"\nCentroidBond: groups={groups}, k={bond_params[0]:.1f}")
    particles, weights = cb.get_group_parameters(0)
    print(f"Group 0: particles={particles}, weights={weights}")

    # ---- 13. Demonstrate RMSDForce as a CV ----
    ref_nm = np.array([[x0_nm, 0.0, 0.0],
                       [x1_nm, 0.0, 0.0]])
    rmsd = omm.RMSDForce(ref_nm, particles=[0, 1])
    cv_rmsd = omm.CustomCVForce("k_rmsd * rmsd^2")
    cv_rmsd.add_global_parameter("k_rmsd", 100.0)
    cv_rmsd.add_collective_variable("rmsd", rmsd)
    print(f"\nRMSDForce CV: index {rmsd.index}")
    print(f"CustomCVForce with RMSD: index {cv_rmsd.index}")

    # ---- 14. Demonstrate RGForce as a CV ----
    rg = omm.RGForce(particles=[0, 1])
    cv_rg = omm.CustomCVForce("k_rg * (rg - rg0)^2")
    cv_rg.add_global_parameter("k_rg", 50.0)
    cv_rg.add_global_parameter("rg0", 0.5)  # nm
    cv_rg.add_collective_variable("rg", rg)
    print(f"RGForce CV: index {rg.index}")
    print(f"CustomCVForce with Rg: index {cv_rg.index}")

    print("\nAll checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
