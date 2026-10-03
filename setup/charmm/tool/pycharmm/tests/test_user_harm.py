"""User-defined harmonic Python energy function called per-step.

Migrated from a legacy procedural script. Skips when CHARMM_DATA_DIR
is unset.
"""

import numpy as np

from pycharmm import (
    EnergyFunc,
    SelectAtoms,
    charmm_script,
    cons_harm,
    coor,
    energy,
    lingo,
    minimize,
)


def test_user_harmonic_function(alanine_dipeptide_with_nbonds, tmp_path):
    """Body of original script, wrapped as a pytest test."""
    coor.show()

    # Impose harmonic restraints via external python function,
    # check against harmonic restraints
    # Set current build coordinates as reference coordinates
    xref = coor.get_main()

    # Note the variables x_pos, y_pos, z_pos, dx, dy and dz are
    # C-type pointers and thus need some indexing
    def harm(natoms, x_pos, y_pos, z_pos, dx, dy, dz):
        eharm = np.sum(
            np.power((x_pos[:natoms] - xref.x), 2)
            + np.power((y_pos[:natoms] - xref.y), 2)
            + np.power((z_pos[:natoms] - xref.z), 2)
        )

        for i in range(natoms):
            dx[i] += 2.0 * (x_pos[i] - xref.x[i])
            dy[i] += 2.0 * (y_pos[i] - xref.y[i])
            dz[i] += 2.0 * (z_pos[i] - xref.z[i])

        return eharm

    energy.show()
    minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)
    energy.show()

    # Test the usere harmonic restraint
    charmm_script("skipe incl all excl harm user")
    e_func = EnergyFunc(harm)
    energy.show()
    grms_harm = lingo.get_energy_value("GRMS")
    e_harm = lingo.get_energy_value("ENER")

    # Now get restraint energy with conventional CHARMM harmonic restraints
    # Turn off usere function
    e_func.unset_func()
    coor.set_comparison(xref)
    cons_harm.setup_absolute(force_const=1.0, q_mass=False, comparison=True)
    energy.show()
    grms_cons = lingo.get_energy_value("GRMS")
    e_cons = lingo.get_energy_value("ENER")
    cons_harm.turn_off()

    tol = 1e-5
    assert np.abs(e_harm - e_cons) <= tol, (
        f"User-func harmonic energy disagrees with cons_harm: "
        f"E_harm = {e_harm}, E_cons = {e_cons}, "
        f"|delta| = {abs(e_harm - e_cons)} > tol = {tol}"
    )
    assert np.abs(grms_harm - grms_cons) <= tol, (
        f"User-func GRMS disagrees with cons_harm: "
        f"GRMS_harm = {grms_harm}, GRMS_cons = {grms_cons}, "
        f"|delta| = {abs(grms_harm - grms_cons)} > tol = {tol}"
    )

    # Now test a selection of atoms
    rcs = SelectAtoms().by_res_and_type("ADP", "1", "CY N CA C NT")
    # and make an natoms boolean list from this
    rcs_bool = list(rcs)
    print("Number of selected atoms", np.sum(rcs_bool))
    cons_harm.setup_absolute(force_const=10.0, q_mass=False, selection=rcs, comparison=True)
    energy.show()
    grms_cons = lingo.get_energy_value("GRMS")
    e_cons = lingo.get_energy_value("ENER")
    cons_harm.turn_off()

    # Note the variables x_pos, y_pos, z_pos, dx, dy and dz are
    # C-type pointers and thus need some indexing
    def selharm(natoms, x_pos, y_pos, z_pos, dx, dy, dz):
        eharm = 0.0
        for i, b in enumerate(rcs_bool):
            if b:
                d = x_pos[i] - xref.x[i]
                eharm += 10.0 * d * d
                dx[i] += 2.0 * 10.0 * d
                d = y_pos[i] - xref.y[i]
                eharm += 10.0 * d * d
                dy[i] += 2.0 * 10.0 * d
                d = z_pos[i] - xref.z[i]
                eharm += 10.0 * d * d
                dz[i] += 2.0 * 10.0 * d
        return eharm

    e_func.set_func(selharm)
    energy.show()
    grms_harm = lingo.get_energy_value("GRMS")
    e_harm = lingo.get_energy_value("ENER")

    assert np.abs(e_harm - e_cons) <= tol, (
        f"User-func selharm energy disagrees with cons_harm (selected): "
        f"E_harm = {e_harm}, E_cons = {e_cons}, "
        f"|delta| = {abs(e_harm - e_cons)} > tol = {tol}"
    )
    assert np.abs(grms_harm - grms_cons) <= tol, (
        f"User-func selharm GRMS disagrees with cons_harm (selected): "
        f"GRMS_harm = {grms_harm}, GRMS_cons = {grms_cons}, "
        f"|delta| = {abs(grms_harm - grms_cons)} > tol = {tol}"
    )
