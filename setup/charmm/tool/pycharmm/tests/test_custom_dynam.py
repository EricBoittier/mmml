"""Custom dynamics callback invoked per integration step.

Migrated from a legacy procedural script. Skips when CHARMM_DATA_DIR
is unset.
"""

import numpy as np

from pycharmm import (
    CustomDynam,
    DynamicsScript,
    EnergyFunc,
    SelectAtoms,
    charmm_script,
    cons_harm,
    coor,
    energy,
    gen,
    ic,
    lingo,
    minimize,
    nbonds,
    read,
    settings,
)


def test_custom_dynamics_callback(tmp_path):
    """Body of original script, wrapped as a pytest test."""

    # Set-up blocked alanne residue in CHARMM
    rtf_fn = "data/top_all36_prot.rtf"
    read.rtf(rtf_fn)

    prm_fn = "data/par_all36_prot.prm"
    read.prm(prm_fn, flex=True)

    # begin data/toppar_water_ions.str
    read.rtf("data/water_ions.rtf", append=True)
    read.prm("data/water_ions.prm", append=True, flex=True)

    old_warn_level = settings.set_warn_level(-1)
    old_bomb_level = settings.set_bomb_level(-1)

    read.prm("data/sodium_oxygen_nbfixes.prm", append=True, flex=True)

    settings.set_warn_level(old_warn_level)
    settings.set_bomb_level(old_bomb_level)
    # end data/toppar_water_ions.str

    read.sequence_string("ALA")

    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)

    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()

    coor.orient()
    coor.show()

    nbonds.configure(
        cutnb=18.0,
        ctonnb=15.0,
        ctofnb=13.0,
        eps=1.0,
        cdie=True,
        atom=True,
        fswitch=True,
        vatom=True,
        vfswitch=True,
    )

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
    print("Number of selected atoms", rcs.get_n_selected())
    cons_harm.setup_absolute(force_const=10.0, q_mass=False, selection=rcs, comparison=True)
    energy.show()
    grms_cons = lingo.get_energy_value("GRMS")
    e_cons = lingo.get_energy_value("ENER")
    cons_harm.turn_off()

    rcs_indexes = rcs.get_atom_indexes()

    # Note the variables x_pos, y_pos, z_pos, dx, dy and dz are
    # C-type pointers and thus need some indexing
    def selharm(natoms, x_pos, y_pos, z_pos, dx, dy, dz):
        eharm = 0.0
        for i in rcs_indexes:
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

    def my_dyn_func(
        current_step, natoms, vx, vy, vz, x_new, y_new, z_new, x_old, y_old, z_old, x, y, z
    ):
        print("DEBUG inside my_dyn_func ", natoms, " step # ", current_step - 1)
        for i in range(natoms):
            print(i, vx[i], vy[i], vz[i])
        for i in range(natoms):
            print(i, x_new[i], y_new[i], z_new[i])
        return 39.5

    # CustomDynam construction registers the callback into CHARMM.
    # Hold a named reference so the binding stays alive for the
    # duration of the dynamics run below.
    dyn_func = CustomDynam(my_dyn_func)
    assert dyn_func is not None

    vanilla_dyn = DynamicsScript()
    vanilla_dyn.run()
