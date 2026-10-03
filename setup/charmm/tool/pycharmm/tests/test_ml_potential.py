"""ML potential (TorchMD-net or similar) integration workflow.

Migrated from a legacy procedural script.
"""

import os

import pytest

# A leftover torch directory under site-packages with no __init__.py is a
# namespace package, so `import torch' succeeds while torch.nn does not
# exist.  Ask for the submodule, which such a directory cannot provide.
pytest.importorskip("torch.nn")
pytest.importorskip("torch")
pytest.importorskip("openmm")

# Test script for the MLPot module in PyCHARMM.
# Here, a PaiNN model trained of ammonia (NH3) reference data is used to
# simulate ammonia in a TIP3P water box. This is not a chemical usefull setup
# as it is not able to predict the proton transfer reaction from water to
# ammonia.

# Basics
import numpy as np

# ASE (optional dependency)


def test_ml_potential_workflow(tmp_path):
    """Module-level body of the original script, wrapped as a pytest test."""
    try:
        from ase import io
    except ModuleNotFoundError as e:
        pytest.skip(f"missing optional dependency: {e.name}")

    # PyCHARMM
    import pycharmm
    import pycharmm.cons_fix as cons_fix
    import pycharmm.crystal as crystal
    import pycharmm.energy as energy
    import pycharmm.lingo as stream
    import pycharmm.minimize as minimize
    import pycharmm.read as read
    import pycharmm.settings as settings
    import pycharmm.write as write

    # Asparagus (optional dependency)
    try:
        from asparagus import Asparagus
    except ModuleNotFoundError as e:
        pytest.skip(f"missing optional dependency: {e.name}")

    # Step 0: Load parameter files
    # -----------------------------------------------------------

    settings.set_bomb_level(-1)
    settings.set_warn_level(-1)

    # Per-test scratch directory for any file this test writes.
    # `data/...` paths in this file are reads (RTF/PRM/PDB/etc. that
    # ship in tests/data); writes go under `scratch/` so we don't
    # pollute the shared test data directory.
    scratch = tmp_path
    (scratch / "charmm_data").mkdir(exist_ok=True)

    read.rtf("data/top_all36_cgenff.rtf")
    read.prm("data/par_all36_cgenff.prm", flex=True)
    stream.charmm_script("stream data/toppar_water_ions.str")

    settings.set_bomb_level(-2)

    # Step 1: Generate System
    # -----------------------------------------------------------

    # Should ammonia be solvated in water:
    solvation = True

    # Read ML system
    add_ammonia = """
        open read card unit 10 name data/ammonia.pdb
        read sequence pdb unit 10
        generate AMM1 setup warn first none last none

        open read card unit 10 name data/ammonia.pdb
        read coor pdb  unit 10 resid
        """
    stream.charmm_script(add_ammonia)

    # Read solvent
    if solvation:
        # Add TIP3P water residues
        read.psf_card("data/water_cube.psf", append=True)
        read.coor_card("data/water_cube.crd", append=True)

        # Delete overlapping water molecules
        remove_water = """
            delete atom select ( .byres. ( (segid AMM1 .around. 2.0 ) -
                .and. (segid TIP3 .and. type OH2 ))) end
        """
        stream.charmm_script(remove_water)

    write.coor_pdb(str(scratch / "ammonia_water.pdb"), title="Ammonia solvated")
    write.coor_card(str(scratch / "ammonia_water.crd"), title="Ammonia solvated")
    write.psf_card(str(scratch / "ammonia_water.psf"), title="Ammonia solvated")

    # ASE atoms object (just to get atomic numbers)
    ase_ammonia = io.read("data/ammonia.pdb", format="proteindatabank")

    # Step 2: CHARMM Setup
    # -----------------------------------------------------------

    # Non-bonding parameter
    dict_nbonds = {
        "atom": True,
        "vdw": True,
        "vswitch": True,
        "cutnb": 14,
        "ctofnb": 12,
        "ctonnb": 10,
        "cutim": 14,
        "lrc": True,
        "inbfrq": -1,
        "imgfrq": -1,
    }
    nbond = pycharmm.NonBondedScript(**dict_nbonds)
    nbond.run()

    if solvation:
        # PBC box
        crystal.define_cubic(length=30.0)
        crystal.build(cutoff=14.0)

        stream.charmm_script("image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end")

        # H-bonds constraint
        # shake.on(bonh=True, tol=1e-7)
        stream.charmm_script("shake bonh para sele resname TIP3 end")

    else:
        # Default Setup
        pass

    # Energy
    energy.show()

    # Step 3: Asparagus Setup
    # -----------------------------------------------------------

    # Load Asparagus model
    ml_model = Asparagus(config="data/model_nh3/config.json")

    # Get atomic number from ASE atoms object
    ml_Z = ase_ammonia.get_atomic_numbers()

    # Prepare PhysNet input parameter
    ml_selection = pycharmm.SelectAtoms(seg_id="AMM1")

    # Initialize the PhysNet calculator. Construction registers the
    # ML potential into CHARMM; we keep a named reference so the GC
    # doesn't drop it before the energy/dynamics calls below.
    calc = pycharmm.MLpot(
        ml_model,
        ml_Z,
        ml_selection,
        ml_charge=0,
        ml_fq=True,
    )
    assert calc is not None

    # Custom energy
    energy.show()

    # Step 4: Minimization
    # -----------------------------------------------------------

    file_mini_pdb = str(scratch / "charmm_data" / "mini_ammonia.pdb")
    file_mini_crd = str(scratch / "charmm_data" / "mini_ammonia.crd")

    if not os.path.exists(file_mini_crd):
        # Reduce bomlev due to double constraint
        settings.set_bomb_level(-2)

        # Fix ML atoms
        cons_fix.setup(pycharmm.SelectAtoms(seg_id="AMM1"))

        # Optimization with PhysNet parameter
        minimize.run_sd(**{"nstep": 100, "nprint": 10, "tolenr": 1e-5, "tolgrd": 1e-5})

        # Unfix ML atoms
        cons_fix.turn_off()

        # Reset bomlev
        settings.set_bomb_level(-1)

        # Optimization with PhysNet parameter
        minimize.run_sd(**{"nstep": 100, "nprint": 10, "tolenr": 1e-5, "tolgrd": 1e-5})

        # Verify minimization converged sensibly.
        e_mini = energy.get_total()
        grms_mini = energy.get_grms()
        assert np.isfinite(e_mini), f"ML potential minimization energy not finite: {e_mini}"
        # tolgrd was 1e-5; allow some slop for the SD steps to satisfy.
        assert grms_mini < 1.0, (
            f"ML potential minimization did not converge: GRMS = {grms_mini} (expected < 1.0)"
        )

        # Write pdb file
        write.coor_pdb(file_mini_pdb, title="Mini SD")
        write.coor_card(file_mini_crd, title="Mini SD")
        assert os.path.isfile(file_mini_pdb), f"Minimized PDB not written: {file_mini_pdb}"
        assert os.path.isfile(file_mini_crd), f"Minimized CRD not written: {file_mini_crd}"

    else:
        # Read optimized coordinates - Do not read from pdb files
        # as it yield a weird non-bonding atom pair bug where ML atoms
        # are not excluded from non-bonding interaction.
        read.coor_card(file_mini_crd)

    # Minimized custom energy
    energy.show()

    # Step 5: Heating - CHARMM, PhysNet
    # -----------------------------------------------------------

    if True:
        timestep = 0.00025  # 0.25 fs
        nsteps = 1.0 * 1.0 / timestep  # 1 ps
        nsavc = 0.100 * 1.0 / timestep  # every 100 fs
        temp = 300.0

        res_file = pycharmm.CharmmFile(
            file_name=str(scratch / "heat.res"), file_unit=2, formatted=True, read_only=False
        )
        dcd_file = pycharmm.CharmmFile(
            file_name=str(scratch / "heat.dcd"), file_unit=1, formatted=False, read_only=False
        )

        # Run some dynamics
        dynamics_dict = {
            "verlet": True,
            "new": True,
            "start": True,
            "timestep": timestep,
            "nstep": nsteps,
            "nsavc": nsavc,
            "inbfrq": -1,
            "ihbfrq": 50,
            "ilbfrq": 50,
            "imgfrq": 50,
            "ixtfrq": 1000,
            "iunwri": res_file.file_unit,
            "iuncrd": dcd_file.file_unit,
            "nprint": 100,  # Frequency to write to output
            "iprfrq": 500,  # Frequency to calculate averages
            "isvfrq": 1000,  # Frequency to save restart file
            "ntrfrq": 1000,
            "ihtfrq": 200,
            "ieqfrq": 1000,
            "firstt": temp / 2.0,
            "finalt": temp,
            "tbath": temp,
            "echeck": -1,
        }

        dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
        dyn_heat.run()

        res_file.close()
        dcd_file.close()

    # Step 6: NVE - CHARMM, PhysNet
    # -----------------------------------------------------------

    if True:
        timestep = 0.00025  # 0.25 fs
        nsteps = 1.0 * 1.0 / timestep  # 1 ps
        nsavc = 0.01 * 1.0 / timestep  # every 10 fs

        str_file = pycharmm.CharmmFile(
            file_name=str(scratch / "heat.res"), file_unit=3, formatted=True, read_only=False
        )
        res_file = pycharmm.CharmmFile(
            file_name=str(scratch / "nve.res"), file_unit=2, formatted=True, read_only=False
        )
        dcd_file = pycharmm.CharmmFile(
            file_name=str(scratch / "nve.dcd"), file_unit=1, formatted=False, read_only=False
        )

        # Run some dynamics
        dynamics_dict = {
            "verlet": True,
            "new": False,
            "start": False,
            "restart": True,
            "timestep": timestep,
            "nstep": nsteps,
            "nsavc": nsavc,
            "inbfrq": -1,
            "ihbfrq": 50,
            "ilbfrq": 50,
            "imgfrq": 50,
            "ixtfrq": 1000,
            "iunrea": str_file.file_unit,
            "iunwri": res_file.file_unit,
            "iuncrd": dcd_file.file_unit,
            "nprint": 10,  # Frequency to write to output
            "iprfrq": 500,  # Frequency to calculate averages
            "isvfrq": 1000,  # Frequency to save restart file
            "ntrfrq": 0,
            "ihtfrq": 0,
            "ieqfrq": 0,
            "echeck": -1,
        }

        dyn_nve = pycharmm.DynamicsScript(**dynamics_dict)
        dyn_nve.run()

        str_file.close()
        res_file.close()
        dcd_file.close()
