"""E14 factor agreement between OpenMM and BLaDE backends.

Migrated from a legacy procedural script.
"""

import pytest

pytest.importorskip("openmm")

import numpy as np  # noqa: E402

import pycharmm  # noqa: E402
from pycharmm import (
    CharmmFile,
    charmm_script,
    crystal,
    dyn,
    energy,
    lingo,
    psf,
    read,
    settings,
)


# Requires CHARMM compiled with BLaDE (CUDA/NVIDIA only). On builds
# without BLaDE the test is auto-skipped by the requires_feature hook
# in conftest.py.
@pytest.mark.slow
@pytest.mark.requires_feature("BLADE")
def test_e14fac_openmm_vs_blade(tmp_path):
    """Module-level body of the original script, wrapped as a pytest test."""
    read.rtf('"data/top_all36_prot.rtf"')
    read.prm('"data/par_all36m_prot.prm"', flex=True)
    read.stream('"data/toppar_water_ions.str"')
    cbl = settings.set_bomb_level(-2)
    read.rtf('"lig_test/UNK_5E51CD.rtf"', append=True)
    read.prm('"lig_test/UNK_5E51CD.prm"', flex=True, append=True)

    # ## Here we set how the interactions will be computed
    # ## Choices are:
    # > ### all_geom = True <- use all geometric combination
    # > ### all_arit = True <- use all aritmatic combining rules
    # > ### lig_geom = True <- use geometric for ligand only, arithmatic for environment-environment and ligand environment
    # > ### lig_ligenv_geom = True <- use arithmatic for environment-environment and arithmatic for ligand-ligand and ligand-environment

    # In[3]:

    # Combining-rule mode: exactly one of these may be True at a
    # time. The default (all four False) corresponds to the
    # "lig_ligenv_geom" mode handled by the `else` branch below.
    all_geom = False
    all_arith = False
    lig_geom = False
    if all_geom:
        header = "ENVIRONMENT - GEOMETRIC: LIGAND - GEOMETRIC: LIGAND-ENVIRONMENT - GEOMETRIC"
        read.prm("lig_test/nbfix_c36_geom.prm", flex=True, append=True)
        read.prm("lig_test/nbfix_ligand_geom.prm", flex=True, append=True)
        read.prm("lig_test/nbfix_c36-ligand_geom.prm", flex=True, append=True)
    elif all_arith:
        header = "ENVIRONMENT - ARITHMATIC: LIGAND - ARITHMATIC: LIGAND-ENVIRONMENT - ARITHMATIC"
        read.prm("lig_test/nbfix_c36_arith.prm", flex=True, append=True)
        read.prm("lig_test/nbfix_ligand_arith.prm", flex=True, append=True)
        read.prm("lig_test/nbfix_c36-ligand_arith.prm", flex=True, append=True)
    elif lig_geom:
        header = "ENVIRONMENT - ARITHMATIC: LIGAND - GEOMETRIC: LIGAND-ENVIRONMENT - ARITHMATIC"
        read.prm("lig_test/nbfix_ligand_geom.prm", flex=True, append=True)
    else:
        header = "ENVIRONMENT - ARITHMATIC: LIGAND - GEOMETRIC: LIGAND-ENVIRONMENT - GEOMETRIC"
        # Use lig_ligenv geometric
        read.prm("lig_test/nbfix_ligand_geom.prm", flex=True, append=True)
        read.prm("lig_test/nbfix_c36-ligand_geom.prm", flex=True, append=True)

    # ## Read system in - see below to build solvated system

    # In[4]:

    read.psf_card("lig_test/lig-unk+wt00.psf")
    read.pdb("lig_test/lig-unk+wt00-min.pdb", resid=True)
    # setup crystal, images and nonbonded parameters
    crystal.define_cubic(23.11)
    crystal.build(11)
    pycharmm.NonBondedScript(
        cutnb=11,
        ctonnb=8,
        ctofnb=10,
        ewald=True,
        pmewald=True,
        kappa=0.32,
        fftx=30,
        ffty=30,
        fftz=30,
        order=4,
        e14fac=0.5,
    ).run()

    # # Test 1
    #
    # ## Check e14fac scaling methods - at present CHARMM energy functions only support single uniform e14fac but OpenMM and BLaDE should support e14fac as an NATOM array
    # ### Ensure CHARMM and OpenMM match for different uniform e14fac values

    # In[5]:

    ovl = settings.set_verbosity(0)
    owl = settings.set_warn_level(-4)
    settings.set_bomb_level(cbl)
    # Both should be default FF value of 0.5
    test_results = [True]
    charmm_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        pycharmm.NonBondedScript(e14fac=f"{e14}").run()
        energy.show()
        charmm_E[e14] = energy.get_total()

    OMM_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        pycharmm.NonBondedScript(e14fac=f"{e14}").run()
        charmm_script("energy omm")
        OMM_E[e14] = lingo.get_energy_value("ENER")

    # NOTE: This works when we call update. Where is BLaDE initialized?
    BLADE_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        pycharmm.NonBondedScript(e14fac=f"{e14}").run()
        charmm_script("energy blade")
        BLADE_E[e14] = lingo.get_energy_value("ENER")
    for e14 in [0.5, 1.0, 0.35]:
        test_results[-1] = test_results[-1] and abs(charmm_E[e14] - OMM_E[e14]) <= 0.05
        test_results[-1] = test_results[-1] and abs(charmm_E[e14] - BLADE_E[e14]) <= 0.05
        test_results[-1] = test_results[-1] and abs(BLADE_E[e14] - OMM_E[e14]) <= 0.05
        print(
            f"Energy differences: CHARMM {charmm_E[e14]:.4f}, OpenMM {OMM_E[e14]:.4f}, Difference {charmm_E[e14] - OMM_E[e14]:.4f}"
        )  # e14fac = 0.5
        print(
            f"Energy differences: CHARMM {charmm_E[e14]:.4f}, BLADE {BLADE_E[e14]:.4f}, Difference {charmm_E[e14] - BLADE_E[e14]:.4f}"
        )  # e14fac = 1.0
        print(
            f"Energy differences: BLADE {BLADE_E[e14]:.4f}, OpenMM {OMM_E[e14]:.4f}, Difference {BLADE_E[e14] - OMM_E[e14]:.4f}"
        )  # e14fac = 0.35for e14 in [0.5, 1.0, 0.35]:

    # ## This sections exercises BLaDE bug where call to update from energy command does not trigger BLaDE rebuild
    # ### Compare BLaDE uniform e14fac with OpenMM and CHARMM

    # # Test 2

    # In[6]:

    charmm_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        charmm_script(f"energy e14fac {e14}")
        charmm_E[e14] = energy.get_total()

        OMM_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        charmm_script(f"energy omm e14fac {e14}")
        OMM_E[e14] = lingo.get_energy_value("ENER")

    # NOTE: !!!!! calling nonbond options from energy command !!!!!
    #       !!!!! does not trigger BLaDE rebuild system       !!!!!
    BLADE_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        charmm_script(f"energy blade e14fac {e14}")
        BLADE_E[e14] = lingo.get_energy_value("ENER")

    test_results.append(True)
    for e14 in [0.5, 1.0, 0.35]:
        test_results[-1] = test_results[-1] and abs(charmm_E[e14] - OMM_E[e14]) <= 0.05
        test_results[-1] = test_results[-1] and abs(charmm_E[e14] - BLADE_E[e14]) <= 0.05
        test_results[-1] = test_results[-1] and abs(BLADE_E[e14] - OMM_E[e14]) <= 0.05
        print(
            f"Energy differences: CHARMM {charmm_E[e14]:.4f}, OpenMM {OMM_E[e14]:.4f}, Difference {charmm_E[e14] - OMM_E[e14]:.4f}"
        )  # e14fac = 0.5
        print(
            f"Energy differences: CHARMM {charmm_E[e14]:.4f}, BLADE {BLADE_E[e14]:.4f}, Difference {charmm_E[e14] - BLADE_E[e14]:.4f}"
        )  # e14fac = 1.0
        print(
            f"Energy differences: BLADE {BLADE_E[e14]:.4f}, OpenMM {OMM_E[e14]:.4f}, Difference {BLADE_E[e14] - OMM_E[e14]:.4f}"
        )  # e14fac = 0.35

    # # Test 3
    #
    # ### Check whether scalar array uniform e14fac works - should match across CHARMM, BLaDE and OpenMM

    # In[7]:

    OMM_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        charmm_script(f"scalar e14fac set {e14} select all end")
        charmm_script("energy omm")
        OMM_E[e14] = lingo.get_energy_value("ENER")
    BLADE_E = {}
    for e14 in [0.5, 1.0, 0.35]:
        charmm_script(f"scalar e14fac set {e14} select all end")
        charmm_script("energy BLADE")
        BLADE_E[e14] = lingo.get_energy_value("ENER")
    test_results.append(True)
    for e14 in [0.5, 1.0, 0.35]:
        test_results[-1] = test_results[-1] and abs(charmm_E[e14] - OMM_E[e14]) <= 0.05
        test_results[-1] = test_results[-1] and abs(charmm_E[e14] - BLADE_E[e14]) <= 0.05
        test_results[-1] = test_results[-1] and abs(BLADE_E[e14] - OMM_E[e14]) <= 0.05
        print(
            f"Energy differences: CHARMM {charmm_E[e14]:.4f}, OpenMM {OMM_E[e14]:.4f}, Difference {charmm_E[e14] - OMM_E[e14]:.4f}"
        )  # e14fac = 0.5
        print(
            f"Energy differences: CHARMM {charmm_E[e14]:.4f}, BLADE {BLADE_E[e14]:.4f}, Difference {charmm_E[e14] - BLADE_E[e14]:.4f}"
        )  # e14fac = 1.0
        print(
            f"Energy differences: BLADE {BLADE_E[e14]:.4f}, OpenMM {OMM_E[e14]:.4f}, Difference {BLADE_E[e14] - OMM_E[e14]:.4f}"
        )  # e14fac = 0.35

    # # Test 4
    #
    # ### Test whether partioned e14fac between ligand and environment matches between OpenMM and BLaDE

    # In[8]:

    OMM_E = {}
    for e14 in [(0.5, 1.0), (1.0, 0.5), (0.35, 0.5)]:
        charmm_script(f"scalar e14fac set {e14[0]} select segid lig end")
        charmm_script(f"scalar e14fac set {e14[1]} select segid wt00 end")
        charmm_script("energy omm")
        OMM_E[e14[0]] = lingo.get_energy_value("ENER")
    BLADE_E = {}
    for e14 in [(0.5, 1.0), (1.0, 0.5), (0.35, 0.5)]:
        charmm_script(f"scalar e14fac set {e14[0]} select segid lig end")
        charmm_script(f"scalar e14fac set {e14[1]} select segid wt00 end")
        charmm_script("energy blade")
        BLADE_E[e14[0]] = lingo.get_energy_value("ENER")
    test_results.append(True)
    for e14 in [(0.5, 1.0), (1.0, 0.5), (0.35, 0.5)]:
        test_results[-1] = test_results[-1] and abs(BLADE_E[e14[0]] - OMM_E[e14[0]]) <= 0.05
        print(
            f"Energy differences: BLADE {BLADE_E[e14[0]]:.4f}, OpenMM {OMM_E[e14[0]]:.4f}, Difference {BLADE_E[e14[0]] - OMM_E[e14[0]]:.4f}"
        )

    settings.set_verbosity(ovl)
    settings.set_warn_level(owl)

    # In[9]:

    test_type = [
        "Set e14fac in update",
        "Set e14fac in energy",
        "Set e14fac w/ scalar - uniform",
        "Set e14fac w/scalar - mixed",
    ]
    print(f"{header}")
    for i, r in enumerate(test_results):
        assert r, (
            f"E14 factor test '{test_type[i]}' failed: "
            "CHARMM/OpenMM/BLaDE energies disagreed by more than 0.05 "
            "kcal/mol. See test output for details."
        )

    # ## Test whether dynamics works under similar conditions
    # > * ### Energies should match energies from test above on OpenMM and BLaDE
    # > * ### Step 0 in dynamics should match energy from energy call preceding dynamics
    # > * ### Step 1 in dynamics should match energy from energy call following dynamics

    # In[10]:

    # Don't use SHAKE since internal water forces not computed in BLaDE
    # shake.on(bonh=True, fast=True, tol=1e-7)
    settings.set_verbosity(ovl)
    dyn.set_fbetas(np.full((psf.get_natom()), 1.0, dtype=float))
    # `res_file` is opened but the dynamics call below uses `iunwri=0`
    # (no restart write). Kept open so the unit number stays reserved
    # while the dynamics run; the file is empty on exit.
    res_file = CharmmFile(
        file_name=str(tmp_path / "test.res"), file_unit=2, formatted=True, read_only=False
    )
    assert res_file.file_unit == 2
    lam_file = CharmmFile(
        file_name=str(tmp_path / "test.lam"), file_unit=3, formatted=False, read_only=False
    )
    useomm = "gamma 2 prmc pref 1 iprsfrq 10"
    useblade = ""
    read.pdb("lig_test/lig-unk+wt00-min.pdb", resid=True)
    charmm_script(f"scalar e14fac set {0.5} select segid lig end")
    charmm_script(f"scalar e14fac set {1.0} select segid wt00 end")
    charmm_script("energy omm")
    pycharmm.DynamicsScript(
        leap=True,
        lang=True,
        start=True,
        nstep=1,
        timest=0.002,
        firstt=298.0,
        finalt=298.0,
        tbath=298.0,
        tstruc=298.0,
        teminc=0.0,
        twindh=0.0,
        twindl=0.0,
        iunwri=0,
        iunlam=lam_file.file_unit,
        inbfrq=-1,
        imgfrq=-1,
        iasors=0,
        iasvel=1,
        ichecw=0,
        iscale=0,
        iscvel=0,
        echeck=-1.0,
        nsavc=0,
        nsavv=0,
        nsavl=0,
        ntrfrq=0,
        isvfrq=1,
        iprfrq=2,
        nprint=1,
        ihtfrq=0,
        ieqfrq=0,
        ilbfrq=0,
        ihbfrq=0,
        blade=useblade,
        omm=useomm,
    ).run()

    charmm_script("energy omm")

    useomm = ""
    useblade = "prmc pref 1 iprs 100 prdv 100"
    read.pdb("lig_test/lig-unk+wt00-min.pdb", resid=True)
    charmm_script(f"scalar e14fac set {0.5} select segid lig end")
    charmm_script(f"scalar e14fac set {1.0} select segid wt00 end")
    charmm_script("energy blade")
    pycharmm.DynamicsScript(
        leap=True,
        lang=True,
        start=True,
        nstep=1,
        timest=0.002,
        firstt=298.0,
        finalt=298.0,
        tbath=298.0,
        tstruc=298.0,
        teminc=0.0,
        twindh=0.0,
        twindl=0.0,
        iunwri=0,
        iunlam=lam_file.file_unit,
        inbfrq=-1,
        imgfrq=-1,
        iasors=0,
        iasvel=1,
        ichecw=0,
        iscale=0,
        iscvel=0,
        echeck=-1.0,
        nsavc=0,
        nsavv=0,
        nsavl=0,
        ntrfrq=0,
        isvfrq=1,
        iprfrq=2,
        nprint=1,
        ihtfrq=0,
        ieqfrq=0,
        ilbfrq=0,
        ihbfrq=0,
        blade=useblade,
        omm=useomm,
    ).run()

    charmm_script("energy blade")

    # In[9]:

    test_type = [
        "Set e14fac in update",
        "Set e14fac in energy",
        "Set e14fac w/ scalar - uniform",
        "Set e14fac w/scalar - mixed",
    ]
    print(f"{header}")
    for i, r in enumerate(test_results):
        assert r, (
            f"E14 factor test '{test_type[i]}' failed: "
            "CHARMM/OpenMM/BLaDE energies disagreed by more than 0.05 "
            "kcal/mol. See test output for details."
        )
