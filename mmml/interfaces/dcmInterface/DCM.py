"""
Example: build mdcm from DCMNet H5 and run CHARMM DCM.

Usage:
    from mmml.interfaces.dcmInterface import dcmnet_to_mdcm, generate_dcm_xyz
    import h5py

    with h5py.File("charmm_ml_comparison.h5", "r") as f:
        R = f["R"][frame_idx]
        Z = f["Z"][frame_idx]
        charges = f["dcmnet_charges"][frame_idx]
        positions = f["dcmnet_charge_positions"][frame_idx]

    dcmnet_to_mdcm(R, Z, charges, positions, "MEOH", "meoh.mdcm")
    # Python-side dcm.xyz (for regression without CHARMM):
    from mmml.interfaces.dcmInterface.topology import get_frames_meoh_like
    from mmml.interfaces.dcmInterface.dcmnet_to_mdcm import dcmnet_to_mdcm
    from mmml.interfaces.dcmInterface.dcm_xyz import generate_dcm_xyz
    from mmml.interfaces.dcmInterface.convert import global_to_local
    from mmml.interfaces.dcmInterface.frame import compute_dcm_frame
    from mmml.interfaces.dcmInterface.mdcm_writer import write_mdcm

    # See tests/integration/test_dcm_charmm_regression.py for full flow.
"""

# *AN EXAMPLE OF LOADING THE DCM MODULE IN pycharmm*
#
#  import mmml
# from mmml.interfaces.pycharmmInterface import import_pycharmm
# import os
# import sys
# from pathlib import Path
#
# from mmml.cli.make.make_res import main_loop
# import argparse
# import pycharmm
#
#
# def main():
#     args = argparse.Namespace(res="MEOH", skip_energy_show=True)
#     print("=== 01: make_res programmatic ===")
#     atoms = main_loop(args)
#     print(f"Generated {len(atoms)} atoms")
#     print("Output: pdb/initial.pdb, psf/initial.psf, xyz/initial.xyz, CHARMM topology files")
#
#
# output =    main()
#
#
# dcm_str = """1 0
#
# MEOH
# 4
# 1 2 3 BO
# 2 0
# 0.0 0.0 0.0 1.0
# 0.0 0.0 0.0 1.0
# 1 0
# 0.0 0.0 0.0 1.0
# 1 0
# 0.0 0.0 0.0 1.0
# 1 2 4 BO
# 0 0
# 0 0
# 1 0
# 0.0 0.0 0.0 1.0
# 1 2 5 BO
# 0 0
# 0 0
# 1 0
# 0.0 0.0 0.0 1.0
# 1 2 6 BO
# 0 0
# 0 0
# 1 0
# 0.0 0.0 0.0 1.0
# """
#
# with open("meoh.mdcm", "w") as f:
#     f.write(dcm_str)
#
# dcm_script = """
# open unit 11 card read name meoh.mdcm
# !open unit 12 card read name water.kern
# open unit 99 write card name dcm.xyz
# DCM IUDCM 11 TSHIFT XYZ 99
# !DCM KERN 12 IUDCM 11 TSHIFT XYZ 15
# """
# pycharmm.lingo.charmm_script(dcm_script)
# pycharmm.lingo.charmm_script("ENER")
# pycharmm.lingo.charmm_script("STOP")
