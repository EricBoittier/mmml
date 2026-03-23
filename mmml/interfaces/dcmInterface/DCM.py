"""
Example: build mdcm from DCMNet H5 and run CHARMM DCM.

Usage:
    from mmml.interfaces.dcmInterface import build_mdcm_from_dcmnet, generate_dcm_xyz

    # Single frame:
    frames, charges_per_frame = build_mdcm_from_dcmnet(
        "charmm_ml_comparison.h5", frame_idx=0, out_mdcm="meoh.mdcm"
    )

    # Average over all conformations:
    frames, charges_per_frame = build_mdcm_from_dcmnet(
        "charmm_ml_comparison.h5",
        out_mdcm="meoh_avg.mdcm",
        average_over_frames=True,
        frame_indices=None,  # None = all frames
    )
    # Python dcm.xyz (without CHARMM):
    import h5py
    with h5py.File("charmm_ml_comparison.h5", "r") as f:
        R = f["R"][0][:int(f["N"][0])]
    generate_dcm_xyz(R, frames, charges_per_frame, "dcm.xyz")
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
