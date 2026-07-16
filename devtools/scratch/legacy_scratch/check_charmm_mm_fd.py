import argparse
import sys
import numpy as np
from pathlib import Path

# Add mmml root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    add_cluster_args,
    build_cluster_from_args_with_tag,
)
from mmml.interfaces.pycharmmInterface.cutoffs import add_handoff_cutoff_args

import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
import pycharmm
import pycharmm.energy as energy
import pycharmm.coor as coor
import pycharmm.lib as lib

def calculate_charmm_mm_energy_force():
    """Call PyCHARMM's ENERGY command and extract MM energy and forces."""
    energy.show()
    e = energy.get_total()
    # Forces are stored in the DX, DY, DZ arrays in PyCHARMM
    # We can retrieve them using coor.get_forces() if that exists,
    # or by accessing the common block.
    # Actually, in PyCHARMM, `pycharmm.lib.get_forces()` returns the forces?
    # Let's try `pycharmm.lingo.charmm_script("ENER")` and getting forces.
    
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_total_forces_kcalmol_A
    
    force = charmm_total_forces_kcalmol_A()
    
    return e, np.asarray(force)

def set_charmm_positions(pos):
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions
    sync_charmm_positions(pos)

def fd_check(pos, step=1e-4):
    n = len(pos)
    analytic_e, analytic_f = calculate_charmm_mm_energy_force()
    
    fd_f = np.zeros_like(pos)
    
    for i in range(n):
        for j in range(3):
            pos_plus = pos.copy()
            pos_plus[i, j] += step
            set_charmm_positions(pos_plus)
            e_plus, _ = calculate_charmm_mm_energy_force()
            
            pos_minus = pos.copy()
            pos_minus[i, j] -= step
            set_charmm_positions(pos_minus)
            e_minus, _ = calculate_charmm_mm_energy_force()
            
            # Central difference for force (F = -dE/dx)
            fd_f[i, j] = -(e_plus - e_minus) / (2 * step)
            
    # Restore original positions
    set_charmm_positions(pos)
    
    max_diff = np.max(np.abs(analytic_f - fd_f))
    print(f"Base Energy: {analytic_e:.6f} kcal/mol")
    print(f"Max absolute difference between analytical and FD forces: {max_diff:.6e} kcal/mol/A")
    print("Analytical Force sample (first atom):", analytic_f[0])
    print("FD Force sample (first atom):", fd_f[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("composition_arg", nargs="?", default="DCM:2")
    add_cluster_args(parser)
    add_handoff_cutoff_args(parser)
    args = parser.parse_args()

    if args.composition_arg:
        if args.composition is not None and args.composition != args.composition_arg:
            parser.error("pass composition either positionally or with --composition, not both")
        args.composition = args.composition_arg

    z, positions, n_monomers, tag = build_cluster_from_args_with_tag(args)
    
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_use_pbc, resolve_pbc_box_side
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds
    
    if not resolve_use_pbc(args):
        setup_default_nbonds()
    else:
        box_side = resolve_pbc_box_side(args, positions)
        setup_charmm_environment(use_pbc=True, cubic_box_side_A=box_side)

    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array
    pos = np.asarray(get_charmm_positions_array(), dtype=np.float64)
    
    print(f"Testing CHARMM pure MM forces on {n_monomers} monomers ({len(z)} atoms)")
    fd_check(pos)

if __name__ == "__main__":
    main()
