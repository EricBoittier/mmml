import argparse
import sys
import pycharmm
from pycharmm.lingo import charmm_script

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    build_cluster_from_args
)
from mmml.interfaces.pycharmmInterface.cutoffs import add_handoff_cutoff_args
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    add_cluster_args,
    add_charmm_output_args,
)
from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
    enforce_charmm_energy_term_policies
)

parser = argparse.ArgumentParser()
parser.add_argument("composition_arg", nargs="?", help="Shortcut for --composition")
add_cluster_args(parser)
add_charmm_output_args(parser)
add_handoff_cutoff_args(parser)
parser.add_argument("--charmm-zero-energy-terms", default="vdw")

args = parser.parse_args(["DCM:4"])
handoff = build_cluster_from_args(args)

charmm_script("ENER")

class MockMLSelection:
    def get_atom_indexes(self): return []

from mmml.interfaces.pycharmmInterface.charmm_image_geometry import run_mlpot_pbc_image_registration_gate
run_mlpot_pbc_image_registration_gate(20.4, args, "test", True)

enforce_charmm_energy_term_policies(
    args,
    ml_selection=MockMLSelection(),
    use_pbc=True,
    cubic_box_side_A=20.4,
    verbose=True,
    skip_ener_probe=False
)
