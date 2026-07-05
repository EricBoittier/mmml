import argparse
import sys
from mmml.interfaces.pycharmmInterface.import_pycharmm import *
from scripts.check_mlpot_forces_fd import _parse_args, _setup_charmm
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import build_cluster_from_args_with_tag
import pycharmm
import pycharmm.energy as energy

args = _parse_args(["ACO:2"])
z, positions, n_monomers, tag = build_cluster_from_args_with_tag(args)
_setup_charmm(args, positions)

pycharmm.lingo.charmm_script("ENER")
print("Before zeroing VDW:", energy.get_vdw())

from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import _reload_prm_overlay
import pycharmm.select as select
from mmml.interfaces.pycharmmInterface.mlpot.setup import SelectionWrapper

overlay = Path("artifacts/pycharmm_mlpot/force_fd/charmm_energy_policy/zeroed_vdw.prm")
ml_selection = SelectionWrapper(select.SelectAtoms().all_atoms())

_reload_prm_overlay(
    overlay,
    use_pbc=False,
    cubic_box_side_A=None,
    ml_selection=ml_selection,
    zero_ml_charges=False,
    verbose=True,
    workflow_args=args,
)

pycharmm.lingo.charmm_script("ENER")
print("After zeroing VDW:", energy.get_vdw())
