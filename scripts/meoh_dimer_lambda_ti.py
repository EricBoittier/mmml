#!/usr/bin/env python3
"""
MMML lambda dynamics / thermodynamic integration for arbitrary clusters.

Build any composition (as ``mmml md-system``), select coupled residues by
1-based index in cluster order, and sample λ windows. MBAR is separate:
``mmml lambda-mbar`` or ``scripts/meoh_dimer_lambda_mbar.py``.

Example (methanol dimer, couple residue 1):

  mmml md-system --setup lambda_ti --composition MEOH:2 --couple-residues 1 \\
    --output-dir artifacts/meoh_dimer_lambda_ti --checkpoint PATH

Example (mixed cluster, couple residues 1 and 3):

  mmml md-system --setup lambda_ti --composition MEOH:2,TIP3:1 --couple-residues 1,3 \\
    --spacing 6 --n-prod 2000
"""

from mmml.cli.run.lambda_dynamics import main_lambda_dynamics

if __name__ == "__main__":
    raise SystemExit(main_lambda_dynamics())
