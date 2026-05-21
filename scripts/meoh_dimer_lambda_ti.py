#!/usr/bin/env python3
"""
MMML lambda dynamics / thermodynamic integration for arbitrary clusters.

Prefer ``mmml md-system --setup lambda_ti`` (see ``mmml.cli.run.lambda_dynamics``).
MBAR: ``mmml lambda-mbar`` or ``scripts/meoh_dimer_lambda_mbar.py``.
"""

from mmml.cli.run.lambda_dynamics import main_lambda_dynamics

if __name__ == "__main__":
    raise SystemExit(main_lambda_dynamics())
