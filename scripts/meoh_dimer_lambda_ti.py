#!/usr/bin/env python3
"""
KARML lambda dynamics / thermodynamic integration for arbitrary clusters.

Prefer ``karml md-system --setup lambda_ti`` (see ``karml.cli.run.lambda_dynamics``).
MBAR: ``karml lambda-mbar`` or ``scripts/meoh_dimer_lambda_mbar.py``.
"""

from karml.cli.run.lambda_dynamics import main_lambda_dynamics

if __name__ == "__main__":
    raise SystemExit(main_lambda_dynamics())
