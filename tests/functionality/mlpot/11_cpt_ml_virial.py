#!/usr/bin/env python3
"""One ENER FORCE: CHARMM PRSI vs atomic Σ F·r vs strain -dE/dV.

No CPT / dyna. Diagnoses whether the 2211 34→36.4 Å walk is missing USER
virial (H1) or CHARMM CPT following atomic virial while JAX-MD uses strain (H2).

Cluster (gpu08), after this script exists::

    python tests/functionality/mlpot/11_cpt_ml_virial.py \\
      --psf ~/gpu_jobs/etoh34_box/model.psf \\
      --crd ~/gpu_jobs/etoh34_box/model.crd \\
      --checkpoint ~/metatomic-runs/training/students/etoh_omol_l_A_best.json \\
      --box-side 34 --mm-switch-width 3.0 --composition ETOH:405

    python tests/functionality/mlpot/11_cpt_ml_virial.py \\
      --psf ~/gpu_jobs/etoh34_box/model.psf \\
      --crd ~/gpu_jobs/etoh34_box/model.crd \\
      --continue-from ~/gpu_jobs/charmm_npt6ps_etoh34_2211/equi.res \\
      --checkpoint ~/metatomic-runs/training/students/etoh_omol_l_A_best.json \\
      --box-side 36.41 --mm-switch-width 3.0 --composition ETOH:405

Do not use 2241 as the primary fixture (Hoover piston already dumped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psf", type=Path, required=True)
    p.add_argument("--crd", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--box-side", type=float, required=True, help="cubic side, Å")
    p.add_argument("--composition", type=str, default="ETOH:405")
    p.add_argument("--mm-switch-width", type=float, default=3.0)
    p.add_argument("--continue-from", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Scratch dir for md-system argv (default: cwd/cpt_ml_virial)",
    )
    p.add_argument("--rel-dv", type=float, default=1.0e-4)
    p.add_argument(
        "-o",
        "--json-out",
        type=Path,
        default=None,
        help="Write the JSON report here",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from mmml.interfaces.pycharmmInterface.mlpot.virial_compare import (
        report_to_json,
        run_live_cpt_ml_virial,
    )

    report = run_live_cpt_ml_virial(
        psf=args.psf,
        crd=args.crd,
        checkpoint=args.checkpoint,
        box_side=float(args.box_side),
        composition=str(args.composition),
        mm_switch_width=float(args.mm_switch_width),
        continue_from=args.continue_from,
        output_dir=args.output_dir,
        rel_dv=float(args.rel_dv),
    )
    text = report_to_json(report)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(
        f"\n  P_PRSI     {report['p_prsi_atm']:12.2f} atm"
        f"\n  P_atomic   {report['p_atomic_atm']:12.2f} atm"
        f"\n  P_strain   {report['p_strain_atm'] if report['p_strain_atm'] is not None else float('nan'):12.2f} atm"
        f"\n  VIRE       {report['vire_kcal']}"
        f"\n  VIRI       {report['viri_kcal']}"
        f"\n  {report['hypothesis']}: {report['reason']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
