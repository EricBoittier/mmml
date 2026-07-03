#!/usr/bin/env python3
"""Fresh CHARMM session for isolated bonded recovery (sidecar worker).

User-run under ``mmml-charmm-mpirun.sh`` from
:func:`mmml.interfaces.pycharmmInterface.mlpot.charmm_recovery_sidecar.run_charmm_recovery_sidecar`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHARMM bonded recovery sidecar worker")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest from charmm_recovery_sidecar.build_sidecar_manifest",
    )
    return parser.parse_args()


def _run_recovery(manifest_path: Path) -> dict[str, float]:
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_recovery_sidecar import (
        SidecarRecoveryManifest,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _import_pycharmm_modules
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        apply_crd_file_to_charmm,
    )
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    manifest = SidecarRecoveryManifest.load(manifest_path)
    psf = Path(manifest.psf).expanduser().resolve()
    input_crd = Path(manifest.input_crd).expanduser().resolve()
    if not psf.is_file():
        raise FileNotFoundError(f"sidecar PSF not found: {psf}")
    if not input_crd.is_file():
        raise FileNotFoundError(f"sidecar input CRD not found: {input_crd}")

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import read_psf_card_file

    read_cgenff_toppar()
    read_psf_card_file(psf)
    apply_crd_file_to_charmm(input_crd)

    if manifest.use_pbc:
        if manifest.box_side_A is None or float(manifest.box_side_A) <= 0.0:
            raise ValueError("sidecar PBC recovery requires positive box_side_A")
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

        setup_charmm_environment(
            use_pbc=True,
            cubic_box_side_A=float(manifest.box_side_A),
        )
    else:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

        setup_default_nbonds()

    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import (
        apply_bonded_mm_only_block,
        apply_bonded_vdw_recovery_block,
    )

    if manifest.include_vdw:
        apply_bonded_vdw_recovery_block(verbose=manifest.verbose)
    else:
        apply_bonded_mm_only_block(verbose=manifest.verbose)

    pycharmm, cons_fix, *_ = _import_pycharmm_modules()
    minimize = _import_pycharmm_modules()[3]
    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        charmm_quiet_output,
        run_charmm_script_quiet,
    )

    run_charmm_script_quiet("ENER")
    grms_before = float(charmm_grms())
    if manifest.verbose:
        print(
            f"charmm_bonded_recovery_worker: start GRMS={grms_before:.4f} kcal/mol/Å",
            flush=True,
        )
    if int(manifest.nstep_sd) > 0:
        with charmm_quiet_output():
            minimize.run_sd(
                nstep=int(manifest.nstep_sd),
                nprint=max(1, int(manifest.nprint)),
                tolenr=float(manifest.tolenr),
                tolgrd=float(manifest.tolgrd),
                inbfrq=0,
                ihbfrq=0,
            )
    run_charmm_script_quiet("ENER")
    grms_after = float(charmm_grms())
    cons_fix.turn_off()

    import pycharmm.write as write

    out_crd = Path(manifest.output_crd).expanduser().resolve()
    out_crd.parent.mkdir(parents=True, exist_ok=True)
    write.coor_card(str(out_crd))

    result = {
        "grms": grms_after,
        "grms_before": grms_before,
        "nstep_sd": float(manifest.nstep_sd),
    }
    result_path = Path(manifest.output_result).expanduser().resolve()
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if manifest.verbose:
        print(
            f"charmm_bonded_recovery_worker: end GRMS={grms_after:.4f} kcal/mol/Å",
            flush=True,
        )
    return result


def main() -> int:
    args = _parse_args()
    try:
        _run_recovery(args.manifest.expanduser().resolve())
    except Exception as exc:
        print(f"charmm_bonded_recovery_worker: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
