#!/usr/bin/env python3
"""Run one DCM:3 NVE job (preset × geometry) via mmml md-system."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cutoff_lib import (  # noqa: E402
    build_md_system_argv,
    composition_tag,
    expected_nve_nstep,
    load_config,
    run_dir,
    workflow_root,
)


def _repo_root() -> Path:
    return workflow_root().parents[1]


def _resolve_mpirun_wrapper(cfg: dict) -> Path:
    raw = Path(str(cfg.get("mpirun_wrapper", "../../scripts/mmml-charmm-mpirun.sh")))
    if raw.is_absolute():
        return raw
    return (workflow_root() / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset_id", help="Cutoff preset key from config.yaml")
    parser.add_argument("geom_id", help="Geometry variant key from config.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_dir(cfg, args.preset_id, args.geom_id)
    out.mkdir(parents=True, exist_ok=True)
    md_argv = build_md_system_argv(cfg, args.preset_id, args.geom_id, output_dir=out)

    os.chdir(_repo_root())
    mpirun_wrapper = _resolve_mpirun_wrapper(cfg)
    if not mpirun_wrapper.is_file():
        raise SystemExit(f"MPI wrapper not found: {mpirun_wrapper}")
    cmd = [str(mpirun_wrapper), "md-system", *md_argv]

    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(
            f"Job preset={args.preset_id!r} geom={args.geom_id!r} failed: exit {rc}",
            file=sys.stderr,
        )
        return rc

    tag = composition_tag(cfg)
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )

    try:
        assert_stage_dynamics_completed(
            stage="nve",
            expected_nstep=expected_nve_nstep(cfg),
            nsavc=int(cfg["dcd_nsavc"]),
            dcd_path=out / f"nve_{tag}.dcd",
            restart_path=out / f"nve_{tag}.res",
        )
    except RuntimeError as exc:
        print(f"Post-run validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
