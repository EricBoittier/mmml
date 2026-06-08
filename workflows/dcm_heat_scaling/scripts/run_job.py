#!/usr/bin/env python3
"""Run one DCM:N heat scaling job via mmml md-system."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from heat_lib import (  # noqa: E402
    build_md_system_argv,
    dt_fs_from_slug,
    load_config,
    paths_for_run,
    workflow_root,
)


def _repo_root() -> Path:
    return workflow_root().parents[1]


def _resolve_mmml_cmd(md_argv: list[str]) -> list[str]:
    mmml_bin = os.environ.get("MMML_BIN")
    if mmml_bin:
        return [mmml_bin, "md-system", *md_argv]

    venv_mmml = _repo_root() / ".venv" / "bin" / "mmml"
    if venv_mmml.is_file():
        return [str(venv_mmml), "md-system", *md_argv]

    from shutil import which

    on_path = which("mmml")
    if on_path:
        return [on_path, "md-system", *md_argv]

    return [sys.executable, "-m", "mmml.cli.__main__", "md-system", *md_argv]


def _resolve_mpirun_wrapper(cfg: dict) -> Path:
    raw = Path(str(cfg.get("mpirun_wrapper", "../../scripts/mmml-charmm-mpirun.sh")))
    if raw.is_absolute():
        return raw
    return (workflow_root() / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_monomers", type=int, help="Cluster size N in DCM:N")
    parser.add_argument("repeat", type=int, help="Repeat index N in dcmX_npt_x64_N")
    parser.add_argument("dt_slug", type=str, help="Timestep slug (dt025 or dt0125)")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
        help="Workflow config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sizes = [int(x) for x in cfg.get("cluster_sizes", [])]
    repeats = [int(x) for x in cfg.get("repeats", [1])]
    dt_values = [float(x) for x in cfg.get("dt_fs_values", [0.25, 0.125])]

    n = int(args.n_monomers)
    rep = int(args.repeat)
    dt_fs = dt_fs_from_slug(args.dt_slug)

    if n not in sizes:
        raise SystemExit(f"n_monomers={n} not in cluster_sizes {sizes}")
    if rep not in repeats:
        raise SystemExit(f"repeat={rep} not in repeats {repeats}")
    if dt_fs not in dt_values:
        raise SystemExit(f"dt_fs={dt_fs} not in dt_fs_values {dt_values}")

    md_argv = build_md_system_argv(cfg, n, rep, dt_fs)
    os.chdir(_repo_root())

    if bool(cfg.get("use_mpirun", False)):
        mpirun_wrapper = _resolve_mpirun_wrapper(cfg)
        if not mpirun_wrapper.is_file():
            raise SystemExit(f"MPI wrapper not found: {mpirun_wrapper}")
        cmd = [str(mpirun_wrapper), "md-system", *md_argv]
    else:
        cmd = _resolve_mmml_cmd(md_argv)

    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"DCM:{n} heat repeat={rep} dt={dt_fs} failed with exit code {rc}", file=sys.stderr)
        return rc

    paths = paths_for_run(cfg, n, rep, dt_fs)
    if not paths["heat_dcd"].is_file():
        print(f"Expected heat DCD missing: {paths['heat_dcd']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
