#!/usr/bin/env python3
"""Dispatch one reaction-path campaign job and write status.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from campaign_lib import load_config

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[3]


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _env(repo: Path, cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    example = repo / str(cfg.get("example_dir", "examples/m"))
    env.setdefault("JAX_ENABLE_X64", "1")
    env["MMML_CKPT"] = str(repo / cfg["checkpoint"])
    env["MMML_CGENFF_EXTRA_RTF"] = str(example / "top_ch3cl.rtf")
    env["MMML_CGENFF_EXTRA_PRM"] = str(example / "par_ch3cl.prm")
    env["ARTIFACTS_DIR"] = str(repo / cfg.get("output_root", "artifacts/nh3_ch3cl_reaction_path"))
    # Prefer CUDA on studix GPU nodes; fall back if unset by launcher.
    if "JAX_PLATFORMS" not in env:
        env["JAX_PLATFORMS"] = "cuda,cpu"
    return env


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def job_endpoints(repo: Path, cfg: dict[str, Any], out: Path, env: dict[str, str]) -> dict[str, Any]:
    script = repo / "examples/m/07_export_neb_endpoints.py"
    neb_dir = out / "neb_xyz"
    _run(
        [sys.executable, str(script), "-o", str(neb_dir)],
        cwd=repo,
        env=env,
    )
    return {
        "job": "endpoints",
        "ok": True,
        "reag": str(neb_dir / "reag_0_opt.xyz"),
        "prod": str(neb_dir / "prod_0_opt.xyz"),
    }


def job_make_boxes(repo: Path, cfg: dict[str, Any], out: Path, env: dict[str, str]) -> dict[str, Any]:
    """Build solvated boxes under ``$ARTIFACTS_DIR/boxes/{solvent}/``.

    ``out`` is the campaign root (same as ``ARTIFACTS_DIR``), not a nested
    ``boxes/`` directory — ``08_make_boxes.sh`` already appends ``boxes/``.
    """
    mb = cfg.get("make_boxes") or {}
    env = dict(env)
    env["BOX_SIZE"] = str(cfg.get("box_size", 30.0))
    env["N_SOLVENT"] = str(mb.get("n_solvent", 12))
    env["USE_DENSITY"] = "1" if mb.get("use_density") else "0"
    env["ARTIFACTS_DIR"] = str(out)
    _run(["bash", str(repo / "examples/m/08_make_boxes.sh")], cwd=repo, env=env)
    boxes = {}
    for sol in (cfg.get("solvents") or ["tip3"]):
        pdb = out / "boxes" / sol / "model.pdb"
        psf = out / "boxes" / sol / "model.psf"
        boxes[sol] = {"pdb": str(pdb), "psf": str(psf), "exists": pdb.is_file() and psf.is_file()}
    if not all(v["exists"] for v in boxes.values()):
        raise RuntimeError(f"make-box incomplete: {boxes}")
    return {"job": "make_boxes", "ok": True, "boxes": boxes}


def job_neb(repo: Path, cfg: dict[str, Any], out: Path, env: dict[str, str]) -> dict[str, Any]:
    neb_cfg = cfg.get("neb") or {}
    endpoints = Path(env["ARTIFACTS_DIR"]) / "endpoints" / "neb_xyz"
    # Prefer campaign endpoints; fall back to examples/m/neb
    reag = endpoints / "reag_0_opt.xyz"
    prod = endpoints / "prod_0_opt.xyz"
    if not reag.is_file():
        reag = repo / "examples/m/neb/reag_0_opt.xyz"
        prod = repo / "examples/m/neb/prod_0_opt.xyz"
        if not reag.is_file():
            _run(
                [sys.executable, str(repo / "examples/m/07_export_neb_endpoints.py")],
                cwd=repo,
                env=env,
            )
    cmd = [
        "uv",
        "run",
        "mmml",
        "neb",
        "--checkpoint",
        str(repo / cfg["checkpoint"]),
        "--initial",
        str(reag),
        "--final",
        str(prod),
        "--output-dir",
        str(out),
        "--n-images",
        str(int(neb_cfg.get("n_images", 11))),
        "--max-steps",
        str(int(neb_cfg.get("max_steps", 80))),
        "--fmax",
        str(float(neb_cfg.get("fmax", 0.05))),
        "--overwrite",
    ]
    _run(cmd, cwd=repo, env=env)
    summary = out / "neb_summary.json"
    if not summary.is_file():
        raise RuntimeError(f"missing {summary}")
    return {"job": "neb", "ok": True, "summary": str(summary)}


def job_dmc(
    repo: Path,
    cfg: dict[str, Any],
    out: Path,
    env: dict[str, str],
    *,
    basin: str,
) -> dict[str, Any]:
    dmc = cfg.get("dmc") or {}
    endpoints = Path(env["ARTIFACTS_DIR"]) / "endpoints" / "neb_xyz"
    mapping = {"react": "reag_0_opt.xyz", "product": "prod_0_opt.xyz"}
    if basin not in mapping:
        raise ValueError(f"unknown dmc basin {basin!r}")
    xyz = endpoints / mapping[basin]
    if not xyz.is_file():
        xyz = repo / "examples/m/neb" / mapping[basin]
    cmd = [
        "uv",
        "run",
        "mmml",
        "dmc",
        "--natm",
        "9",
        "--nwalker",
        str(int(dmc.get("nwalker", 64))),
        "--stepsize",
        str(float(dmc.get("stepsize", 5e-4))),
        "--nstep",
        str(int(dmc.get("nstep", 200))),
        "--eqstep",
        str(int(dmc.get("eqstep", 50))),
        "--alpha",
        str(float(dmc.get("alpha", 1200.0))),
        "--max-batch",
        str(int(dmc.get("nwalker", 64))),
        "--seed",
        str(int(cfg.get("seed", 0))),
        "--checkpoint",
        str(repo / cfg["checkpoint"]),
        "--input",
        str(xyz),
        "--output-dir",
        str(out),
    ]
    _run(cmd, cwd=repo, env=env)
    logs = list(out.glob("*.log"))
    if not logs:
        raise RuntimeError(f"no DMC log under {out}")
    return {"job": "dmc", "basin": basin, "ok": True, "log": str(logs[0])}


def _umbrella_variant(cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    variants = ((cfg.get("umbrella") or {}).get("variants") or {})
    if variant not in variants:
        raise KeyError(f"unknown umbrella variant {variant!r}")
    return dict(variants[variant])


def job_umbrella_gas(
    repo: Path,
    cfg: dict[str, Any],
    out: Path,
    env: dict[str, str],
    *,
    variant: str,
) -> dict[str, Any]:
    knobs = _umbrella_variant(cfg, variant)
    endpoints = Path(env["ARTIFACTS_DIR"]) / "endpoints" / "neb_xyz"
    structure = endpoints / "reag_0_opt.xyz"
    if not structure.is_file():
        structure = repo / "examples/m/neb/reag_0_opt.xyz"
    umb_yaml = {
        "engine": "packed_ml",
        "checkpoint": str(repo / cfg["checkpoint"]),
        "structure": str(structure),
        "output_dir": str(out),
        "atom_i": 2,
        "atom_j": 1,
        "move_with": [1, 3, 4, 5],
        "xi_min": float(knobs["xi_min"]),
        "xi_max": float(knobs["xi_max"]),
        "n_windows": int(knobs["n_windows"]),
        "k_ev_A2": float(knobs["k_ev_A2"]),
        "temperature_K": 300.0,
        "timestep_fs": float(knobs.get("timestep_fs_gas", 0.1)),
        "nsteps": int(knobs["nsteps"]),
        "printfreq": int(knobs.get("printfreq", 50)),
        "savefreq": int(knobs.get("savefreq", 50)),
        "seed": int(cfg.get("seed", 0)),
        "seed_mode": "stretch",
        "thermostat": "langevin",
        "overwrite": True,
    }
    cfg_path = out / "umbrella_config.yaml"
    _dump_yaml(cfg_path, umb_yaml)
    _run(
        ["uv", "run", "mmml", "umbrella-sample", "--config", str(cfg_path), "--overwrite"],
        cwd=repo,
        env=env,
    )
    return {
        "job": "umbrella_gas",
        "variant": variant,
        "ok": True,
        "config": str(cfg_path),
        "snapshots": str(out / "umbrella_snapshots.npz"),
    }


def job_umbrella_sol(
    repo: Path,
    cfg: dict[str, Any],
    out: Path,
    env: dict[str, str],
    *,
    solvent: str,
    variant: str,
) -> dict[str, Any]:
    knobs = _umbrella_variant(cfg, variant)
    # make_boxes writes ARTIFACTS_DIR/boxes/{sol}/...
    boxes = Path(env["ARTIFACTS_DIR"]) / "boxes" / solvent
    psf = boxes / "model.psf"
    pdb = boxes / "model.pdb"
    if not psf.is_file() or not pdb.is_file():
        raise FileNotFoundError(
            f"missing make-box artifacts for {solvent}: {psf} / {pdb} "
            "(enable make_boxes and re-run)"
        )

    # Resolve AMM1 move-with from PSF
    from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds

    atoms, _ = read_psf_atoms_and_bonds(psf)
    move_with = [a.index for a in atoms if a.resname.upper() == "AMM1"]

    umb_yaml = {
        "engine": "hybrid_jaxmd",
        "checkpoint": str(repo / cfg["checkpoint"]),
        "from_psf": str(psf),
        "from_pdb": str(pdb),
        "box_size": float(cfg.get("box_size", 30.0)),
        "output_dir": str(out),
        "ml_resnames": ["AMM1", "CH3CL"],
        "atom_name_i": "C1",
        "atom_name_j": "N1",
        "atom_i": 0,
        "atom_j": 1,
        "move_with": move_with,
        "xi_min": float(knobs["xi_min"]),
        "xi_max": float(knobs["xi_max"]),
        "n_windows": int(knobs["n_windows"]),
        "k_ev_A2": float(knobs["k_ev_A2"]),
        "temperature_K": 300.0,
        "timestep_fs": float(knobs.get("timestep_fs_sol", 0.5)),
        "nsteps": int(knobs["nsteps"]),
        "printfreq": int(knobs.get("printfreq", 50)),
        "savefreq": int(knobs.get("savefreq", 50)),
        "seed": int(cfg.get("seed", 0)),
        "seed_mode": "stretch",
        "overwrite": True,
        "lr_solver": "mic",
    }
    cfg_path = out / "umbrella_config.yaml"
    _dump_yaml(cfg_path, umb_yaml)
    _run(
        ["uv", "run", "mmml", "umbrella-sample", "--config", str(cfg_path), "--overwrite"],
        cwd=repo,
        env=env,
    )
    return {
        "job": "umbrella_sol",
        "solvent": solvent,
        "variant": variant,
        "ok": True,
        "config": str(cfg_path),
        "snapshots": str(out / "umbrella_snapshots.npz"),
        "ml_atoms": len(move_with) + sum(
            1 for a in atoms if a.resname.upper() == "CH3CL"
        ),
    }


def job_adumb_gas(repo: Path, cfg: dict[str, Any], out: Path, env: dict[str, str]) -> dict[str, Any]:
    adumb = cfg.get("adumb") or {}
    env = dict(env)
    env["ARTIFACTS_DIR"] = str(out.parent)  # script writes adumb_* under ARTIFACTS_DIR
    env["OUT"] = str(out)
    env["USE_NPZ_PDB"] = "1" if adumb.get("use_npz_pdb", True) else "0"
    env["SOLVATED"] = "0"
    # Point 09 script at a campaign-local copy of the vacuum YAML with redirected output.
    src = repo / "examples/m/yaml/adumb_nc_distance.yaml"
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["output_dir"] = str(out)
    cfg_path = out / "adumb_config.yaml"
    _dump_yaml(cfg_path, data)
    env["CFG"] = str(cfg_path)
    # 09 script hardcodes OUT from solvated flag; override by setting CFG and patching via env
    # Easiest: call md-system directly with the YAML.
    _run(
        ["uv", "run", "mmml", "md-system", "--config", str(cfg_path)],
        cwd=repo,
        env=env,
    )
    return {"job": "adumb_gas", "ok": True, "config": str(cfg_path)}


def job_adumb_sol(
    repo: Path,
    cfg: dict[str, Any],
    out: Path,
    env: dict[str, str],
    *,
    solvent: str,
) -> dict[str, Any]:
    adumb = cfg.get("adumb") or {}
    src = repo / f"examples/m/yaml/adumb_nc_distance_{solvent}.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"missing ADUMB YAML for solvent={solvent}: {src}")
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["output_dir"] = str(out)
    data["box_size"] = float(cfg.get("box_size", data.get("box_size", 30.0)))
    cfg_path = out / "adumb_config.yaml"
    _dump_yaml(cfg_path, data)
    env = dict(env)
    if adumb.get("use_npz_pdb", True):
        # Seed solute from NPZ PDB via md-system --from-pdb path is handled by 09 script;
        # for campaign we use Packmol composition in the YAML as-is.
        pass
    _run(
        ["uv", "run", "mmml", "md-system", "--config", str(cfg_path)],
        cwd=repo,
        env=env,
    )
    return {"job": "adumb_sol", "solvent": solvent, "ok": True, "config": str(cfg_path)}


def job_mbar(repo: Path, cfg: dict[str, Any], out: Path, env: dict[str, str], *, run_dir: Path) -> dict[str, Any]:
    _run(
        [
            "uv",
            "run",
            "mmml",
            "umbrella-mbar",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(repo / cfg["checkpoint"]),
        ],
        cwd=repo,
        env=env,
    )
    summary = run_dir / "umbrella_summary.json"
    return {"job": "mbar", "ok": True, "run_dir": str(run_dir), "summary": str(summary)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--solvent", default=None)
    parser.add_argument("--basin", default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="For mbar: umbrella run dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = _env(repo, cfg)
    # Campaign ARTIFACTS_DIR is the output_root (shared across jobs).
    env["ARTIFACTS_DIR"] = str((repo / cfg.get("output_root", "artifacts/nh3_ch3cl_reaction_path")).resolve())

    t0 = time.time()
    try:
        if args.job == "endpoints":
            result = job_endpoints(repo, cfg, out, env)
        elif args.job == "make_boxes":
            result = job_make_boxes(repo, cfg, out, env)
        elif args.job == "neb":
            result = job_neb(repo, cfg, out, env)
        elif args.job == "dmc":
            result = job_dmc(repo, cfg, out, env, basin=str(args.basin))
        elif args.job == "umbrella_gas":
            result = job_umbrella_gas(repo, cfg, out, env, variant=str(args.variant))
        elif args.job == "umbrella_sol":
            result = job_umbrella_sol(
                repo, cfg, out, env, solvent=str(args.solvent), variant=str(args.variant)
            )
        elif args.job == "adumb_gas":
            result = job_adumb_gas(repo, cfg, out, env)
        elif args.job == "adumb_sol":
            result = job_adumb_sol(repo, cfg, out, env, solvent=str(args.solvent))
        elif args.job == "mbar":
            if args.run_dir is None:
                raise SystemExit("--run-dir required for mbar")
            result = job_mbar(repo, cfg, out, env, run_dir=args.run_dir.resolve())
        else:
            raise SystemExit(f"unknown job {args.job!r}")
        result["elapsed_s"] = time.time() - t0
        result["host"] = os.uname().nodename
        _write_status(args.status, result)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        payload = {
            "job": args.job,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.time() - t0,
            "host": os.uname().nodename,
        }
        _write_status(args.status, payload)
        print(json.dumps(payload, indent=2), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
