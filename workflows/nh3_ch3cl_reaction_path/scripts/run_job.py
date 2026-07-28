#!/usr/bin/env python3
"""Dispatch one NH₃–CH₃Cl reaction-path campaign job and write status.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import load_config  # noqa: E402

_SOLVENT_RESI = {
    "tip3": "TIP3",
    "acn": "ACN",
    "dmso": "DMSO",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument(
        "--job",
        required=True,
        choices=(
            "endpoints",
            "make_boxes",
            "neb",
            "dmc",
            "umbrella_gas",
            "umbrella_sol",
            "adumb_gas",
            "adumb_sol",
            "mbar",
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--basin", default=None)
    p.add_argument("--solvent", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--run-dir", type=Path, default=None, help="For mbar: umbrella run directory")
    return p.parse_args()


def _setup_env(repo: Path, cfg: dict[str, Any]) -> None:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    ckpt = repo / str(cfg.get("checkpoint", "examples/m/kl.json"))
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # Prefer GPU on Slurm GPU allocations; leave local/login defaults alone.
    if os.environ.get("SLURM_JOB_ID") and (
        os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or str((cfg.get("slurm") or {}).get("partition", "")).lower() == "gpu"
    ):
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("MMML_MLPOT_DEVICE", "gpu")
        os.environ.setdefault("MMML_JAX_WARMUP_DEVICE", "gpu")
    else:
        os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))
        os.environ.setdefault("MMML_MLPOT_DEVICE", os.environ.get("MMML_MLPOT_DEVICE", "cpu"))
    os.environ["MMML_CKPT"] = str(ckpt.resolve())
    os.environ.setdefault("MMML_DATA", str((example / "nh3_ch3cl_filtered.npz").resolve()))
    os.environ.setdefault("MMML_CGENFF_EXTRA_RTF", str((example / "top_ch3cl.rtf").resolve()))
    os.environ.setdefault("MMML_CGENFF_EXTRA_PRM", str((example / "par_ch3cl.prm").resolve()))
    os.environ.setdefault("MMML_MM_PAIR_SOURCE", "jax")
    os.environ.setdefault("MMML_COMPOSITION", "AMM1:1,CH3CL:1")


def _uv_run(repo: Path, args: list[str], *, cwd: Path | None = None) -> None:
    cmd = ["uv", "run", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or repo), check=True)


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variant_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    variants = (cfg.get("umbrella") or {}).get("variants") or {}
    if name not in variants:
        raise KeyError(f"unknown umbrella variant {name!r}")
    return dict(variants[name])


def _ensure_endpoints(repo: Path, example: Path) -> None:
    reag = example / "neb" / "reag_0_opt.xyz"
    prod = example / "neb" / "prod_0_opt.xyz"
    if reag.is_file() and prod.is_file():
        return
    _uv_run(repo, ["python", "examples/m/07_export_neb_endpoints.py"])


def _amm1_move_with(psf: Path) -> str:
    from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds

    atoms, _ = read_psf_atoms_and_bonds(psf)
    idxs = [str(a.index) for a in atoms if a.resname.upper() == "AMM1"]
    if not idxs:
        raise RuntimeError(f"no AMM1 atoms in {psf}")
    return ",".join(idxs)


def _write_hybrid_yaml(
    *,
    template: Path,
    out_yaml: Path,
    solvent: str,
    box_psf: Path,
    box_pdb: Path,
    output_dir: Path,
    box_size: float,
    variant: dict[str, Any],
) -> Path:
    data = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"bad umbrella template: {template}")
    data["engine"] = "hybrid_jaxmd"
    data["from_psf"] = str(box_psf.resolve())
    data["from_pdb"] = str(box_pdb.resolve())
    data["box_size"] = float(box_size)
    data["output_dir"] = str(output_dir.resolve())
    data["overwrite"] = True
    for key in ("xi_min", "xi_max", "n_windows", "k_ev_A2", "nsteps", "printfreq", "savefreq"):
        if key in variant:
            data[key] = variant[key]
    if "timestep_fs_sol" in variant:
        data["timestep_fs"] = variant["timestep_fs_sol"]
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_yaml


def job_endpoints(repo: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    neb_dir = out / "neb_xyz"
    neb_dir.mkdir(parents=True, exist_ok=True)
    _uv_run(
        repo,
        [
            "python",
            "examples/m/07_export_neb_endpoints.py",
            "-o",
            str(neb_dir),
        ],
    )
    # Also refresh the canonical examples/m/neb paths used by YAML defaults.
    _uv_run(repo, ["python", "examples/m/07_export_neb_endpoints.py"])
    return {
        "reag": str(neb_dir / "reag_0_opt.xyz"),
        "prod": str(neb_dir / "prod_0_opt.xyz"),
        "example_neb": str(example / "neb"),
    }


def job_make_boxes(repo: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """``out`` is the campaign artifact root; boxes land in ``out/boxes/{solvent}/``."""
    mb = cfg.get("make_boxes") or {}
    solvents = [str(s).lower() for s in (cfg.get("solvents") or ["tip3"])]
    env = os.environ.copy()
    env["ARTIFACTS_DIR"] = str(out.resolve())
    env["BOX_SIZE"] = str(float(cfg.get("box_size", 30.0)))
    env["N_SOLVENT"] = str(int(mb.get("n_solvent", 12)))
    env["USE_DENSITY"] = "1" if mb.get("use_density") else "0"
    # Restrict 08_make_boxes.sh to requested solvents by wrapping per solvent.
    written: list[str] = []
    solute = out / "solute_amm1_ch3cl.pdb"
    _uv_run(repo, ["python", "examples/m/07_export_solute_pdb.py", "-o", str(solute)])
    for sol in solvents:
        tag = sol.lower()
        work = out / f"make_box_work_{tag}"
        box_out = out / "boxes" / tag
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        box_out.mkdir(parents=True, exist_ok=True)
        resi = _SOLVENT_RESI[tag]
        cmd = [
            "uv",
            "run",
            "mmml",
            "make-box",
            "--pdb",
            str(solute),
            "--res",
            f"nh3ch3cl_{tag}",
            "--box-size",
            env["BOX_SIZE"],
            "--solvent",
            resi,
        ]
        if env["USE_DENSITY"] == "1":
            density = {"ACN": "786", "TIP3": "1000", "DMSO": "1100"}[resi]
            cmd += ["--density", density]
        else:
            cmd += ["--n", env["N_SOLVENT"]]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(work), check=True, env=env)
        pdb_src = work / f"pdb/init-nh3ch3cl_{tag}.pdb"
        psf_src = work / f"psf/system-nh3ch3cl_{tag}.psf"
        shutil.copy2(pdb_src, box_out / "model.pdb")
        shutil.copy2(psf_src, box_out / "model.psf")
        (box_out / "box.json").write_text(
            json.dumps(
                {
                    "box_size": float(env["BOX_SIZE"]),
                    "side_length_A": float(env["BOX_SIZE"]),
                    "solvent": resi,
                    "n_solvent": int(env["N_SOLVENT"]),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(str(box_out))
    return {"boxes": written, "solvents": solvents}


def job_neb(repo: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    _ensure_endpoints(repo, example)
    neb = cfg.get("neb") or {}
    _uv_run(
        repo,
        [
            "mmml",
            "neb",
            "--config",
            str(example / "yaml" / "neb.yaml"),
            "--output-dir",
            str(out),
            "--n-images",
            str(int(neb.get("n_images", 11))),
            "--max-steps",
            str(int(neb.get("max_steps", 200))),
            "--fmax",
            str(float(neb.get("fmax", 0.05))),
            "--overwrite",
        ],
    )
    summary = out / "neb_summary.json"
    data = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    return {
        "summary": str(summary),
        "barrier_kcal_mol": data.get("barrier_kcal_mol"),
        "delta_e_product_kcal_mol": data.get("delta_e_product_kcal_mol"),
    }


def job_dmc(repo: Path, cfg: dict[str, Any], out: Path, basin: str) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    _ensure_endpoints(repo, example)
    xyz = {
        "react": example / "neb" / "reag_0_opt.xyz",
        "product": example / "neb" / "prod_0_opt.xyz",
    }.get(basin)
    if xyz is None or not xyz.is_file():
        raise FileNotFoundError(f"unknown or missing basin {basin!r}")
    dmc = cfg.get("dmc") or {}
    _uv_run(
        repo,
        [
            "mmml",
            "dmc",
            "--natm",
            "9",
            "--nwalker",
            str(int(dmc.get("nwalker", 128))),
            "--stepsize",
            str(float(dmc.get("stepsize", 5.0e-4))),
            "--nstep",
            str(int(dmc.get("nstep", 1000))),
            "--eqstep",
            str(int(dmc.get("eqstep", 200))),
            "--alpha",
            str(float(dmc.get("alpha", 1200.0))),
            "--max-batch",
            str(int(dmc.get("nwalker", 128))),
            "--seed",
            str(int(cfg.get("seed", 0))),
            "--checkpoint",
            os.environ["MMML_CKPT"],
            "--input",
            str(xyz),
            "--output-dir",
            str(out),
        ],
    )
    logs = sorted(out.glob("*.log"))
    return {"basin": basin, "input": str(xyz), "log": str(logs[0]) if logs else ""}


def job_umbrella_gas(
    repo: Path, cfg: dict[str, Any], out: Path, variant: str
) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    _ensure_endpoints(repo, example)
    v = _variant_cfg(cfg, variant)
    cmd = [
        "mmml",
        "umbrella-sample",
        "--config",
        str(example / "yaml" / "umbrella_nc_gas.yaml"),
        "--output-dir",
        str(out),
        "--xi-min",
        str(float(v["xi_min"])),
        "--xi-max",
        str(float(v["xi_max"])),
        "--n-windows",
        str(int(v["n_windows"])),
        "--k",
        str(float(v["k_ev_A2"])),
        "--nsteps",
        str(int(v["nsteps"])),
        "--timestep",
        str(float(v.get("timestep_fs_gas", 0.1))),
        "--printfreq",
        str(int(v.get("printfreq", 50))),
        "--savefreq",
        str(int(v.get("savefreq", v.get("printfreq", 50)))),
        "--seed",
        str(int(cfg.get("seed", 0))),
        "--overwrite",
    ]
    _uv_run(repo, cmd)
    return {
        "variant": variant,
        "summary": str(out / "umbrella_summary.json"),
        "snapshots": str(out / "umbrella_snapshots.npz"),
    }


def job_umbrella_sol(
    repo: Path,
    cfg: dict[str, Any],
    out: Path,
    *,
    solvent: str,
    variant: str,
    artifact_root: Path,
) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    sol = solvent.lower()
    box = artifact_root / "boxes" / sol
    psf = box / "model.psf"
    pdb = box / "model.pdb"
    if not psf.is_file() or not pdb.is_file():
        raise FileNotFoundError(f"missing make-box artifacts under {box}")
    v = _variant_cfg(cfg, variant)
    yaml_path = out / "umbrella_hybrid.yaml"
    _write_hybrid_yaml(
        template=example / "yaml" / "umbrella_nc_tip3.yaml",
        out_yaml=yaml_path,
        solvent=sol,
        box_psf=psf,
        box_pdb=pdb,
        output_dir=out,
        box_size=float(cfg.get("box_size", 30.0)),
        variant=v,
    )
    move_with = _amm1_move_with(psf)
    _uv_run(
        repo,
        [
            "mmml",
            "umbrella-sample",
            "--config",
            str(yaml_path),
            "--output-dir",
            str(out),
            "--move-with",
            move_with,
            "--overwrite",
        ],
    )
    return {
        "solvent": sol,
        "variant": variant,
        "move_with": move_with,
        "summary": str(out / "umbrella_summary.json"),
        "snapshots": str(out / "umbrella_snapshots.npz"),
    }


def job_adumb_gas(repo: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    use_npz = bool((cfg.get("adumb") or {}).get("use_npz_pdb", True))
    env = os.environ.copy()
    env["ARTIFACTS_DIR"] = str(out.parent.resolve())  # unused; we pass --output-dir
    cmd = [
        "uv",
        "run",
        "mmml",
        "md-system",
        "--config",
        str(example / "yaml" / "adumb_nc_distance.yaml"),
        "--output-dir",
        str(out),
    ]
    if use_npz:
        solute = out / "solute_amm1_ch3cl.pdb"
        _uv_run(repo, ["python", "examples/m/07_export_solute_pdb.py", "-o", str(solute)])
        cmd += [
            "--composition",
            str(solute),
            "--from-pdb",
            str(solute),
            "--no-packmol",
            "--charmm-sd-steps",
            "0",
            "--charmm-abnr-steps",
            "0",
            "--no-monomer-physnet-mini",
        ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True, env=env)
    adumb = out / "ADUMB-WUNI.DAT"
    if not adumb.is_file() and not (out / "adumb-wuni.dat").is_file():
        raise FileNotFoundError(f"missing ADUMB-WUNI.DAT under {out}")
    return {"adumb_wuni": str(adumb if adumb.is_file() else out / "adumb-wuni.dat")}


def job_adumb_sol(
    repo: Path, cfg: dict[str, Any], out: Path, solvent: str
) -> dict[str, Any]:
    example = repo / str(cfg.get("example_dir", "examples/m"))
    sol = solvent.lower()
    yaml_cfg = example / "yaml" / f"adumb_nc_distance_{sol}.yaml"
    if not yaml_cfg.is_file():
        raise FileNotFoundError(yaml_cfg)
    use_npz = bool((cfg.get("adumb") or {}).get("use_npz_pdb", True))
    cmd = [
        "uv",
        "run",
        "mmml",
        "md-system",
        "--config",
        str(yaml_cfg),
        "--output-dir",
        str(out),
    ]
    if use_npz:
        solute = out / "solute_amm1_ch3cl.pdb"
        _uv_run(repo, ["python", "examples/m/07_export_solute_pdb.py", "-o", str(solute)])
        n = int((cfg.get("make_boxes") or {}).get("n_solvent", 12))
        resi = _SOLVENT_RESI[sol]
        cmd += ["--composition", f"{solute}:1,{resi}:{n}"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)
    adumb = out / "ADUMB-WUNI.DAT"
    if not adumb.is_file() and not (out / "adumb-wuni.dat").is_file():
        raise FileNotFoundError(f"missing ADUMB-WUNI.DAT under {out}")
    return {
        "solvent": sol,
        "adumb_wuni": str(adumb if adumb.is_file() else out / "adumb-wuni.dat"),
    }


def job_mbar(repo: Path, run_dir: Path, out: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    _uv_run(
        repo,
        [
            "mmml",
            "umbrella-mbar",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(out),
        ],
    )
    return {"run_dir": str(run_dir), "mbar_dir": str(out)}


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    repo = args.repo_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _setup_env(repo, cfg)

    # Campaign artifact root: parent of job-specific dirs for box lookup.
    output_root = str(cfg.get("output_root", "artifacts/nh3_ch3cl_reaction_path"))
    artifact_root = (repo / output_root).resolve()

    t0 = time.time()
    payload: dict[str, Any] = {
        "job": args.job,
        "completed": False,
        "error": "",
        "output_dir": str(out),
        "config": str(Path(args.config).resolve()),
        "basin": args.basin,
        "solvent": args.solvent,
        "variant": args.variant,
    }
    try:
        if args.job == "endpoints":
            payload.update(job_endpoints(repo, cfg, out))
        elif args.job == "make_boxes":
            payload.update(job_make_boxes(repo, cfg, out))
        elif args.job == "neb":
            payload.update(job_neb(repo, cfg, out))
        elif args.job == "dmc":
            if not args.basin:
                raise ValueError("--basin required for dmc")
            payload.update(job_dmc(repo, cfg, out, args.basin))
        elif args.job == "umbrella_gas":
            if not args.variant:
                raise ValueError("--variant required for umbrella_gas")
            payload.update(job_umbrella_gas(repo, cfg, out, args.variant))
        elif args.job == "umbrella_sol":
            if not args.solvent or not args.variant:
                raise ValueError("--solvent and --variant required for umbrella_sol")
            payload.update(
                job_umbrella_sol(
                    repo,
                    cfg,
                    out,
                    solvent=args.solvent,
                    variant=args.variant,
                    artifact_root=artifact_root,
                )
            )
        elif args.job == "adumb_gas":
            payload.update(job_adumb_gas(repo, cfg, out))
        elif args.job == "adumb_sol":
            if not args.solvent:
                raise ValueError("--solvent required for adumb_sol")
            payload.update(job_adumb_sol(repo, cfg, out, args.solvent))
        elif args.job == "mbar":
            if args.run_dir is None:
                raise ValueError("--run-dir required for mbar")
            payload.update(job_mbar(repo, args.run_dir.resolve(), out))
        else:
            raise ValueError(f"unknown job {args.job}")
        payload["completed"] = True
    except Exception as exc:  # noqa: BLE001 — status.json must capture failures
        payload["completed"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        print(payload["traceback"], file=sys.stderr, flush=True)
        payload["elapsed_seconds"] = time.time() - t0
        _write_status(args.status, payload)
        return 1

    payload["elapsed_seconds"] = time.time() - t0
    _write_status(args.status, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
