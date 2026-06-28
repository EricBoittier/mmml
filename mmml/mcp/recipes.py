"""Recipe loading and stage execution for MCP runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mmml.cli.configure_presets import PRESET_BY_KEY, apply_preset
from mmml.mcp.env import default_checkpoint, ensure_run_dir, recipes_dir, repo_root
from mmml.mcp.manifest import RunManifest, save_manifest
from mmml.mcp.runner import CommandResult, run_console_script, run_mmml

# CGENFF partial charges (e) for smoke IR when ML checkpoint has charges=False.
_CGENFF_MONOMER_CHARGES: dict[str, np.ndarray] = {
    "DCM": np.array([-0.018, 0.090, 0.090, -0.081, -0.081], dtype=np.float64),
}


def _checkpoint_charges_enabled(ckpt: Path) -> bool:
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    cfg = data.get("config") or {}
    if "charges" in cfg:
        return bool(cfg["charges"])
    return True


def _parse_composition(composition: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for part in composition.split(","):
        part = part.strip()
        if not part:
            continue
        name, count = part.split(":")
        out.append((name.strip().upper(), int(count)))
    return out


def _classical_cgenff_charges(n_atoms: int, composition: str) -> np.ndarray:
    specs = _parse_composition(composition)
    charges: list[np.ndarray] = []
    for name, count in specs:
        monomer = _CGENFF_MONOMER_CHARGES.get(name)
        if monomer is None:
            raise ValueError(f"no CGENFF charge template for residue {name}")
        charges.extend([monomer] * count)
    flat = np.concatenate(charges) if charges else np.array([], dtype=np.float64)
    if flat.shape[0] != n_atoms:
        raise ValueError(
            f"charge template length {flat.shape[0]} != trajectory atoms {n_atoms} "
            f"(composition={composition!r})"
        )
    return flat


def _stage_ir_classical_cgenff(
    run_dir: Path,
    traj: Path,
    *,
    composition: str,
    dt_fs: float,
    dry_run: bool,
) -> CommandResult:
    from ase.io.trajectory import Trajectory

    from mmml.spectra.spectra_md import (
        autocorrelation,
        compute_magnetic_dipoles,
        correlation_to_spectrum,
        cross_correlation,
    )

    out_dir = run_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "classical_cgenff_ir",
        str(traj.relative_to(run_dir)),
        f"composition={composition}",
        f"dt_fs={dt_fs}",
    ]
    if dry_run:
        return CommandResult(
            command=cmd,
            cwd=str(run_dir),
            returncode=0,
            stdout="(dry run — classical CGENFF IR not executed)",
            stderr="",
        )

    frames = list(Trajectory(str(traj)))
    charges = _classical_cgenff_charges(len(frames[0]), composition)
    positions = np.stack([f.get_positions() for f in frames], axis=0).astype(np.float64)
    velocities = np.stack(
        [np.nan_to_num(f.get_velocities()) for f in frames], axis=0
    ).astype(np.float64)
    dipoles = np.sum(positions * charges[None, :, None], axis=1)
    charges_t = np.broadcast_to(charges, (len(frames), len(charges))).astype(np.float32)
    mag = compute_magnetic_dipoles(
        positions.astype(np.float32), velocities.astype(np.float32), charges_t
    )
    acf = autocorrelation(dipoles.astype(np.float32))
    ccf = cross_correlation(dipoles.astype(np.float32), mag)
    freq_cm, ir_spec = correlation_to_spectrum(acf, dt_fs)
    _, vcd_spec = correlation_to_spectrum(ccf, dt_fs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    mask = (freq_cm >= 200) & (freq_cm <= 4000)
    ax[0].plot(freq_cm[mask], ir_spec[mask])
    ax[0].set_ylabel("IR (arb.)")
    ax[1].plot(freq_cm[mask], vcd_spec[mask])
    ax[1].set_ylabel("VCD (arb.)")
    ax[1].set_xlabel("cm$^{-1}$")
    fig.tight_layout()
    fig.savefig(out_dir / "ir.png", dpi=120)
    plt.close(fig)
    np.savez(
        out_dir / "correlation_spectra.npz",
        freq_cm=freq_cm,
        ir=ir_spec,
        vcd=vcd_spec,
        acf=acf,
        ccf=ccf,
        dipoles=dipoles,
        charges=charges,
        method="classical_cgenff_dipole",
    )
    mu_norm = np.linalg.norm(dipoles, axis=1)
    return CommandResult(
        command=cmd,
        cwd=str(run_dir),
        returncode=0,
        stdout=(
            f"classical CGENFF dipole IR -> {out_dir}/ir.png\n"
            f"|mu| range: {mu_norm.min():.4f} – {mu_norm.max():.4f}"
        ),
        stderr="",
    )


def list_recipe_names() -> list[str]:
    return sorted(p.stem for p in recipes_dir().glob("*.yaml"))


def load_recipe(name: str) -> dict[str, Any]:
    path = recipes_dir() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"recipe not found: {name}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def configure_run(
    run_id: str,
    *,
    recipe: str | None = "dimer_smoke",
    mode: str = "smoke",
    preset: str | None = None,
    cluster: str | None = None,
) -> dict[str, Any]:
    run_dir = ensure_run_dir(run_id)
    manifest = RunManifest(
        run_id=run_id,
        recipe=recipe,
        preset=preset,
        cluster=cluster or "local",
    )
    for stage_name in ("configure", "structures", "qm", "train", "md", "ir"):
        manifest.stage(stage_name)

    written: list[str] = []
    recipe_data: dict[str, Any] | None = None
    if recipe:
        recipe_data = load_recipe(recipe)
        manifest.recipe = recipe
        mode_cfg = (recipe_data.get("modes") or {}).get(mode) or {}
        preset_key = preset or mode_cfg.get("preset") or (
            (recipe_data.get("stages") or {}).get("configure", {}).get("preset")
        )
        if preset_key:
            if preset_key not in PRESET_BY_KEY:
                raise ValueError(f"unknown preset: {preset_key}")
            out = run_dir / "configs"
            out.mkdir(parents=True, exist_ok=True)
            paths = apply_preset(PRESET_BY_KEY[preset_key], out)
            written.extend(str(p) for p in paths)
            manifest.preset = preset_key
            manifest.stage("configure").state = "done"
            manifest.stage("configure").artifacts = written

        skip = set(mode_cfg.get("skip_stages") or [])
        for name in skip:
            st = manifest.stage(name)
            st.state = "skipped"
            st.meta["reason"] = f"{mode} mode"

        _write_md_smoke_config(run_dir, recipe_data, mode_cfg)
        written.append(str(run_dir / "configs" / "md_smoke.yaml"))

    save_manifest(run_dir, manifest)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest.to_dict(),
        "written_files": written,
        "recipe": recipe,
        "mode": mode,
    }


def _write_md_smoke_config(
    run_dir: Path,
    recipe_data: dict[str, Any] | None,
    mode_cfg: dict[str, Any],
) -> Path:
    configs = run_dir / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    ckpt_rel = mode_cfg.get("checkpoint", "examples/ckpts_json/DESdimers_params.json")
    ckpt = (repo_root() / ckpt_rel).resolve()
    if not ckpt.is_file():
        ckpt = default_checkpoint()
    composition = mode_cfg.get("composition", "DCM:2")
    ps = float(mode_cfg.get("md_ps", 0.5))
    body = {
        "defaults": {
            "composition": composition,
            "checkpoint": str(ckpt),
            "box_size": 28.0,
            "dt_fs": 0.5,
            "temperature": 300.0,
            "seed": 42,
        },
        "campaign_output": str(run_dir / "md"),
        "runs": {
            "jaxmd_smoke": {
                "description": "Short NVT smoke for MCP IR test",
                "backend": "jaxmd",
                "setup": "pbc_nvt",
                "ps": ps,
                "output_dir": "results/jaxmd_smoke",
                "extra_args": ["--steps-per-recording", "50"],
            }
        },
    }
    path = configs / "md_smoke.yaml"
    path.write_text(
        "# Generated by mmml MCP configure_run\n" + yaml.safe_dump(body, sort_keys=False),
        encoding="utf-8",
    )
    return path


def run_recipe_stage(
    run_id: str,
    stage: str,
    *,
    dry_run: bool = False,
    background: bool = False,
    mode: str = "smoke",
) -> dict[str, Any]:
    from mmml.mcp.manifest import load_manifest

    run_dir = ensure_run_dir(run_id)
    manifest = load_manifest(run_dir)
    recipe_name = manifest.recipe or "dimer_smoke"
    recipe_data = load_recipe(recipe_name)
    mode_cfg = (recipe_data.get("modes") or {}).get(mode) or {}
    skip = set(mode_cfg.get("skip_stages") or [])

    if stage in skip:
        rec = manifest.stage(stage)
        rec.state = "skipped"
        save_manifest(run_dir, manifest)
        return {"run_id": run_id, "stage": stage, "state": "skipped", "reason": f"{mode} mode"}

    st = manifest.stage(stage)
    st.state = "running"
    save_manifest(run_dir, manifest)

    result: CommandResult | None = None
    try:
        if stage == "configure":
            result = _stage_configure(run_dir, manifest, dry_run=dry_run)
        elif stage == "md":
            result = run_mmml(
                "md-system",
                ["--config", "configs/md_smoke.yaml", "--run-all"],
                run_dir=run_dir,
                dry_run=dry_run,
                background=background,
                timeout_s=None if background else 3600,
            )
        elif stage == "ir":
            result = _stage_ir(run_dir, dry_run=dry_run, background=background)
        elif stage == "qm":
            result = run_mmml(
                "pyscf-evaluate",
                ["--config", "configs/qm_pipeline/pyscf_evaluate.yaml"],
                run_dir=run_dir,
                dry_run=dry_run,
                background=background,
            )
        elif stage == "train":
            result = run_mmml(
                "fix-and-split",
                ["--config", "configs/qm_pipeline/fix_and_split.yaml"],
                run_dir=run_dir,
                dry_run=dry_run,
                background=background,
            )
            if not dry_run and not background and result.returncode == 0:
                train = run_mmml(
                    "physnet-train",
                    ["--config", "configs/qm_pipeline/physnet_train.yaml"],
                    run_dir=run_dir,
                    background=background,
                )
                result = train
        elif stage == "structures":
            result = CommandResult(
                command=["(manual)"],
                cwd=str(run_dir),
                returncode=0,
                stdout=(
                    "structures stage requires sampled geometries. "
                    "Run: mmml normal-mode-sample or provide structures/sampled.npz"
                ),
                stderr="",
            )
        else:
            raise ValueError(f"unknown stage: {stage}")

        st.command = " ".join(result.command)
        st.log_path = result.log_path
        if result.background:
            st.state = "running"
            st.job_id = str(result.pid)
        elif result.returncode == 0:
            st.state = "done"
        else:
            st.state = "failed"
            st.error = (result.stderr or result.stdout)[-2000:]
    except Exception as exc:
        st.state = "failed"
        st.error = str(exc)
        raise
    finally:
        save_manifest(run_dir, manifest)

    return {
        "run_id": run_id,
        "stage": stage,
        "state": st.state,
        "result": result.to_dict() if result else None,
        "manifest": manifest.to_dict(),
    }


def _stage_configure(
    run_dir: Path,
    manifest: RunManifest,
    *,
    dry_run: bool,
) -> CommandResult:
    if manifest.preset and (run_dir / "configs").is_dir():
        return CommandResult(
            command=["configure", "preset", manifest.preset],
            cwd=str(run_dir),
            returncode=0,
            stdout=f"preset {manifest.preset} already applied",
            stderr="",
        )
    preset = manifest.preset or "qm-physnet-pipeline"
    if preset not in PRESET_BY_KEY:
        raise ValueError(f"unknown preset: {preset}")
    if dry_run:
        return CommandResult(
            command=["apply_preset", preset],
            cwd=str(run_dir),
            returncode=0,
            stdout=f"would apply preset {preset}",
            stderr="",
        )
    out = run_dir / "configs"
    paths = apply_preset(PRESET_BY_KEY[preset], out)
    manifest.stage("configure").artifacts = [str(p) for p in paths]
    return CommandResult(
        command=["apply_preset", preset],
        cwd=str(run_dir),
        returncode=0,
        stdout=json.dumps([str(p) for p in paths], indent=2),
        stderr="",
    )


def _stage_ir(
    run_dir: Path,
    *,
    dry_run: bool,
    background: bool,
) -> CommandResult:
    traj_roots = (run_dir / "md", run_dir / "results")
    traj_candidates: list[Path] = []
    for root in traj_roots:
        if root.is_dir():
            traj_candidates.extend(root.rglob("*.traj"))
            traj_candidates.extend(root.rglob("*.h5"))
    prod = [
        p
        for p in traj_candidates
        if "bfgs_min" not in p.name and not p.name.endswith("_min.traj")
    ]
    traj_candidates = prod or traj_candidates
    if not traj_candidates:
        return CommandResult(
            command=["mmml-spectra-md"],
            cwd=str(run_dir),
            returncode=1,
            stdout="",
            stderr=f"no trajectory under {run_dir}/{{md,results}}; run md stage first",
        )
    traj = sorted(traj_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    ckpt = default_checkpoint()
    out_dir = run_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_cfg_path = run_dir / "configs" / "md_smoke.yaml"
    dt_fs = 0.5
    composition = "DCM:2"
    if md_cfg_path.is_file():
        md_cfg = yaml.safe_load(md_cfg_path.read_text(encoding="utf-8")) or {}
        defaults = md_cfg.get("defaults") or {}
        dt_fs = float(defaults.get("dt_fs", dt_fs))
        comp = defaults.get("composition")
        if comp:
            composition = str(comp)

    if not _checkpoint_charges_enabled(ckpt):
        return _stage_ir_classical_cgenff(
            run_dir,
            traj,
            composition=composition,
            dt_fs=dt_fs,
            dry_run=dry_run,
        )

    args = [
        "--trajectory",
        str(traj.relative_to(run_dir) if traj.is_relative_to(run_dir) else traj),
        "--params",
        str(ckpt),
        "--output-dir",
        str(out_dir.relative_to(run_dir)),
        "--method",
        "correlation",
    ]
    return run_console_script(
        "mmml-spectra-md",
        args,
        run_dir=run_dir,
        dry_run=dry_run,
        background=background,
        timeout_s=None if background else 1800,
    )
