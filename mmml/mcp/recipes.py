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
    steps_per_recording: int = 1,
    dry_run: bool,
) -> CommandResult:
    from ase.io.trajectory import Trajectory

    from mmml.spectra.spectra_md import (
        autocorrelation,
        compute_magnetic_dipoles,
        correlation_to_spectrum,
        cross_correlation,
        dipole_fluctuation_ir_spectrum,
    )

    out_dir = run_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "classical_cgenff_ir",
        str(traj.relative_to(run_dir)),
        f"composition={composition}",
        f"dt_fs={dt_fs}",
        f"steps_per_recording={steps_per_recording}",
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
    mu_fluct = (dipoles - dipoles.mean(axis=0)).astype(np.float32)
    frame_dt_fs = float(dt_fs) * max(1, int(steps_per_recording))
    freq_cm, ir_spec = dipole_fluctuation_ir_spectrum(dipoles, frame_dt_fs)
    acf = autocorrelation(mu_fluct)
    ccf = cross_correlation(mu_fluct, mag)
    _, vcd_spec = correlation_to_spectrum(ccf, frame_dt_fs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    mask = (freq_cm >= 200) & (freq_cm <= 4000)
    ax[0].plot(freq_cm[mask], ir_spec[mask])
    ax[0].set_ylabel("IR (arb.)")
    ax[0].set_ylim(bottom=0)
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


# NIST gas-phase DCM (JCAMP C75092) — reference sticks for smoke harmonic overlay.
_NIST_DCM_IR_STICKS: list[tuple[float, float]] = [
    (3019.0, 1.0),
    (2996.0, 0.85),
    (1575.0, 0.35),
    (1470.0, 0.25),
    (1150.0, 0.2),
    (948.0, 0.15),
    (748.0, 0.9),
    (707.0, 0.75),
]


def _monomer_atom_count(composition: str) -> int:
    specs = _parse_composition(composition)
    if not specs:
        raise ValueError(f"invalid composition: {composition!r}")
    name, _ = specs[0]
    monomer = _CGENFF_MONOMER_CHARGES.get(name)
    if monomer is None:
        raise ValueError(f"no monomer template for {name}")
    return int(monomer.shape[0])


def _resolve_dipole_checkpoint(run_dir: Path, mode_cfg: dict[str, Any]) -> Path | None:
    env = __import__("os").environ.get("MMML_DIPOLE_CKPT", "").strip()
    if env:
        path = Path(env).resolve()
        return path if path.is_file() or path.is_dir() else None
    rel = mode_cfg.get("dipole_checkpoint")
    if rel:
        for base in (run_dir, repo_root()):
            cand = (base / rel).resolve()
            if cand.is_file() or cand.is_dir():
                return cand
    for pattern in (
        "ckpts/dcm_dipole_smoke/dcm_dipole_smoke/epoch-*",
        "ckpts/dcm_dipole_smoke/**/params*.json",
        "ckpts/**/params*.json",
    ):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[-1] if hits[-1].is_file() else hits[-1].parent
    return None


def _load_physnet_calc(ckpt: Path, n_atoms: int):
    from mmml.interfaces.calculators.checkpoint_loading import (
        create_calculator_from_checkpoint,
    )

    calc = create_calculator_from_checkpoint(ckpt)
    model = getattr(calc, "_mmml_physnet_model", None)
    if model is not None and int(model.max_padded_atoms) < n_atoms:
        raise ValueError(
            f"checkpoint pads {model.max_padded_atoms} atoms but trajectory has {n_atoms}"
        )
    return calc


def _fd_hessian_forces(calc, atoms, *, delta: float = 0.005) -> np.ndarray:
    """Central finite-difference Hessian from ASE forces (eV/Å² flat 3N×3N)."""
    import ase

    n_atoms = len(atoms)
    ndof = 3 * n_atoms
    pos0 = atoms.get_positions().copy()
    work = atoms.copy()
    work.calc = calc
    hess = np.zeros((ndof, ndof), dtype=np.float64)

    for i in range(n_atoms):
        for a in range(3):
            row = 3 * i + a
            for sign in (-1.0, 1.0):
                disp = np.zeros((n_atoms, 3), dtype=np.float64)
                disp[i, a] = sign * delta
                work.set_positions(pos0 + disp)
                forces = np.asarray(work.get_forces(), dtype=np.float64).reshape(-1)
                hess[row] += -sign * forces / (2.0 * delta)
    work.set_positions(pos0)
    hess = 0.5 * (hess + hess.T)
    return hess


def _pointcharge_apt(charges: np.ndarray) -> np.ndarray:
    """APT for μ = Σ q_i r_i  → shape (3, N, 3)."""
    n_atoms = charges.shape[0]
    apt = np.zeros((3, n_atoms, 3), dtype=np.float64)
    for i, q in enumerate(charges):
        apt[:, i, :] = np.eye(3) * q
    return apt


def _stage_ir_harmonic_pointcharge(
    run_dir: Path,
    traj: Path,
    *,
    composition: str,
    ckpt: Path,
    dry_run: bool,
) -> CommandResult:
    from ase import Atoms
    from ase.data import atomic_masses as ASE_ATOMIC_MASSES
    from ase.io.trajectory import Trajectory

    from mmml.models.efield.calc_spectra import broaden, compute_ir, compute_normal_modes

    out_dir = run_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_mono = _monomer_atom_count(composition)
    cmd = ["harmonic_pointcharge", str(traj), f"monomer_atoms={n_mono}"]
    if dry_run:
        return CommandResult(
            command=cmd,
            cwd=str(run_dir),
            returncode=0,
            stdout="(dry run — harmonic point-charge IR not executed)",
            stderr="",
        )

    frames = list(Trajectory(str(traj)))
    atoms = frames[0][:n_mono].copy()
    charges = _classical_cgenff_charges(len(frames[0]), composition)[:n_mono]
    calc = _load_physnet_calc(ckpt, len(frames[0]))

    hess_flat = _fd_hessian_forces(calc, atoms)
    masses = ASE_ATOMIC_MASSES[atoms.get_atomic_numbers()]
    freqs, _, evec_cart = compute_normal_modes(
        hess_flat.reshape(len(atoms), 3, len(atoms), 3), masses
    )
    apt = _pointcharge_apt(charges)
    ir_int, _ = compute_ir(apt, evec_cart)

    freq_ax = np.linspace(200.0, 4000.0, 2000)
    gamma = 12.0
    ml_spec = broaden(freq_ax, freqs, ir_int, gamma)
    nist_spec = np.zeros_like(freq_ax)
    for nu, wt in _NIST_DCM_IR_STICKS:
        nist_spec += wt * np.exp(-0.5 * ((freq_ax - nu) / gamma) ** 2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    mask = (freq_ax >= 200) & (freq_ax <= 4000)
    ax.plot(freq_ax[mask], ml_spec[mask], label="ML Hessian + CGENFF μ")
    ax.plot(freq_ax[mask], nist_spec[mask], ls="--", alpha=0.8, label="NIST DCM (C75092)")
    ax.set_xlabel("cm$^{-1}$")
    ax.set_ylabel("IR (arb.)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "harmonic_ir.png", dpi=120)
    plt.close(fig)

    active = np.where(np.abs(freqs) > 200.0)[0]
    stick_freqs = freqs[active]
    stick_ir = ir_int[active]
    np.savez(
        out_dir / "harmonic_ir.npz",
        freq_axis=freq_ax,
        ml_broadened=ml_spec,
        nist_broadened=nist_spec,
        stick_frequencies=stick_freqs,
        stick_intensities=stick_ir,
        charges=charges,
        method="harmonic_pointcharge_cgenff",
    )
    top = sorted(zip(stick_freqs, stick_ir), key=lambda x: -x[1])[:6]
    summary = ", ".join(f"{nu:.0f} ({inten:.2e})" for nu, inten in top)
    return CommandResult(
        command=cmd,
        cwd=str(run_dir),
        returncode=0,
        stdout=(
            f"harmonic point-charge IR -> {out_dir}/harmonic_ir.png\n"
            f"top modes (cm⁻¹): {summary}"
        ),
        stderr="",
    )


def _predict_dipoles_physnet(
    frames,
    dipole_ckpt: Path,
    *,
    batch_size: int = 1,
) -> np.ndarray:
    """PhysNet dipole inference (delegates to JIT implementation)."""
    from mmml.mcp.ir_comparison import predict_dipoles_physnet_jit

    _ = batch_size  # kept for API compatibility
    return predict_dipoles_physnet_jit(frames, dipole_ckpt)


def _stage_ir_ml_dipole(
    run_dir: Path,
    traj: Path,
    *,
    dipole_ckpt: Path,
    dt_fs: float,
    steps_per_recording: int,
    max_frames: int = 500,
    dry_run: bool,
) -> CommandResult:
    from ase.io.trajectory import Trajectory

    from mmml.spectra.spectra_md import (
        autocorrelation,
        correlation_to_spectrum,
        dipole_fluctuation_ir_spectrum,
    )

    out_dir = run_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ml_dipole_ir",
        str(traj),
        f"checkpoint={dipole_ckpt}",
        f"max_frames={max_frames}",
    ]
    if dry_run:
        return CommandResult(
            command=cmd,
            cwd=str(run_dir),
            returncode=0,
            stdout="(dry run — ML dipole IR not executed)",
            stderr="",
        )

    frames = list(Trajectory(str(traj)))
    stride = max(1, len(frames) // max_frames)
    frames = frames[::stride][:max_frames]

    dipoles = _predict_dipoles_physnet(frames, dipole_ckpt, batch_size=8)

    frame_dt_fs = float(dt_fs) * max(1, int(steps_per_recording)) * stride
    freq_cm, ir_spec = dipole_fluctuation_ir_spectrum(dipoles, frame_dt_fs)
    mu_fluct = (dipoles - dipoles.mean(axis=0)).astype(np.float32)
    acf = autocorrelation(mu_fluct)
    _, vcd_spec = correlation_to_spectrum(acf, frame_dt_fs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    mask = (freq_cm >= 200) & (freq_cm <= 4000)
    ax.plot(freq_cm[mask], ir_spec[mask])
    ax.set_xlabel("cm$^{-1}$")
    ax.set_ylabel("IR (arb., ML μ)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "ir_ml_dipole.png", dpi=120)
    plt.close(fig)

    np.savez(
        out_dir / "ml_dipole_spectra.npz",
        freq_cm=freq_cm,
        ir=ir_spec,
        dipoles=dipoles,
        stride=stride,
        frame_dt_fs=frame_dt_fs,
        checkpoint=str(dipole_ckpt),
        method="physnet_charges_dipole",
    )
    mu_norm = np.linalg.norm(dipoles, axis=1)
    return CommandResult(
        command=cmd,
        cwd=str(run_dir),
        returncode=0,
        stdout=(
            f"ML dipole IR -> {out_dir}/ir_ml_dipole.png "
            f"({len(frames)} frames, stride={stride})\n"
            f"|mu| range: {mu_norm.min():.4f} – {mu_norm.max():.4f} e·Å"
        ),
        stderr="",
    )


def _recipe_stage_names(recipe_data: dict[str, Any] | None) -> tuple[str, ...]:
    if recipe_data and isinstance(recipe_data.get("stages"), dict):
        return tuple(recipe_data["stages"].keys())
    return (
        "configure",
        "structures",
        "qm",
        "train",
        "train_dipole",
        "md",
        "ir",
    )


def _write_build_configs(
    run_dir: Path,
    mode_cfg: dict[str, Any],
) -> list[Path]:
    """Write liquid-box params + hybrid MD YAML templates for build_smoke."""
    configs = run_dir / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    box_body = {
        "composition": mode_cfg.get("box_composition", "DCM:12"),
        "output_dir": "boxes/liquid",
        "profile": mode_cfg.get("box_profile", "standard"),
        "box_size": float(mode_cfg.get("box_size", 24.0)),
        "target_density_g_cm3": float(mode_cfg.get("target_density_g_cm3", 1.326)),
        "spacing": float(mode_cfg.get("spacing", 4.0)),
        "seed": int(mode_cfg.get("seed", 42)),
    }
    box_path = configs / "build_box.yaml"
    box_path.write_text(
        "# Generated by mmml MCP configure_run (build_smoke)\n"
        + yaml.safe_dump(box_body, sort_keys=False),
        encoding="utf-8",
    )
    written.append(box_path)

    ckpt_rel = mode_cfg.get("checkpoint", "examples/ckpts_json/DESdimers_params.json")
    ckpt = (repo_root() / ckpt_rel).resolve()
    if not ckpt.is_file():
        ckpt = default_checkpoint()
    composition = mode_cfg.get("composition", "DCM:2")
    ps = float(mode_cfg.get("md_ps", 0.05))
    dt_fs = float(mode_cfg.get("dt_fs", 0.5))
    setup = mode_cfg.get("hybrid_setup", "free_nve")

    examples_dir = Path(__file__).resolve().parent / "examples"
    for backend, cfg_name, run_name in (
        ("ase", "hybrid_ase.yaml", "hybrid_ase"),
        ("jaxmd", "hybrid_jaxmd.yaml", "hybrid_jaxmd"),
        ("pycharmm", "hybrid_pycharmm.yaml", "hybrid_pycharmm"),
    ):
        template_path = examples_dir / cfg_name
        if template_path.is_file():
            body = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        else:
            body = {}
        defaults = body.get("defaults") or {}
        defaults.update(
            {
                "composition": composition,
                "checkpoint": str(ckpt),
                "dt_fs": dt_fs,
            }
        )
        if mode_cfg.get("hybrid_box_size") is not None:
            defaults["box_size"] = float(mode_cfg["hybrid_box_size"])
        if mode_cfg.get("packmol_radius") is not None:
            defaults["packmol_radius"] = float(mode_cfg["packmol_radius"])
        if mode_cfg.get("packmol_tolerance") is not None:
            defaults["packmol_tolerance"] = float(mode_cfg["packmol_tolerance"])
        body["defaults"] = defaults
        runs = body.get("runs") or {}
        if run_name in runs:
            runs[run_name]["ps"] = ps
            runs[run_name]["setup"] = setup
        body["runs"] = runs
        body["campaign_output"] = "results"
        out_path = configs / cfg_name
        out_path.write_text(
            f"# Generated by mmml MCP configure_run\n{yaml.safe_dump(body, sort_keys=False)}",
            encoding="utf-8",
        )
        written.append(out_path)

    return written


def _stage_make_res(
    run_dir: Path,
    mode_cfg: dict[str, Any],
    *,
    dry_run: bool,
) -> CommandResult:
    residue = str(mode_cfg.get("residue", "DCM"))
    work = run_dir / "residue"
    work.mkdir(parents=True, exist_ok=True)
    return run_mmml(
        "make-res",
        ["--res", residue, "--skip-energy-show"],
        run_dir=work,
        dry_run=dry_run,
        timeout_s=600,
    )


def _stage_box_build(
    run_dir: Path,
    mode_cfg: dict[str, Any],
    *,
    dry_run: bool,
    background: bool,
) -> CommandResult:
    cfg_path = run_dir / "configs" / "build_box.yaml"
    if cfg_path.is_file():
        box_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        box_cfg = mode_cfg
    composition = str(box_cfg.get("composition", "DCM:12"))
    out_rel = str(box_cfg.get("output_dir", "boxes/liquid"))
    profile = str(box_cfg.get("profile", "standard"))
    args = [
        "--composition",
        composition,
        "-o",
        out_rel,
        "--profile",
        profile,
        "--quiet",
    ]
    if box_cfg.get("box_size") is not None:
        args.extend(["--box-size", str(box_cfg["box_size"])])
    if box_cfg.get("target_density_g_cm3") is not None:
        args.extend(["--target-density-g-cm3", str(box_cfg["target_density_g_cm3"])])
    if box_cfg.get("spacing") is not None:
        args.extend(["--spacing", str(box_cfg["spacing"])])
    if box_cfg.get("seed") is not None:
        args.extend(["--seed", str(int(box_cfg["seed"]))])
    return run_mmml(
        "liquid-box",
        args,
        run_dir=run_dir,
        dry_run=dry_run,
        background=background,
        timeout_s=None if background else 7200,
    )


def _stage_hybrid_md(
    run_dir: Path,
    backend: str,
    *,
    dry_run: bool,
    background: bool,
) -> CommandResult:
    cfg_map = {
        "ase": "hybrid_ase.yaml",
        "jaxmd": "hybrid_jaxmd.yaml",
        "pycharmm": "hybrid_pycharmm.yaml",
    }
    cfg_name = cfg_map.get(backend)
    if not cfg_name:
        raise ValueError(f"unknown hybrid backend: {backend}")
    cfg_path = run_dir / "configs" / cfg_name
    if not cfg_path.is_file():
        raise FileNotFoundError(f"hybrid config not found: {cfg_path}; run configure first")
    return run_mmml(
        "md-system",
        ["--config", f"configs/{cfg_name}", "--run-all"],
        run_dir=run_dir,
        dry_run=dry_run,
        background=background,
        timeout_s=None if background else 3600,
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
    recipe_data: dict[str, Any] | None = load_recipe(recipe) if recipe else None
    manifest = RunManifest(
        run_id=run_id,
        recipe=recipe,
        preset=preset,
        cluster=cluster or "local",
    )
    for stage_name in _recipe_stage_names(recipe_data):
        manifest.stage(stage_name)

    written: list[str] = []
    if recipe and recipe_data:
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

        recipe_name = recipe_data.get("name", recipe)
        if recipe_name == "build_smoke":
            for p in _write_build_configs(run_dir, mode_cfg):
                written.append(str(p))
            manifest.stage("configure").state = "done"
            manifest.stage("configure").artifacts = written
        else:
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
    ps = float(mode_cfg.get("md_ps", 20.0))
    dt_fs = float(mode_cfg.get("dt_fs", 0.1))
    steps_per_recording = int(mode_cfg.get("steps_per_recording", 1))
    body = {
        "defaults": {
            "composition": composition,
            "checkpoint": str(ckpt),
            "box_size": 28.0,
            "dt_fs": dt_fs,
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
                "extra_args": [
                    "--steps-per-recording",
                    str(steps_per_recording),
                ],
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
            result = _stage_ir(
                run_dir,
                dry_run=dry_run,
                background=background,
                mode_cfg=mode_cfg,
            )
        elif stage == "train_dipole":
            result = _stage_train_dipole(
                run_dir,
                mode_cfg=mode_cfg,
                dry_run=dry_run,
                background=background,
            )
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
        elif stage == "make_res":
            result = _stage_make_res(run_dir, mode_cfg, dry_run=dry_run)
        elif stage == "box_build":
            result = _stage_box_build(
                run_dir, mode_cfg, dry_run=dry_run, background=background
            )
        elif stage == "hybrid_md_ase":
            result = _stage_hybrid_md(
                run_dir, "ase", dry_run=dry_run, background=background
            )
        elif stage == "hybrid_md_jaxmd":
            result = _stage_hybrid_md(
                run_dir, "jaxmd", dry_run=dry_run, background=background
            )
        elif stage == "hybrid_md_pycharmm":
            result = _stage_hybrid_md(
                run_dir, "pycharmm", dry_run=dry_run, background=background
            )
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


def _stage_train_dipole(
    run_dir: Path,
    *,
    mode_cfg: dict[str, Any],
    dry_run: bool,
    background: bool,
) -> CommandResult:
    """Mini fix-and-split + physnet-train with charges on bundled DCM MP2 NPZ."""
    data_src = mode_cfg.get("dipole_training_npz", "dcm_mp2_psf_order.npz")
    src = (repo_root() / data_src).resolve()
    if not src.is_file():
        return CommandResult(
            command=["train_dipole"],
            cwd=str(run_dir),
            returncode=1,
            stdout="",
            stderr=f"dipole training NPZ not found: {src}",
        )

    subset_n = int(mode_cfg.get("dipole_training_subset", 800))
    splits = run_dir / "splits"
    ckpt_parent = run_dir / "ckpts" / "dcm_dipole_smoke"
    train_yaml = repo_root() / "mmml" / "mcp" / "dipole_smoke_train.yaml"

    if dry_run:
        return CommandResult(
            command=["train_dipole", str(src), f"subset={subset_n}"],
            cwd=str(run_dir),
            returncode=0,
            stdout="(dry run — dipole train not executed)",
            stderr="",
        )

    subset_path = run_dir / "data" / "dcm_mp2_subset.npz"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    if not subset_path.is_file():
        raw = np.load(src, allow_pickle=True)
        n = min(subset_n, int(raw["R"].shape[0]))
        idx = np.arange(n)
        np.savez(
            subset_path,
            **{k: np.asarray(raw[k])[idx] for k in raw.files},
        )

    split_res = run_mmml(
        "fix-and-split",
        [
            "--efd",
            str(subset_path.relative_to(run_dir)),
            "-o",
            "splits",
            "--dipole-in",
            "debye",
            "--dipole-out",
            "e-angstrom",
        ],
        run_dir=run_dir,
        dry_run=False,
        background=False,
        timeout_s=600,
    )
    if split_res.returncode != 0:
        return split_res

    train_res = run_mmml(
        "physnet-train",
        ["--config", str(train_yaml), "--ckpt-dir", str(ckpt_parent)],
        run_dir=run_dir,
        dry_run=False,
        background=background,
        timeout_s=None if background else 7200,
    )
    if train_res.returncode != 0:
        return train_res

    test_npz = splits / "energies_forces_dipoles_test.npz"
    eval_out = run_dir / "eval" / "dipole_smoke"
    ckpt_for_eval = ckpt_parent
    json_ckpts = sorted(ckpt_parent.glob("params*.json"))
    if json_ckpts:
        ckpt_for_eval = json_ckpts[-1]
    if test_npz.is_file():
        eval_res = run_mmml(
            "physnet-evaluate",
            [
                "--checkpoint",
                str(
                    ckpt_for_eval.relative_to(run_dir)
                    if ckpt_for_eval.is_relative_to(run_dir)
                    else ckpt_for_eval
                ),
                "--data",
                str(test_npz.relative_to(run_dir)),
                "-o",
                str(eval_out.relative_to(run_dir)),
                "--natoms",
                "10",
                "--batch-size",
                "16",
            ],
            run_dir=run_dir,
            dry_run=False,
            background=False,
            timeout_s=1800,
        )
        stdout = (
            f"{train_res.stdout}\n\nphysnet-evaluate -> {eval_out}\n{eval_res.stdout}"
        )
        return CommandResult(
            command=train_res.command + ["+", "physnet-evaluate"],
            cwd=str(run_dir),
            returncode=eval_res.returncode,
            stdout=stdout,
            stderr=train_res.stderr + eval_res.stderr,
        )

    return train_res


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
    mode_cfg: dict[str, Any] | None = None,
) -> CommandResult:
    mode_cfg = mode_cfg or {}
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
    dipole_ckpt = _resolve_dipole_checkpoint(run_dir, mode_cfg) or ckpt
    out_dir = run_dir / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_cfg_path = run_dir / "configs" / "md_smoke.yaml"
    dt_fs = 0.1
    steps_per_recording = 1
    composition = "DCM:2"
    if md_cfg_path.is_file():
        md_cfg = yaml.safe_load(md_cfg_path.read_text(encoding="utf-8")) or {}
        defaults = md_cfg.get("defaults") or {}
        dt_fs = float(defaults.get("dt_fs", dt_fs))
        comp = defaults.get("composition")
        if comp:
            composition = str(comp)
        ckpt_default = defaults.get("checkpoint")
        if ckpt_default:
            ckpt = Path(ckpt_default).resolve()
        extra = (md_cfg.get("runs") or {}).get("jaxmd_smoke", {}).get("extra_args") or []
        for i, arg in enumerate(extra):
            if arg == "--steps-per-recording" and i + 1 < len(extra):
                steps_per_recording = int(extra[i + 1])
                break

    ir_methods = mode_cfg.get("ir_methods")
    if ir_methods is None:
        ir_methods = ["classical_cgenff", "harmonic_pointcharge"]
        if _checkpoint_charges_enabled(dipole_ckpt):
            ir_methods.append("ml_dipole")

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    last_cmd: list[str] = ["ir"]
    rc = 0

    if "classical_cgenff" in ir_methods and not _checkpoint_charges_enabled(ckpt):
        res = _stage_ir_classical_cgenff(
            run_dir,
            traj,
            composition=composition,
            dt_fs=dt_fs,
            steps_per_recording=steps_per_recording,
            dry_run=dry_run,
        )
        stdout_parts.append(res.stdout)
        stderr_parts.append(res.stderr)
        last_cmd = res.command
        rc = max(rc, res.returncode)

    if "harmonic_pointcharge" in ir_methods:
        res = _stage_ir_harmonic_pointcharge(
            run_dir,
            traj,
            composition=composition,
            ckpt=ckpt,
            dry_run=dry_run,
        )
        stdout_parts.append(res.stdout)
        stderr_parts.append(res.stderr)
        last_cmd = res.command
        rc = max(rc, res.returncode)

    if "ml_dipole" in ir_methods and _checkpoint_charges_enabled(dipole_ckpt):
        res = _stage_ir_ml_dipole(
            run_dir,
            traj,
            dipole_ckpt=dipole_ckpt,
            dt_fs=dt_fs,
            steps_per_recording=steps_per_recording,
            max_frames=int(mode_cfg.get("ir_ml_max_frames", 500)),
            dry_run=dry_run,
        )
        stdout_parts.append(res.stdout)
        stderr_parts.append(res.stderr)
        last_cmd = res.command
        rc = max(rc, res.returncode)
    elif "ml_dipole" in ir_methods:
        stdout_parts.append(
            "ml_dipole IR skipped: no charges=True checkpoint "
            f"(set dipole_checkpoint or run train_dipole stage)"
        )

    if mode_cfg.get("ir_comparison", True) and dipole_ckpt and _checkpoint_charges_enabled(
        dipole_ckpt
    ):
        try:
            from mmml.mcp.ir_comparison import generate_ir_comparison_figure

            meta = generate_ir_comparison_figure(
                run_dir,
                traj,
                dipole_ckpt,
                dt_fs=dt_fs,
                steps_per_recording=steps_per_recording,
                stride=int(mode_cfg.get("ir_comparison_stride", 1)),
                max_frames=mode_cfg.get("ir_comparison_max_frames"),
            )
            stdout_parts.append(
                "publication IR comparison -> "
                f"{meta['outputs']['comparison_png']}"
            )
        except Exception as exc:
            stderr_parts.append(f"ir_comparison failed: {exc}")
            rc = max(rc, 1)

    if _checkpoint_charges_enabled(ckpt) and "correlation" in ir_methods:
        args = [
            "--trajectory",
            str(traj.relative_to(run_dir) if traj.is_relative_to(run_dir) else traj),
            "--params",
            str(ckpt),
            "--output-dir",
            str(out_dir.relative_to(run_dir)),
            "--method",
            "correlation",
            "--recompute-dipole",
        ]
        return run_console_script(
            "mmml-spectra-md",
            args,
            run_dir=run_dir,
            dry_run=dry_run,
            background=background,
            timeout_s=None if background else 1800,
        )

    if not stdout_parts and not dry_run:
        return CommandResult(
            command=last_cmd,
            cwd=str(run_dir),
            returncode=1,
            stdout="",
            stderr="no IR methods produced output",
        )

    return CommandResult(
        command=last_cmd,
        cwd=str(run_dir),
        returncode=rc,
        stdout="\n\n".join(p for p in stdout_parts if p),
        stderr="\n".join(p for p in stderr_parts if p),
    )
