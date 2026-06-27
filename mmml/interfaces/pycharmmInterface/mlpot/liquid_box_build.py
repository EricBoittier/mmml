"""Phase A liquid box build: Packmol → MC → CHARMM MM → certification (no MLpot)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

LiquidBoxProfile = Literal["standard", "dense", "conservative"]
LiquidBoxStatus = Literal["pass", "fail"]

BOX_JSON = "box.json"
REPORT_MD = "REPORT.md"
LIQUID_BOX_PROFILES: tuple[LiquidBoxProfile, ...] = ("standard", "dense", "conservative")


@dataclass
class LiquidBoxBuildResult:
    """Outcome of :func:`run_liquid_box_build`."""

    status: LiquidBoxStatus
    out_dir: Path
    profile: str
    composition: str | None
    n_molecules: int
    n_atoms: int
    box_side_A: float | None = None
    density_g_cm3: float | None = None
    worst_intermonomer_A: float | None = None
    prep_overlap_floor_A: float | None = None
    mm_grms_kcalmol_A: float | None = None
    mc_density_summary: dict[str, Any] | None = None
    geometry_gate_summary: dict[str, Any] | None = None
    model_psf: Path | None = None
    model_crd: Path | None = None
    model_pdb: Path | None = None
    box_json_path: Path | None = None
    report_path: Path | None = None
    message: str = ""
    steps_applied: list[str] = field(default_factory=list)

    def to_box_json(self, args: argparse.Namespace | None = None) -> dict[str, Any]:
        from mmml.utils.intermonomer_geometry import resolve_dynamics_overlap_reference_A

        return {
            "status": self.status,
            "profile": self.profile,
            "composition": self.composition,
            "n_molecules": int(self.n_molecules),
            "n_atoms": int(self.n_atoms),
            "box_side_A": self.box_side_A,
            "density_g_cm3": self.density_g_cm3,
            "worst_intermonomer_A": self.worst_intermonomer_A,
            "prep_overlap_floor_A": self.prep_overlap_floor_A,
            "dynamics_overlap_reference_A": resolve_dynamics_overlap_reference_A(args),
            "mm_grms_kcalmol_A": self.mm_grms_kcalmol_A,
            "mc_density": self.mc_density_summary,
            "geometry_gate": self.geometry_gate_summary,
            "artifacts": {
                "model_psf": str(self.model_psf) if self.model_psf else None,
                "model_crd": str(self.model_crd) if self.model_crd else None,
                "model_pdb": str(self.model_pdb) if self.model_pdb else None,
            },
            "steps_applied": list(self.steps_applied),
            "message": self.message,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def apply_liquid_box_profile(
    args: argparse.Namespace,
    profile: str | None = None,
) -> LiquidBoxProfile:
    """Apply profile defaults; return the resolved profile name."""
    name = str(profile or getattr(args, "profile", "dense")).strip().lower()
    if name not in LIQUID_BOX_PROFILES:
        raise ValueError(
            f"Unknown liquid-box profile {name!r}; choose from {', '.join(LIQUID_BOX_PROFILES)}"
        )
    resolved: LiquidBoxProfile = name  # type: ignore[assignment]
    setattr(args, "profile", resolved)

    if getattr(args, "setup", None) is None:
        args.setup = "pbc_nvt"
    if getattr(args, "box_auto", None) is None and getattr(args, "box_size", None) is None:
        args.box_auto = "density"
    if getattr(args, "charmm_pre_minimize", None) is None:
        args.charmm_pre_minimize = True
    if getattr(args, "save", None) is None:
        args.save = True

    if resolved == "standard":
        if getattr(args, "liquid_prep", None) is None:
            args.liquid_prep = False
        return resolved

    args.liquid_prep = True
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        apply_density_prep_resilient_defaults,
    )

    apply_density_prep_resilient_defaults(args)

    if getattr(args, "charmm_mm_pretreat_echeck", None) is None and not getattr(
        args, "no_scale_echeck", False
    ):
        args.no_echeck = True

    if resolved == "conservative":
        if getattr(args, "box_size", None) is None and getattr(
            args, "target_density_g_cm3", None
        ) is None:
            current = getattr(args, "bulk_density_fraction", None)
            if current is None:
                args.bulk_density_fraction = 0.55
            else:
                args.bulk_density_fraction = min(float(current), 0.55)

    return resolved


def estimate_density_g_cm3(
    *,
    composition: dict[str, int] | None,
    box_side_A: float | None,
    n_molecules: int,
) -> float | None:
    if box_side_A is None or box_side_A <= 0.0 or composition is None:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import total_mass_g_for_composition

    try:
        total_mass_g = total_mass_g_for_composition(composition)
    except ValueError:
        return None
    volume_cm3 = (float(box_side_A) * 1.0e-8) ** 3
    if volume_cm3 <= 0.0:
        return None
    return float(total_mass_g) / volume_cm3


def measure_worst_intermonomer_A(
    positions: np.ndarray,
    atoms_per_list: list[int],
    *,
    box_side: float | None,
    use_pbc: bool,
) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.utils.geometry_checks import find_worst_intermonomer_overlap

    offsets = monomer_offsets_from_atoms_per(atoms_per_list)
    cell = np.diag([float(box_side), float(box_side), float(box_side)]) if box_side else None
    best_dist, _violation = find_worst_intermonomer_overlap(
        positions,
        offsets,
        cell=cell if use_pbc else None,
    )
    return float(best_dist)


def certify_intermonomer_geometry(
    positions: np.ndarray,
    atoms_per_list: list[int],
    *,
    args: argparse.Namespace,
    box_side: float | None,
    use_pbc: bool,
) -> tuple[float, bool, str]:
    """Return (worst_contact_A, passed, message)."""
    from mmml.utils.intermonomer_geometry import resolve_pre_mlpot_overlap_min_distance

    floor = resolve_pre_mlpot_overlap_min_distance(args)
    try:
        worst = measure_worst_intermonomer_A(
            positions,
            atoms_per_list,
            box_side=box_side,
            use_pbc=use_pbc,
        )
    except RuntimeError as exc:
        return float("nan"), False, str(exc)

    if worst < floor:
        return (
            worst,
            False,
            f"worst inter-monomer contact {worst:.3f} Å < prep floor {floor:.3f} Å",
        )
    return (
        worst,
        True,
        f"worst inter-monomer contact {worst:.3f} Å (prep floor {floor:.3f} Å)",
    )


def write_liquid_box_artifacts(
    result: LiquidBoxBuildResult,
    *,
    args: argparse.Namespace | None = None,
) -> None:
    result.out_dir.mkdir(parents=True, exist_ok=True)
    box_path = result.out_dir / BOX_JSON
    report_path = result.out_dir / REPORT_MD
    box_path.write_text(json.dumps(result.to_box_json(args=args), indent=2), encoding="utf-8")
    report_path.write_text(render_liquid_box_report(result, args=args), encoding="utf-8")
    result.box_json_path = box_path
    result.report_path = report_path


def render_liquid_box_report(
    result: LiquidBoxBuildResult,
    *,
    args: argparse.Namespace | None = None,
) -> str:
    from mmml.utils.intermonomer_geometry import resolve_dynamics_overlap_reference_A

    dyn_ref = resolve_dynamics_overlap_reference_A(args)
    lines = [
        "# Liquid box report",
        "",
        f"**Status:** {result.status.upper()}",
        f"**Profile:** {result.profile}",
        "",
        "## System",
        "",
        f"- Composition: `{result.composition or 'n/a'}`",
        f"- Molecules: {result.n_molecules}",
        f"- Atoms: {result.n_atoms}",
        "",
        "## Box",
        "",
    ]
    if result.box_side_A is not None:
        lines.append(f"- Cubic side: {result.box_side_A:.3f} Å")
    if result.density_g_cm3 is not None:
        lines.append(f"- Density: {result.density_g_cm3:.4f} g/cm³")
    lines.extend(
        [
            "",
            "## Geometry certification (MM)",
            "",
        ]
    )
    if result.worst_intermonomer_A is not None:
        lines.append(f"- Worst inter-monomer contact: {result.worst_intermonomer_A:.3f} Å")
    if result.prep_overlap_floor_A is not None:
        lines.append(f"- Prep floor: {result.prep_overlap_floor_A:.3f} Å")
    lines.append(f"- Dynamics overlap reference: {dyn_ref:.3f} Å")
    if result.mm_grms_kcalmol_A is not None:
        lines.append(f"- CHARMM MM GRMS: {result.mm_grms_kcalmol_A:.4f} kcal/mol/Å")
    if result.steps_applied:
        lines.extend(["", "## Steps applied", ""])
        lines.extend(f"- {step}" for step in result.steps_applied)
    lines.extend(["", "## Artifacts", ""])
    if result.model_psf:
        lines.append(f"- `{result.model_psf.name}`")
    if result.model_crd:
        lines.append(f"- `{result.model_crd.name}`")
    if result.model_pdb:
        lines.append(f"- `{result.model_pdb.name}`")
    lines.append(f"- `{BOX_JSON}`")
    prep_dir = result.out_dir / "prep_ladder"
    if prep_dir.is_dir():
        lines.append(f"- `{prep_dir.name}/` (checkpoints)")
    lines.extend(["", "## Next step", ""])
    if result.status == "pass" and result.model_psf and result.model_crd:
        lines.extend(
            [
                "```bash",
                "mmml md-system \\",
                f"  --from-psf {result.model_psf} \\",
                f"  --from-crd {result.model_crd} \\",
                "  --checkpoint /path/to/checkpoint.json \\",
                '  --md-stages mini,heat,equi \\',
                f"  --output-dir runs/{result.out_dir.name}_equil",
                "```",
            ]
        )
    else:
        lines.append(result.message or "Rebuild with a looser profile or adjust density knobs.")
    lines.append("")
    return "\n".join(lines)


def run_liquid_box_build(args: argparse.Namespace) -> LiquidBoxBuildResult:
    """MM-only liquid box pipeline (spike: mirrors early ``staged_workflow`` legs)."""
    profile = apply_liquid_box_profile(args, getattr(args, "profile", None))
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
        model_psf as model_psf_path,
        staged_artifact_paths,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import parse_composition_dict
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_charmm_output_from_args,
        build_cluster_from_args_with_tag,
        print_cluster_geometry_summary,
        resolve_charmm_use_pbc,
        resolve_pbc_box_side,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
        liquid_prep_enabled,
        run_pre_mlpot_geometry_gate,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
        setup_charmm_environment,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        _charmm_pre_minimize_before_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        save_cluster_topology_for_vmd,
        sync_charmm_positions,
    )
    from mmml.utils.intermonomer_geometry import resolve_pre_mlpot_overlap_min_distance

    composition_str = getattr(args, "composition", None)
    comp = parse_composition_dict(composition_str)
    z, r, n_mol, tag = build_cluster_from_args_with_tag(args)
    print_cluster_geometry_summary(r, n_mol)

    paths = staged_artifact_paths(out_dir, tag)
    charmm_pbc = resolve_charmm_use_pbc(args)
    box_side = resolve_pbc_box_side(args, r) if charmm_pbc else None
    atoms_per_list = getattr(args, "_cluster_atoms_per_list", None)
    if atoms_per_list is None and int(n_mol) > 0 and len(z) % int(n_mol) == 0:
        atoms_per_list = [len(z) // int(n_mol)] * int(n_mol)

    mc_summary: dict[str, Any] | None = None
    steps_applied: list[str] = ["packmol_cluster"]
    if atoms_per_list is not None and charmm_pbc and box_side is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
            apply_mc_density_equalization,
        )

        r, box_side, mc_result = apply_mc_density_equalization(
            args,
            r - r.mean(axis=0) + 0.5 * float(box_side),
            atoms_per_list=list(atoms_per_list),
            composition=getattr(args, "_cluster_composition_summary", None),
            box_side_A=box_side,
            use_pbc=charmm_pbc,
            handoff_present=False,
            min_intermonomer_distance_A=float(
                getattr(args, "min_intermonomer_atom_distance", 0.1) or 0.1
            ),
            min_box_side_A=cubic_box_length_from_geometry(
                r,
                ml_cutoff=float(getattr(args, "ml_cutoff", 12.0)),
            ),
        )
        mc_summary = mc_result.to_dict()
        if mc_result.ran:
            steps_applied.append("mc_density_equalize")

    mini_nprint = apply_charmm_output_from_args(args)
    setup_charmm_environment(use_pbc=charmm_pbc, cubic_box_side_A=box_side)
    sync_charmm_positions(r)

    vmd_files = save_cluster_topology_for_vmd(
        out_dir, r, stem="model", title="liquid-box cluster"
    )
    model_psf = model_psf_path(out_dir)
    model_pdb = Path(vmd_files["pdb"])
    steps_applied.append("save_model_topology")

    pretreat_restart_path: Path | None = None
    if getattr(args, "charmm_pre_minimize", True):
        r = _charmm_pre_minimize_before_mlpot(
            args,
            nprint=mini_nprint,
            reference_positions=r,
            save_crd_path=paths["charmm_mm_crd"],
            save_pdb_path=paths["charmm_mm_pdb"],
            save_psf_path=paths["charmm_mm_psf"],
            save_energy_json_path=paths["charmm_mm_energy_json"],
            use_pbc=charmm_pbc,
        )
        sync_charmm_positions(r)
        steps_applied.append("charmm_mm_pre_minimize")

    from mmml.interfaces.pycharmmInterface.mlpot.box_lattice_abnr import (
        run_mini_lattice_abnr,
        should_run_mini_lattice_abnr,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import should_run_mini_box_equil

    mm_stages = ["mini"]
    if should_run_mini_lattice_abnr(args, charmm_pbc=charmm_pbc, stages=mm_stages):
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            sync_workflow_pbc_box_side_after_mm_pretreat,
        )

        new_side = run_mini_lattice_abnr(
            args,
            box_side=box_side,
            use_pbc=charmm_pbc,
            pretreat_restart=pretreat_restart_path,
        )
        if new_side is not None:
            box_side = float(new_side)
        if charmm_pbc and box_side is not None:
            box_side = sync_workflow_pbc_box_side_after_mm_pretreat(
                box_side,
                pretreat_restart=None,
                args=args,
                quiet=bool(getattr(args, "quiet", False)),
            )
        r = get_charmm_positions_array()
        steps_applied.append("mini_lattice_abnr")

    if should_run_mini_box_equil(
        args,
        charmm_pbc=charmm_pbc,
        pretreat_mm=False,
        stages=mm_stages,
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.box_equil import run_mini_box_equilibration
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            sync_workflow_pbc_box_side_after_mm_pretreat,
        )

        dt_fs = float(getattr(args, "dt_fs", 0.25))
        timestep_ps = dt_fs * 1.0e-3
        temp = float(getattr(args, "temperature", getattr(args, "temp", 300.0)))
        echeck = float(getattr(args, "echeck", 100.0))
        run_mini_box_equilibration(
            args,
            paths=paths,
            timestep_ps=timestep_ps,
            temp=temp,
            echeck=echeck,
            duration_ps=float(getattr(args, "mini_box_equil_ps", 0.0) or 0.0),
            use_pbc=charmm_pbc,
            box_side=box_side,
        )
        if charmm_pbc and box_side is not None:
            box_side = sync_workflow_pbc_box_side_after_mm_pretreat(
                box_side,
                pretreat_restart=paths.get("mini_box_equil_res"),
                args=args,
                quiet=bool(getattr(args, "quiet", False)),
            )
        r = get_charmm_positions_array()
        steps_applied.append("mini_box_equil")

    gate_summary: dict[str, Any] | None = None
    if liquid_prep_enabled(args) and atoms_per_list is not None:
        r, box_side, gate_result = run_pre_mlpot_geometry_gate(
            args,
            positions=get_charmm_positions_array(),
            atoms_per_list=list(atoms_per_list),
            composition=getattr(args, "_cluster_composition_summary", None),
            box_side=box_side,
            charmm_pbc=charmm_pbc,
            n_mol=n_mol,
            n_atoms=len(z),
            atomic_numbers=np.asarray(z, dtype=int),
        )
        sync_charmm_positions(r)
        gate_summary = gate_result.to_dict()
        steps_applied.extend(gate_result.steps_applied)
    else:
        r = get_charmm_positions_array()

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import save_minimization_results

    mm_grms = float(charmm_grms())
    model_crd = out_dir / "model.crd"
    save_minimization_results(
        pdb_path=model_pdb,
        crd_path=model_crd,
        positions=r,
        title="liquid-box certified",
        show_energy=False,
    )
    steps_applied.append("write_model_crd")

    prep_floor = resolve_pre_mlpot_overlap_min_distance(args)
    worst = gate_summary.get("worst_intermonomer_A") if gate_summary else None
    passed = True
    cert_message = ""
    if worst is None and atoms_per_list is not None:
        worst, passed, cert_message = certify_intermonomer_geometry(
            r,
            list(atoms_per_list),
            args=args,
            box_side=box_side,
            use_pbc=charmm_pbc,
        )
    elif gate_summary is not None:
        passed = not bool(gate_summary.get("aborted"))
        worst = gate_summary.get("worst_intermonomer_A")
        cert_message = gate_summary.get("reason", "")

    density = estimate_density_g_cm3(
        composition=comp,
        box_side_A=box_side,
        n_molecules=int(n_mol),
    )

    result = LiquidBoxBuildResult(
        status="pass" if passed else "fail",
        out_dir=out_dir,
        profile=profile,
        composition=composition_str,
        n_molecules=int(n_mol),
        n_atoms=len(z),
        box_side_A=float(box_side) if box_side is not None else None,
        density_g_cm3=density,
        worst_intermonomer_A=float(worst) if worst is not None else None,
        prep_overlap_floor_A=float(prep_floor),
        mm_grms_kcalmol_A=mm_grms,
        mc_density_summary=mc_summary,
        geometry_gate_summary=gate_summary,
        model_psf=model_psf if model_psf.is_file() else None,
        model_crd=model_crd if model_crd.is_file() else None,
        model_pdb=model_pdb if model_pdb.is_file() else None,
        message=cert_message,
        steps_applied=steps_applied,
    )
    write_liquid_box_artifacts(result, args=args)

    if not getattr(args, "quiet", False):
        print(f"\nliquid-box: {result.status.upper()} → {out_dir}")
        if result.report_path is not None:
            print(f"  report: {result.report_path}")
        if result.box_json_path is not None:
            print(f"  box.json: {result.box_json_path}")

    return result
