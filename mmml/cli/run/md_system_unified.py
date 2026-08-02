"""Opt-in unified-stack path for ``mmml md-system --backend jaxmd``.

Wires the legacy ``md-system`` CLI onto the shared ``mmml.md`` pipeline
(``runconfig_from_md_system_args`` → ``assemble_and_run``) instead of the
legacy ``mmml.cli.run.md_pbc_suite.jaxmd`` inline loop. Opt-in via
``--jaxmd-unified`` (see ``build_parser()`` / ``main()`` in
``mmml.cli.run.md_system``) so the existing default path is untouched until
this one is validated across more of md-system's feature surface.

Deliberately NOT yet supported (raise clearly rather than silently diverge
from the legacy backend):

- ``--builder pyxtal`` / ``--template-pdb`` (only the packmol composition
  builder and ``--from-pdb`` full-system loading are wired here).
- lambda-TI.

Geometry handoff (``--continue-from`` / campaign ``depends_on``) is supported:
positions + box are applied; velocities / thermostat / barostat state are not
(rethermalize downstream). FIRE is skipped after handoff unless
``--handoff-pre-minimize``.

See ``docs/md-cg-unification-design.md`` (§0, §9, §11) and
``docs/md-cg-unification-handoff.md`` for the surrounding architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "check_md_system_args_supported",
    "build_packmol_system_with_ffparams",
    "build_from_pdb_system_with_ffparams",
    "build_energy_context",
    "run_unified_jaxmd",
    "npt_volume_ratio_ok",
    "format_npt_volume_pressure_line",
]

# Short-smoke gate: catch explode/collapse, not density equilibration.
_NPT_VOLUME_RATIO_MIN = 0.5
_NPT_VOLUME_RATIO_MAX = 2.0


def npt_volume_ratio_ok(
    volumes_A3: Any,
    *,
    ratio_min: float = _NPT_VOLUME_RATIO_MIN,
    ratio_max: float = _NPT_VOLUME_RATIO_MAX,
) -> tuple[bool, float | None]:
    """Return ``(ok, Vfinal/V0)`` for NPT smoke gating."""
    vols = np.asarray(volumes_A3, dtype=np.float64).reshape(-1)
    if vols.size < 1 or not np.all(np.isfinite(vols)):
        return False, None
    v0 = float(vols[0])
    if v0 <= 0.0:
        return False, None
    ratio = float(vols[-1]) / v0
    if not np.isfinite(ratio):
        return False, ratio
    return ratio_min <= ratio <= ratio_max, ratio


def format_npt_volume_pressure_line(
    metadata: dict[str, Any],
    *,
    target_pressure_bar: float | None = None,
) -> str | None:
    """One-line NPT diagnostics, or ``None`` when volume metadata is absent."""
    volumes = metadata.get("volumes_A3")
    if volumes is None:
        boxes = metadata.get("boxes")
        if boxes is None:
            return None
        boxes_arr = np.asarray(boxes)
        volumes = np.array(
            [abs(float(np.linalg.det(np.asarray(b, dtype=np.float64)))) for b in boxes_arr],
            dtype=np.float64,
        )
    vols = np.asarray(volumes, dtype=np.float64).reshape(-1)
    if vols.size < 1:
        return None
    v0 = float(vols[0])
    vf = float(vols[-1])
    ratio = vf / v0 if v0 > 0 else float("nan")
    L0 = v0 ** (1.0 / 3.0) if v0 > 0 else float("nan")
    Lf = vf ** (1.0 / 3.0) if vf > 0 else float("nan")
    pressures = metadata.get("pressures_bar")
    p_target = metadata.get("target_pressure_bar", target_pressure_bar)
    if pressures is not None and len(pressures):
        p0 = float(np.asarray(pressures, dtype=np.float64).reshape(-1)[0])
        pf = float(np.asarray(pressures, dtype=np.float64).reshape(-1)[-1])
        p_part = f" P0={p0:.4g} bar Pfinal={pf:.4g} bar"
    else:
        p_part = ""
    p_kin = metadata.get("pressures_kin_bar")
    p_vir = metadata.get("pressures_vir_bar")
    if p_kin is not None and len(p_kin) and p_vir is not None and len(p_vir):
        pk0 = float(np.asarray(p_kin, dtype=np.float64).reshape(-1)[0])
        pv0 = float(np.asarray(p_vir, dtype=np.float64).reshape(-1)[0])
        p_part += f" Pkin0={pk0:.4g} bar Pvir0={pv0:.4g} bar"
    t_part = f" P_target={float(p_target):.4g} bar" if p_target is not None else ""
    return (
        f"mmml md-system (jaxmd-unified): NPT "
        f"V0={v0:.4g} A3 (L~{L0:.4g} A) "
        f"Vfinal={vf:.4g} A3 (L~{Lf:.4g} A) "
        f"Vfinal/V0={ratio:.4g}{p_part}{t_part}"
    )


def _system_with_positions(system, positions: np.ndarray):
    """Return a copy of ``system`` with updated ``R`` (MolecularSystem is frozen)."""
    from dataclasses import replace

    return replace(system, R=np.asarray(positions, dtype=np.float64))


def _system_with_positions_and_box(system, positions: np.ndarray, box: np.ndarray | None):
    """Return a copy of ``system`` with updated ``R`` and optional ``box``."""
    from dataclasses import replace

    kwargs: dict[str, Any] = {"R": np.asarray(positions, dtype=np.float64)}
    if box is not None:
        kwargs["box"] = np.asarray(box, dtype=np.float64)
    return replace(system, **kwargs)


def _resolve_handoff_in(args: Any):
    """Return campaign/context handoff or load ``--continue-from`` if set."""
    from mmml.cli.run.md_handoff import get_handoff_in, load_handoff

    handoff = get_handoff_in()
    if handoff is not None:
        return handoff
    path = getattr(args, "continue_from", None)
    if not path:
        return None
    frame = int(getattr(args, "continue_from_frame", -1) or -1)
    return load_handoff(Path(str(path)), frame=frame)


def _apply_incoming_handoff(args: Any, system):
    """Overlay handoff positions (+ cell) onto the built topology system.

    Returns ``(system, from_handoff)``. Geometry-only: velocities are ignored
    (driver rethermalizes). Raises on atom-count / Z mismatches.
    """
    handoff = _resolve_handoff_in(args)
    if handoff is None:
        return system, False

    pos = np.asarray(handoff.positions, dtype=np.float64)
    if pos.shape != tuple(system.R.shape):
        raise ValueError(
            "jaxmd-unified handoff positions shape "
            f"{pos.shape} != system.R shape {tuple(system.R.shape)}"
        )
    z_h = np.asarray(handoff.atomic_numbers, dtype=np.int32).reshape(-1)
    z_s = np.asarray(system.Z, dtype=np.int32).reshape(-1)
    if z_h.size == z_s.size and np.any(z_h != 0) and not np.array_equal(z_h, z_s):
        raise ValueError(
            "jaxmd-unified handoff atomic_numbers do not match the built system"
        )
    box = None
    if handoff.cell is not None:
        box = np.asarray(handoff.cell, dtype=np.float64)
        if box.shape != (3, 3):
            raise ValueError(
                f"jaxmd-unified handoff cell must be (3, 3); got {box.shape}"
            )
    elif system.box is not None and bool(getattr(handoff, "pbc", False)):
        print(
            "mmml md-system (jaxmd-unified): WARNING handoff has pbc but no cell; "
            "keeping built-system box",
            flush=True,
        )
    system = _system_with_positions_and_box(system, pos, box)
    src = (handoff.metadata or {}).get("source") or getattr(args, "continue_from", None) or "context"
    print(
        f"mmml md-system (jaxmd-unified): continue-from geometry "
        f"N={system.n_atoms} box={'yes' if system.box is not None else 'no'} "
        f"({src})",
        flush=True,
    )
    return system, True


def _publish_unified_handoff(args: Any, system, traj) -> None:
    """Publish final geometry for campaign ``depends_on`` / ``save_handoff``."""
    from mmml.cli.run.md_handoff import MdHandoffState, set_handoff_out

    positions = traj.metadata.get("positions")
    if positions is None or len(positions) == 0:
        return
    R = np.asarray(positions[-1], dtype=np.float64)
    boxes = traj.metadata.get("boxes")
    if boxes is not None and len(boxes):
        cell = np.asarray(boxes[-1], dtype=np.float64)
        pbc = True
    elif system.box is not None:
        cell = np.asarray(system.box, dtype=np.float64)
        pbc = True
    else:
        cell = None
        pbc = False
    set_handoff_out(
        MdHandoffState(
            positions=R,
            atomic_numbers=np.asarray(system.Z, dtype=np.int32),
            velocities=None,
            cell=cell,
            pbc=pbc,
            temperature_K=float(getattr(args, "temperature", 300.0)),
            metadata={
                "backend": "jaxmd-unified",
                "source": "run_unified_jaxmd",
                "note": "geometry-only; velocities not preserved",
            },
        )
    )


def _fire_minimize_system(
    args: Any,
    run_config: Any,
    system,
    ctx: Any,
    term_kwargs: dict[str, dict] | None,
):
    """Optional FIRE relax before NVT/NPT/NVE (Packmol cold starts)."""
    from dataclasses import replace

    from mmml.md.assemble import assemble_and_run

    n_steps = int(getattr(args, "jaxmd_minimize_steps", 0) or 0)
    if n_steps <= 0 or run_config.ensemble.ensemble == "min":
        return system

    min_params = dict(run_config.ensemble.params)
    min_params["float64"] = True
    min_params["seed"] = int(getattr(args, "seed", 0) or 0)
    min_ens = replace(
        run_config.ensemble,
        ensemble="min",
        n_steps=n_steps,
        thermostat=None,
        barostat=None,
        params=min_params,
    )
    min_cfg = replace(run_config, ensemble=min_ens, output_dir=None)
    print(
        f"mmml md-system (jaxmd-unified): FIRE minimize {n_steps} steps before "
        f"{run_config.ensemble.ensemble}",
        flush=True,
    )
    traj = assemble_and_run(
        min_cfg, system=system, ctx=ctx, term_kwargs=term_kwargs or None
    )
    positions = traj.metadata.get("positions")
    energies = traj.metadata.get("energies")
    if positions is None or len(positions) == 0:
        print(
            "mmml md-system (jaxmd-unified): WARNING FIRE produced no frames; "
            "continuing with input coordinates",
            flush=True,
        )
        return system
    e = np.asarray(energies, dtype=np.float64) if energies is not None else None
    if e is not None and np.any(np.isfinite(e)):
        best = int(np.nanargmin(e))
        print(
            f"mmml md-system (jaxmd-unified): FIRE E0={e[0]:.4f} eV "
            f"Ebest={e[best]:.4f} eV (frame {best})",
            flush=True,
        )
        if not np.isfinite(e[best]) or float(e[best]) > 1.0e6:
            print(
                "mmml md-system (jaxmd-unified): WARNING FIRE best energy still "
                f"pathological ({e[best]}); check packing / box size",
                flush=True,
            )
    else:
        best = len(positions) - 1
        print(
            "mmml md-system (jaxmd-unified): WARNING FIRE energies non-finite; "
            "using last frame",
            flush=True,
        )
    return _system_with_positions(system, positions[best])


def _print_unified_ensemble_banner(args: Any, run_config: Any) -> None:
    ens = run_config.ensemble
    params = dict(ens.params or {})
    print(
        f"mmml md-system (jaxmd-unified): ensemble={ens.ensemble} "
        f"thermostat={ens.thermostat or 'default'} "
        f"dt_fs={ens.dt_fs} n_steps={ens.n_steps} "
        f"float64={bool(params.get('float64', False))} "
        f"T={ens.temperature_K} K P_target={ens.pressure_bar} bar "
        f"barostat_tau={((params.get('barostat_kwargs') or {}).get('tau'))}",
        flush=True,
    )


def check_md_system_args_supported(args: Any) -> None:
    """Raise clearly (before any CHARMM build) for combinations not yet wired."""
    from mmml.md.lowering import terms_from_md_system_args

    builder = getattr(args, "builder", None)
    if builder not in (None, "packmol", "from_pdb"):
        raise NotImplementedError(
            f"--jaxmd-unified supports the packmol composition and from_pdb "
            f"builders; got --builder {builder!r}"
        )
    if getattr(args, "template_pdb", None):
        raise NotImplementedError("--jaxmd-unified does not yet support --template-pdb")
    if not getattr(args, "from_pdb", None) and not getattr(args, "composition", None):
        raise ValueError(
            "--jaxmd-unified needs either --from-pdb (a prebuilt full-system PDB) "
            "or --composition (for the packmol builder)"
        )
    terms = terms_from_md_system_args(args)
    if "ml_intra" in terms and not getattr(args, "checkpoint", None):
        raise ValueError("--jaxmd-unified with ml_intra requires --checkpoint")


def _durable_psf_path(args: Any, *, stem: str = "md_system_unified") -> Path:
    """Write-live-PSF destination that outlives TemporaryDirectory cleanup.

    ``mm_bonded`` (and other late readers) need ``system.psf_path`` to still
    exist after the builder returns. Prefer ``--output-dir``, else a process
    temp file that is not auto-deleted with a TemporaryDirectory context.
    """
    import os
    import tempfile

    out = getattr(args, "output_dir", None)
    if out is not None:
        dest_dir = Path(out)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir / f"{stem}.psf"
    fd, name = tempfile.mkstemp(suffix=".psf", prefix=f"{stem}_")
    os.close(fd)
    return Path(name)


def build_packmol_system_with_ffparams(spec: Any, args: Any = None):
    """Build a composition system via packmol and lower it to ``FFParams``.

    ``PackmolSystemBuilder`` (``mmml.md.builders.placement``) does not write a
    PSF file by default, since ``build_packmol_composition_cluster`` never
    persists one — it only leaves the built system live in CHARMM. Reuse the
    same building blocks, then write the live PSF to a durable path so
    ``FFParams`` / ``mm_bonded`` can resolve it after the builder returns.
    """
    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster
    from mmml.md.builders.placement import _box, _composition, _lower_optional_psf, _placement_system

    params = dict(spec.params)
    box = _box(spec, params)
    composition = _composition(spec)
    if "center" not in params:
        if box is None:
            raise ValueError("packmol builder requires center or a box size")
        params["center"] = tuple(np.diag(box) / 2.0)
    params.setdefault("cube_side", spec.box_size)

    z, positions, sizes, residues = build_packmol_composition_cluster(
        composition=composition, seed=spec.seed, **params
    )

    import pycharmm.write as write

    psf_path = _durable_psf_path(args if args is not None else spec, stem="md_system_unified")
    write.psf_card(str(psf_path))
    system = _placement_system(
        name="packmol", spec=spec, z=z, positions=positions,
        atoms_per_molecule=list(sizes), residue_names=list(residues), box=box,
    )
    return _lower_optional_psf(system, psf_path=psf_path, prm_paths=())


def build_from_pdb_system_with_ffparams(args: Any, spec: Any):
    """Cold-start from a prebuilt full-system PDB (``--from-pdb``).

    Prefer the sibling ``model.psf`` written by make-box when present. Otherwise
    dump the live CHARMM PSF to a durable path (not a TemporaryDirectory that
    is deleted before ``mm_bonded`` runs).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _sibling_psf_for_pdb,
        load_cluster_from_pdb,
    )
    from mmml.md.builders.placement import _lower_optional_psf, _placement_system

    pdb_path = spec.template_pdb or getattr(args, "from_pdb", None)
    if pdb_path is None:
        raise ValueError("from_pdb builder requires SystemSpec.template_pdb or --from-pdb")
    pdb_path = Path(pdb_path)

    z, positions, n_mol, _tag = load_cluster_from_pdb(args, pdb_path=pdb_path)

    # load_cluster_from_pdb records the per-residue split it derived from the
    # generated PSF; _placement_system needs exactly that to form monomer groups.
    atoms_per_molecule = [int(x) for x in getattr(args, "_cluster_atoms_per_list", [])]
    residue_names = [str(x) for x in getattr(args, "_cluster_residue_labels", [])]
    if len(atoms_per_molecule) != n_mol or len(residue_names) != n_mol:
        raise ValueError(
            f"from_pdb load returned {n_mol} residues but "
            f"{len(atoms_per_molecule)} sizes / {len(residue_names)} labels "
            f"for {pdb_path}"
        )

    # The box is resolved during the load (CRYST1 -> sibling box.json ->
    # --box-size), i.e. after the spec was lowered, so read it back off args.
    side = getattr(args, "box_size", None) or spec.box_size
    box = None if side is None else np.eye(3, dtype=np.float64) * float(side)

    sibling = _sibling_psf_for_pdb(pdb_path.resolve() if pdb_path.is_file() else pdb_path)
    if sibling is None:
        # Path may be relative; also try resolved from CWD.
        sibling = _sibling_psf_for_pdb(Path(str(pdb_path)).expanduser().resolve())

    import pycharmm.write as write

    if sibling is not None and sibling.is_file():
        psf_path = sibling.resolve()
    else:
        psf_path = _durable_psf_path(args, stem="md_system_unified")
        write.psf_card(str(psf_path))

    system = _placement_system(
        name="from_pdb", spec=spec, z=z, positions=positions,
        atoms_per_molecule=atoms_per_molecule, residue_names=residue_names, box=box,
    )
    return _lower_optional_psf(system, psf_path=psf_path, prm_paths=())


def _load_model(checkpoint_path: Path) -> tuple[Any, Any]:
    from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint

    calc = create_calculator_from_checkpoint(str(checkpoint_path))
    model = getattr(calc, "model", getattr(calc, "_mmml_physnet_model", None))
    params = getattr(calc, "params", getattr(calc, "_mmml_physnet_params", None))
    if model is None or params is None:
        raise ValueError(f"could not extract model/params from checkpoint {checkpoint_path}")
    return model, params


def _freeze_multipoles(system, multipole_checkpoint: Path | None) -> dict[str, Any]:
    """Predict fragment multipoles once and return fixed_multipoles options."""
    from ase import Atoms

    from mmml.models.multipoles.electrostatics import (
        LearnedMolecularMultipoleElectrostatics,
        resolve_multipoles_checkpoint,
    )

    ckpt = resolve_multipoles_checkpoint(multipole_checkpoint)
    calc = LearnedMolecularMultipoleElectrostatics(checkpoint=ckpt)
    atoms = Atoms(numbers=np.asarray(system.Z), positions=np.asarray(system.R))
    atoms.arrays["mol_id"] = np.asarray(system.mol_id, dtype=np.int32)
    pred = calc.predict_fragment_multipoles(atoms)
    fragments = [
        np.asarray(ix, dtype=np.int32) for ix in system.monomer_indices
    ]
    charges = np.asarray(pred["charges"], dtype=np.float64)
    dipoles = np.asarray(pred["dipoles_bohr"], dtype=np.float64)
    return {
        "charges": charges,
        "dipoles_body_bohr": dipoles,
        "ref_positions_A": np.asarray(system.R, dtype=np.float64),
        "fragment_indices": fragments,
    }


def _freeze_dispersion(system, mbd_checkpoint: Path | None, mbd_weight: float) -> dict[str, Any]:
    """Predict per-atom C6/alpha once; map to QDO coefficients + damping."""
    from ase import Atoms

    from mmml.models.mbd.calculator import QCMLMBDCalculator, resolve_mbd_checkpoint

    ckpt = resolve_mbd_checkpoint(mbd_checkpoint)
    calc = QCMLMBDCalculator(checkpoint=ckpt)
    atoms = Atoms(numbers=np.asarray(system.Z), positions=np.asarray(system.R))
    pred = calc.predict_mbd(atoms)
    c6 = np.asarray(pred["c6_native"], dtype=np.float64).reshape(-1)
    # QDO wants (N, 3) C6/C8/C10; without higher multipole C_n from the model,
    # use C8=C10=0 and a simple damping radius from polarizability when present.
    coeffs = np.zeros((system.n_atoms, 3), dtype=np.float64)
    coeffs[:, 0] = c6
    if "polarizabilities_bohr3" in pred:
        alpha = np.asarray(pred["polarizabilities_bohr3"], dtype=np.float64).reshape(-1)
        damp = np.maximum(alpha ** (1.0 / 3.0), 1e-3)
    else:
        damp = np.ones(system.n_atoms, dtype=np.float64)
    return {
        "coefficients_per_atom": coeffs,
        "damping_radii": damp,
        "weight": float(mbd_weight),
    }


def build_energy_context(args: Any, system, terms: tuple[str, ...]):
    """Build :class:`EnergyContext` for the selected terms (ML and/or fixed QCML)."""
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.energy.terms.zbl import DEFAULT_ZBL_CUTOFF_A, DEFAULT_ZBL_CUTON_A

    options: dict[str, Any] = {
        "zbl_cuton": DEFAULT_ZBL_CUTON_A,
        "zbl_cutoff": DEFAULT_ZBL_CUTOFF_A,
        "mbd_weight": float(getattr(args, "mbd_weight", 1.0)),
    }

    model = params = None
    if "ml_intra" in terms:
        ckpt = getattr(args, "checkpoint", None)
        if ckpt is None:
            raise ValueError("ml_intra requires --checkpoint")
        model, params = _load_model(Path(ckpt))

    if "multipole" in terms and "fixed_multipoles" not in options:
        options["fixed_multipoles"] = _freeze_multipoles(
            system, getattr(args, "multipole_checkpoint", None)
        )
    if "mbd" in terms and "fixed_dispersion" not in options:
        options["fixed_dispersion"] = _freeze_dispersion(
            system,
            getattr(args, "mbd_checkpoint", None),
            float(getattr(args, "mbd_weight", 1.0)),
        )

    return EnergyContext(model=model, params=params, options=options)


def run_unified_jaxmd(args: Any) -> int:
    """Run ``args`` through the unified ``mmml.md`` pipeline; return an exit code."""
    check_md_system_args_supported(args)

    # Platforms + CUDA runtime libs before import jax. Campaign ``quiet: true``
    # still gets the device banner below (honours MMML_QUIET only).
    from mmml.interfaces.pycharmmInterface.jax_device_policy import (
        apply_mlpot_jax_platform_env,
        mlpot_device_context_fell_back_to_cpu,
        mlpot_jax_device_context,
        print_jax_device_banner,
        reset_mlpot_device_fallback_flag,
    )

    apply_mlpot_jax_platform_env(quiet=True)

    import jax

    jax.config.update("jax_enable_x64", True)

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.md.assemble import assemble_and_run
    from mmml.md.lowering import runconfig_from_md_system_args

    # Explicit and idempotent: unlike PeptideWaterSystemBuilder's underlying
    # build_trialanine_water_box_in_charmm, build_packmol_composition_cluster
    # does not self-bootstrap CHARMM (it assumes the caller already has).
    if not ensure_pycharmm_loaded():
        raise RuntimeError("PyCHARMM not available (CHARMM_LIB_DIR / libcharmm.so)")

    if getattr(args, "from_pdb", None):
        # Same normalisation the ase / staged cold-start paths apply: resolves the
        # path and rejects --from-pdb mixed with --from-psf/--from-crd or with a
        # Packmol composition. Must run before the spec is lowered.
        from mmml.interfaces.pycharmmInterface.mlpot.composition_spec import (
            apply_from_pdb_alias,
        )

        apply_from_pdb_alias(args)

    run_config = runconfig_from_md_system_args(args)
    if (run_config.system.builder or "").lower() == "from_pdb":
        system = build_from_pdb_system_with_ffparams(args, run_config.system)
    else:
        system = build_packmol_system_with_ffparams(run_config.system, args)

    from dataclasses import replace
    from pathlib import Path

    from mmml.md.ml_region import (
        apply_ml_resnames_mechanical_embedding,
        parse_ml_resnames,
    )

    policy_path = getattr(args, "interaction_policy", None)
    if policy_path is not None:
        from mmml.md.interactions import (
            assert_interaction_plan_lowerable,
            compile_interaction_policy,
            load_interaction_policy,
            mechanical_embedding_ml_species,
            policy_is_mechanical_embedding,
        )

        policy = load_interaction_policy(policy_path)
        plan = compile_interaction_policy(system, policy)
        assert_interaction_plan_lowerable(plan, runner="jaxmd-unified", policy=policy)
        # Mechanical embedding ownership → ml_resnames (unless already set).
        if (
            policy_is_mechanical_embedding(policy)
            and parse_ml_resnames(getattr(args, "ml_resnames", None)) is None
        ):
            args.ml_resnames = list(mechanical_embedding_ml_species(policy))
            if not getattr(args, "checkpoint", None):
                for pname in policy.monomers.values():
                    spec = policy.providers.get(str(pname))
                    if spec is not None and spec.kind == "ml" and spec.checkpoint:
                        args.checkpoint = Path(spec.checkpoint)
                        break

    term_kwargs: dict[str, dict] = {}
    ml_resnames = parse_ml_resnames(getattr(args, "ml_resnames", None))
    if ml_resnames is not None:
        if "ml_intra" not in run_config.terms:
            raise ValueError(
                "--ml-resnames / ml_resnames requires ml_intra "
                "(omit --ff cgenff / provide a checkpoint)"
            )
        system, term_kwargs, ml_indices = apply_ml_resnames_mechanical_embedding(
            system, ml_resnames
        )
        # Mechanical embedding: ML solute + MM bonded (solvent) + MM nonbonded.
        terms = list(run_config.terms)
        if "mm_bonded" not in terms:
            if "ml_intra" in terms:
                i = terms.index("ml_intra") + 1
                terms.insert(i, "mm_bonded")
            else:
                terms.insert(0, "mm_bonded")
        run_config = replace(run_config, terms=tuple(terms))
        extra_prm = _resolve_extra_prm_files(args)
        if extra_prm:
            term_kwargs.setdefault("mm_bonded", {})["extra_prm_files"] = extra_prm
        print(
            f"mmml md-system (jaxmd-unified): ML region "
            f"{len(ml_indices)} atoms resnames={list(ml_resnames)}; "
            f"mm_bonded on MM atoms; MM nonbonded for solute–solvent / solvent–solvent",
            flush=True,
        )

    # Overlay campaign / --continue-from geometry after topology + ML-region remap.
    system, from_handoff = _apply_incoming_handoff(args, system)

    # Pin ML + jax-md energy/MD to MMML_MLPOT_DEVICE (default gpu). Without this,
    # a cpu-first JAX_PLATFORMS list (or silent CUDA plugin fallback) leaves
    # Spooky/PhysNet on the host while nvidia-smi stays idle.
    reset_mlpot_device_fallback_flag()
    with mlpot_jax_device_context() as jax_device:
        print_jax_device_banner(active_device=jax_device)
        if mlpot_device_context_fell_back_to_cpu():
            print(
                "mmml md-system (jaxmd-unified): computing on CPU "
                "(GPU requested but unavailable — see WARNING above)",
                flush=True,
            )
        ctx = build_energy_context(args, system, run_config.terms)
        _print_unified_ensemble_banner(args, run_config)
        # Handoff geometry is already dynamics-ready; skip cold-start FIRE unless
        # the user explicitly asked for --handoff-pre-minimize.
        if from_handoff and not bool(getattr(args, "handoff_pre_minimize", False)):
            print(
                "mmml md-system (jaxmd-unified): skipping FIRE "
                "(continue-from / campaign handoff)",
                flush=True,
            )
        else:
            system = _fire_minimize_system(
                args, run_config, system, ctx, term_kwargs
            )

        traj = assemble_and_run(
            run_config, system=system, ctx=ctx, term_kwargs=term_kwargs or None
        )

    _publish_unified_handoff(args, system, traj)

    energies = traj.metadata.get("energies")
    if energies is not None and len(energies):
        print(
            f"mmml md-system (jaxmd-unified): {traj.n_frames} frames, "
            f"E0={energies[0]:.4f} eV, Efinal={energies[-1]:.4f} eV",
            flush=True,
        )
    npt_line = format_npt_volume_pressure_line(
        traj.metadata,
        target_pressure_bar=float(getattr(args, "pressure", 1.0)),
    )
    if npt_line is not None:
        print(npt_line, flush=True)
        pressures = traj.metadata.get("pressures_bar")
        if pressures is not None and len(pressures):
            p0 = float(np.asarray(pressures, dtype=np.float64).reshape(-1)[0])
            if np.isfinite(p0) and abs(p0) > 500.0:
                print(
                    f"mmml md-system (jaxmd-unified): WARNING NPT |P0|={abs(p0):.4g} bar "
                    f">> P_target (dilute/cold-start box). Use a denser box or raise "
                    f"barostat_tau (metal time) so the piston cannot slam the cell "
                    f"on a short smoke.",
                    flush=True,
                )
    if energies is None or not np.all(np.isfinite(energies)):
        print("mmml md-system: jaxmd-unified produced non-finite energies", file=sys.stderr)
        return 1
    # Huge-but-finite energies still mean the run exploded (e.g. missing solvent
    # bonded terms). Fail closed so wrapper scripts do not print PASS.
    abs_max = float(np.max(np.abs(np.asarray(energies, dtype=np.float64))))
    if abs_max > 1.0e6:
        print(
            f"mmml md-system: jaxmd-unified energy blew up "
            f"(|E|_max={abs_max:.4e} eV > 1e6)",
            file=sys.stderr,
        )
        return 1
    if "volumes_A3" in traj.metadata or "boxes" in traj.metadata:
        volumes = traj.metadata.get("volumes_A3")
        if volumes is None and traj.metadata.get("boxes") is not None:
            volumes = [
                abs(float(np.linalg.det(np.asarray(b, dtype=np.float64))))
                for b in np.asarray(traj.metadata["boxes"])
            ]
        ok, ratio = npt_volume_ratio_ok(volumes)
        if not ok:
            ratio_s = "nan" if ratio is None else f"{ratio:.4g}"
            print(
                f"mmml md-system: jaxmd-unified NPT volume ratio out of range "
                f"(Vfinal/V0={ratio_s}; allowed "
                f"[{_NPT_VOLUME_RATIO_MIN}, {_NPT_VOLUME_RATIO_MAX}])",
                file=sys.stderr,
            )
            return 1
        pressures = traj.metadata.get("pressures_bar")
        if pressures is not None and not np.all(
            np.isfinite(np.asarray(pressures, dtype=np.float64))
        ):
            print(
                "mmml md-system: jaxmd-unified produced non-finite NPT pressures",
                file=sys.stderr,
            )
            return 1
    return 0


def _resolve_extra_prm_files(args: Any) -> list[Path]:
    """Append RTF/PRM extras (e.g. CH3CL) for CGenFF bonded loading."""
    import os

    out: list[Path] = []
    raw = getattr(args, "cgenff_extra_prm", None) or os.environ.get(
        "MMML_CGENFF_EXTRA_PRM", ""
    )
    if raw:
        p = Path(str(raw)).expanduser()
        if p.is_file():
            out.append(p)
    # Convention used by examples/m when env is unset but the append file exists.
    fallback = Path(__file__).resolve().parents[3] / "examples" / "m" / "par_ch3cl.prm"
    if fallback.is_file() and fallback.resolve() not in {p.resolve() for p in out}:
        # Only auto-include when the composition / PDB likely needs CH3CL.
        labels = [str(x).upper() for x in (getattr(args, "_cluster_residue_labels", None) or [])]
        if any(lab == "CH3CL" for lab in labels) or "CH3CL" in str(
            getattr(args, "composition", "") or ""
        ).upper():
            out.append(fallback)
    return out
