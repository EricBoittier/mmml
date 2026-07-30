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
- Campaign/handoff continuation (``--continue-from``), lambda-TI.

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
]


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
    if getattr(args, "continue_from", None):
        raise NotImplementedError("--jaxmd-unified does not yet support --continue-from (handoff)")

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

        traj = assemble_and_run(
            run_config, system=system, ctx=ctx, term_kwargs=term_kwargs or None
        )

    energies = traj.metadata.get("energies")
    if energies is not None and len(energies):
        print(
            f"mmml md-system (jaxmd-unified): {traj.n_frames} frames, "
            f"E0={energies[0]:.4f} eV, Efinal={energies[-1]:.4f} eV",
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
