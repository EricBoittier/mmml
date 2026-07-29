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
import tempfile
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


def build_packmol_system_with_ffparams(spec: Any):
    """Build a composition system via packmol and lower it to ``FFParams``.

    ``PackmolSystemBuilder`` (``mmml.md.builders.placement``) does not write a
    PSF file by default, since ``build_packmol_composition_cluster`` never
    persists one — it only leaves the built system live in CHARMM. Reuse the
    same building blocks, then write the live PSF to a scratch file so
    ``FFParams`` can be resolved from it (mirrors ``_lower_optional_psf``).
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

    with tempfile.TemporaryDirectory() as tmp:
        import pycharmm.write as write

        psf_path = Path(tmp) / "md_system_unified.psf"
        write.psf_card(str(psf_path))
        system = _placement_system(
            name="packmol", spec=spec, z=z, positions=positions,
            atoms_per_molecule=list(sizes), residue_names=list(residues), box=box,
        )
        system = _lower_optional_psf(system, psf_path=psf_path, prm_paths=())
    return system


def build_from_pdb_system_with_ffparams(args: Any, spec: Any):
    """Cold-start from a prebuilt full-system PDB (``--from-pdb``).

    ``load_cluster_from_pdb`` generates the PSF live in CHARMM from the PDB's
    residue sequence and overlays its coordinates; like
    ``build_packmol_composition_cluster`` it never persists a PSF, so write the
    live one to scratch for ``FFParams`` resolution.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.setup import load_cluster_from_pdb
    from mmml.md.builders.placement import _lower_optional_psf, _placement_system

    pdb_path = spec.template_pdb or getattr(args, "from_pdb", None)
    if pdb_path is None:
        raise ValueError("from_pdb builder requires SystemSpec.template_pdb or --from-pdb")

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

    with tempfile.TemporaryDirectory() as tmp:
        import pycharmm.write as write

        psf_path = Path(tmp) / "md_system_unified.psf"
        write.psf_card(str(psf_path))
        system = _placement_system(
            name="from_pdb", spec=spec, z=z, positions=positions,
            atoms_per_molecule=atoms_per_molecule, residue_names=residue_names, box=box,
        )
        system = _lower_optional_psf(system, psf_path=psf_path, prm_paths=())
    return system


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
        system = build_packmol_system_with_ffparams(run_config.system)
    policy_path = getattr(args, "interaction_policy", None)
    if policy_path is not None:
        from mmml.md.interactions import (
            assert_interaction_plan_lowerable,
            compile_interaction_policy,
            load_interaction_policy,
        )

        plan = compile_interaction_policy(system, load_interaction_policy(policy_path))
        assert_interaction_plan_lowerable(plan, runner="jaxmd-unified")
    ctx = build_energy_context(args, system, run_config.terms)

    traj = assemble_and_run(run_config, system=system, ctx=ctx)

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
    return 0
