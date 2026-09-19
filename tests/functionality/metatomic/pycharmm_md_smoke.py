#!/usr/bin/env python3
"""Serial PyCHARMM + metatomic ENER / SD / short NVE smoke.

Requires a built serial ``libcharmm.so`` (``rebuild_charmm_mlpot.sh --no-mpi``)
and a TorchScript AtomisticModel (``.pt``). Does **not** download Hub checkpoints.

Pass: CHARMM ``ENER`` reports a finite USER term; SD and NVE complete; restart
and DCD exist.

Example::

    export CHARMM_HOME=$PWD/setup/charmm
    export CHARMM_LIB_DIR=$CHARMM_HOME/lib
    export MMML_NO_CHARMM_MPI=1 MMML_NO_MPI_RERUN=1
    export MMML_METATOMIC_DEVICE=cpu JAX_PLATFORMS=cpu
    uv run python tests/functionality/metatomic/pycharmm_md_smoke.py \\
      --checkpoint /path/to/pet-mad-xs-v1.5.0.pt --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

NVE_TIMESTEP_PS = 0.00025


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonzero_terms(terms: dict[str, float], *, extra: tuple[str, ...] = ()) -> dict[str, float]:
    keys = set(extra) | {"ENER", "USER", "TOTE", "TOTKE", "GRMS"}
    out: dict[str, float] = {}
    for key, value in terms.items():
        if key in keys or abs(float(value)) > 1.0e-8:
            out[str(key)] = float(value)
    return out


def main() -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        add_charmm_output_args,
        add_cluster_args,
        add_dcd_save_args,
        add_dynamics_stability_args,
        add_monomer_constraint_args,
        apply_charmm_output_from_args,
        build_cluster_from_args,
        charmm_energy_row,
        print_cluster_geometry_summary,
        print_header,
        resolve_checkpoint,
        resolve_dcd_nsavc,
        resolve_dynamics_print_kwargs,
        resolve_echeck_from_args,
        resolve_fix_resids,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_cluster_args(parser)
    add_charmm_output_args(parser)
    add_dcd_save_args(parser)
    add_dynamics_stability_args(parser)
    add_monomer_constraint_args(parser, for_dynamics=True)
    parser.add_argument("--run", action="store_true", help="Execute ENER + SD + NVE")
    parser.add_argument("--nstep", type=int, default=5, help="NVE steps")
    parser.add_argument(
        "--temp",
        type=float,
        default=300.0,
        help="Initial Maxwell-Boltzmann temperature (K)",
    )
    parser.add_argument(
        "--metatomic-eval-mode",
        choices=("fragments", "whole_system"),
        default="fragments",
    )
    parser.add_argument(
        "--include-mm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional JAX MM spherical_fn (default: ML-only USER)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/functionality/metatomic/output/pycharmm_md"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write a machine-readable report (default: <out-dir>/report.json)",
    )
    args = parser.parse_args()
    args.ml_potential_mode = "metatomic"
    args.do_ml = True
    args.do_ml_dimer = True
    args.skip_ml_dimers = False

    print_header("PyCHARMM + metatomic MD smoke")
    if not args.run:
        print("STUB: pass --run to execute.")
        return 0

    ckpt = resolve_checkpoint(args.checkpoint)
    report: dict = {
        "ok": False,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "eval_mode": str(args.metatomic_eval_mode),
        "include_mm": bool(args.include_mm),
        "residue": str(args.residue),
        "n_molecules": int(args.n_molecules),
        "spacing_A": float(args.spacing),
        "mini_nstep": int(args.mini_nstep),
        "nve_nstep": int(args.nstep),
        "timestep_ps": NVE_TIMESTEP_PS,
    }
    json_out = (args.json_out or (args.out_dir / "report.json")).resolve()

    def _write_report() -> None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {json_out}")

    try:
        z, r, n_atoms = build_cluster_from_args(args)
        n_mol = int(args.n_molecules)
        if n_atoms % n_mol != 0:
            raise SystemExit(
                f"non-uniform residue size: n_atoms={n_atoms} n_molecules={n_mol}"
            )
        per = [n_atoms // n_mol] * n_mol
        report["n_atoms"] = int(n_atoms)
        report["atoms_per_monomer"] = per
        print_cluster_geometry_summary(r, n_mol)

        import ase
        import pycharmm.energy as energy
        from mmml.interfaces.pycharmmInterface.mlpot import (
            CharmmTrajectoryFiles,
            MinimizeWithMlpotConfig,
            MetatomicMlpotModel,
            build_nve_dynamics,
            get_charmm_positions_array,
            load_physnet_mlpot_bundle,
            minimize_with_mlpot,
            register_mlpot,
            run_dynamics_with_io,
            select_all_atoms,
            setup_default_nbonds,
            sync_charmm_positions,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.setup import assert_mlpot_user_active

        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{str(args.residue).lower()}_{n_mol}mer"
        res_path = out_dir / f"nve_{tag}.res"
        dcd_path = out_dir / f"nve_{tag}.dcd"
        report["restart"] = str(res_path)
        report["dcd"] = str(dcd_path)

        mini_nprint = apply_charmm_output_from_args(args)
        dyn_print = resolve_dynamics_print_kwargs(args, nstep=args.nstep)
        echeck = resolve_echeck_from_args(args)
        dcd_nsavc = resolve_dcd_nsavc(
            dcd_nsavc=args.dcd_nsavc,
            timestep_ps=NVE_TIMESTEP_PS,
            nstep=args.nstep,
        )
        setup_default_nbonds()
        sync_charmm_positions(r)

        atoms = ase.Atoms(numbers=z, positions=r)
        _, _, pyCModel = load_physnet_mlpot_bundle(
            ckpt,
            n_atoms,
            atoms,
            n_monomers=n_mol,
            atoms_per_monomer=per,
            verbose=True,
            args=args,
        )
        report["model_type"] = type(pyCModel).__name__
        if not isinstance(pyCModel, MetatomicMlpotModel):
            raise SystemExit(
                f"expected MetatomicMlpotModel, got {type(pyCModel).__name__}"
            )
        report["eval_mode_resolved"] = str(pyCModel._eval_mode)
        report["do_mm"] = bool(pyCModel._do_mm)

        ctx = register_mlpot(pyCModel, z, select_all_atoms(), verbose=True)
        try:
            user = assert_mlpot_user_active(ctx, context="metatomic smoke ENER")
            energy.show()
            before = charmm_energy_row()
            report["user_kcal_before_sd"] = float(user)
            report["energy_before_sd"] = _nonzero_terms(before)
            print(
                f"USER before SD: {user:.6f} kcal/mol  "
                f"ENER={before.get('ENER', float('nan')):.6f}"
            )

            fix_resids = resolve_fix_resids(args)
            from mmml.interfaces.pycharmmInterface.mlpot import select_by_resids

            fix_sel = select_by_resids(fix_resids) if fix_resids else None
            if not args.no_pre_minimize:
                print(f"SD: {args.mini_nstep} steps/pass")
                minimize_with_mlpot(
                    MinimizeWithMlpotConfig(
                        fixed_ml_selection=fix_sel,
                        nstep=args.mini_nstep,
                        nprint=mini_nprint,
                        verbose=not args.quiet,
                        reference_positions=r,
                        pyCModel=pyCModel,
                        save=False,
                        show_energy=True,
                        skip_if_crd_exists=False,
                    )
                )
                sync_charmm_positions(get_charmm_positions_array())
                after_sd = charmm_energy_row()
                report["energy_after_sd"] = _nonzero_terms(after_sd)
                report["user_kcal_after_sd"] = float(after_sd.get("USER", 0.0))

            print(f"NVE: {args.nstep} steps @ {NVE_TIMESTEP_PS} ps")
            io = CharmmTrajectoryFiles(restart_write=res_path, trajectory=dcd_path)
            kw = build_nve_dynamics(
                timestep_ps=NVE_TIMESTEP_PS,
                duration_ps=args.nstep * NVE_TIMESTEP_PS,
                save_interval_ps=NVE_TIMESTEP_PS * max(1, dcd_nsavc),
                restart=False,
                temp=args.temp,
                nprint=min(int(dyn_print["nprint"]), max(1, args.nstep)),
                iprfrq=min(int(dyn_print["iprfrq"]), max(1, args.nstep)),
                isvfrq=dyn_print["isvfrq"],
                echeck=echeck,
                use_pbc=False,
            )
            kw["new"] = True
            kw["start"] = True
            kw["nstep"] = args.nstep
            kw["nsavc"] = max(1, dcd_nsavc)
            run_dynamics_with_io(kw, io)
            print("CHARMM energy after NVE:")
            energy.show()
            after = charmm_energy_row()
            report["energy_after_nve"] = _nonzero_terms(after)
            report["user_kcal_after_nve"] = float(after.get("USER", 0.0))
        finally:
            ctx.unset()

        missing = [str(p) for p in (res_path, dcd_path) if not p.is_file()]
        report["missing_outputs"] = missing
        if missing:
            raise SystemExit(f"missing outputs: {missing}")
        user_after = float(report.get("user_kcal_after_nve") or 0.0)
        if not np.isfinite(user_after) or abs(user_after) < 1.0e-12:
            raise SystemExit(f"USER after NVE is missing/zero: {user_after}")
        report["ok"] = True
        print("PASS: metatomic USER active through ENER, SD, and NVE.")
        _write_report()
        return 0
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report()
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
