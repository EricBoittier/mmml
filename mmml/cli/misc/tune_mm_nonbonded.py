"""``mmml tune-mm-nonbonded``: fit CGenFF LJ/charge scales of the ML/MM tail.

Two stages::

    # 1. teacher + student interaction labels of liquid frames (GPU helps)
    mmml tune-mm-nonbonded label --frames box/seed_*/traj.extxyz \\
        --teacher pet-omol-l.pt --student student.json --out-dir tune/ --stride 10

    # 2. fit (CPU, seconds-minutes): scales + bootstrap + cohesion budget
    mmml tune-mm-nonbonded fit --labels tune/ --valid-groups 10,18,19,28,31 \\
        --n-boot 10 --out-json tune/lj_elec_tune.json --figs-dir tune/figs

See :mod:`mmml.models.mm_nonbonded_tune` for the model and
:mod:`mmml.distill.box_cohesion` for the labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def _parse_ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


def _parse_grid(text: str | None) -> list[tuple[float, float]]:
    out = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            on, width = item.split(":")
            out.append((float(on), float(width)))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml tune-mm-nonbonded",
        description="Fit per-type CGenFF LJ (eps/Rmin) scales and a charge scale so "
        "student-ML + MM reproduces teacher interaction energies/forces of liquid frames.",
    )
    sub = p.add_subparsers(dest="stage", required=True)

    lab = sub.add_parser("label", help="teacher/student interaction labels of box frames")
    lab.add_argument("--frames", nargs="+", type=Path, required=True, help="extxyz trajectories")
    lab.add_argument("--teacher", type=Path, default=None, help="metatomic .pt teacher")
    lab.add_argument("--student", type=Path, default=None, help="PhysNet hybrid checkpoint")
    lab.add_argument("--out-dir", type=Path, required=True)
    lab.add_argument("--atoms-per-molecule", type=int, default=9)
    lab.add_argument("--phase", default="md", help="extxyz 'phase' key to keep ('' = all)")
    lab.add_argument("--stride", type=int, default=10)
    lab.add_argument("--max-per-file", type=int, default=None)
    lab.add_argument("--r-pair-max", type=float, default=8.0,
                     help="store student pairs up to this COM distance (max refittable mm_switch_on)")
    lab.add_argument("--device", default=None, help="teacher torch device (cuda/cpu)")
    lab.add_argument("--teacher-max-atoms", type=int, default=4096)
    lab.add_argument("--student-batch", type=int, default=512)
    lab.add_argument("--overwrite", action="store_true")
    lab.add_argument("--box-from-file", action="store_true",
                     help="frames were sampled by the teacher: take box E/F from the extxyz "
                     "(only monomers are evaluated)")
    lab.add_argument("--group-every", type=int, default=None,
                     help="frames without a 'seed' key: bootstrap group = block of this many frames")
    lab.add_argument("--tag", default=None, help="label file name stem (default: from path)")
    lab.add_argument("--no-box-forces", action="store_true",
                     help="teacher box energy only (large boxes do not fit GPU memory with forces)")
    lab.add_argument("--time-budget-s", type=float, default=None,
                     help="do not start another trajectory file after this many seconds")

    fit = sub.add_parser("fit", help="fit scales, bootstrap, report cohesion budget")
    fit.add_argument("--labels", nargs="+", type=Path, required=True,
                     help="label .npz files or directories of them")
    fit.add_argument("--out-json", type=Path, required=True)
    fit.add_argument("--figs-dir", type=Path, default=None)
    fit.add_argument("--sidecar", type=Path, default=None,
                     help="write a hybrid_mm.json-style LJ-scale sidecar for MD")
    fit.add_argument("--cache-dir", type=Path, default=None, help="feature cache (default: next to labels)")
    fit.add_argument("--mm-switch-on", type=float, default=6.0)
    fit.add_argument("--mm-switch-width", type=float, default=5.0)
    fit.add_argument("--ml-switch-width", type=float, default=1.5)
    fit.add_argument("--valid-groups", default="", help="held-out seeds (comma list)")
    fit.add_argument("--n-boot", type=int, default=10)
    fit.add_argument("--boot-seed", type=int, default=0)
    fit.add_argument("--no-forces", action="store_true", help="energy-only fit")
    fit.add_argument("--force-weight", type=float, default=1.0)
    fit.add_argument("--sigma-e", type=float, default=0.05, help="kcal/mol per molecule")
    fit.add_argument("--sigma-f", type=float, default=1.0, help="kcal/mol/A")
    fit.add_argument("--tau-eps", type=float, default=0.5)
    fit.add_argument("--tau-rmin", type=float, default=0.03)
    fit.add_argument("--tau-charge", type=float, default=0.1)
    fit.add_argument("--no-charge", action="store_true", help="keep charges at CGenFF")
    fit.add_argument("--no-rmin", action="store_true", help="keep Rmin at CGenFF")
    fit.add_argument("--underlay", action="store_true",
                     help="also fit the ML-region underlay variant (kappa_elec, kappa_disp)")
    fit.add_argument("--handoff-grid", default="",
                     help="extra 'on:width,...' handoff windows to refit (energy-only)")
    fit.add_argument("--dimers", type=Path, default=None,
                     help="teacher dimer NPZ (R, N, kind, E_int eV) for a 2-body check of the "
                     "MM-only region (r_COM >= mm_switch_on)")
    fit.add_argument("--dimer-max", type=int, default=20000)
    fit.add_argument("--pr-url", default=None, help="recorded in the JSON")
    return p


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def _run_label(args: argparse.Namespace) -> int:
    from mmml.distill.box_cohesion import (
        PhysNetPairEvaluator,
        iter_box_frames,
        label_frame,
        save_labels,
    )

    if args.teacher is None or args.student is None:
        raise SystemExit("label needs --teacher and --student")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from mmml.distill.batched_teacher import BatchedMetatomicTeacher

    teacher = BatchedMetatomicTeacher(
        args.teacher, device=args.device, max_atoms_per_batch=args.teacher_max_atoms
    )
    student = PhysNetPairEvaluator(
        args.student, max_atoms=2 * args.atoms_per_molecule, batch_size=args.student_batch
    )
    meta = {
        "teacher": str(args.teacher),
        "student": str(args.student),
        "r_pair_max": float(args.r_pair_max),
        "phase": args.phase,
        "stride": int(args.stride),
    }
    t_start = time.time()
    for path in args.frames:
        if args.time_budget_s is not None and time.time() - t_start > args.time_budget_s:
            print("time budget reached; stopping before", path, flush=True)
            break
        stem = args.tag or (path.parent.name if path.stem == "traj" else path.stem)
        out = args.out_dir / f"labels_{stem}.npz"
        if out.exists() and not args.overwrite:
            print(f"skip {out} (exists)", flush=True)
            continue
        t0 = time.time()
        frames = []
        for fr in iter_box_frames(
            [path], atoms_per_molecule=args.atoms_per_molecule, phase=args.phase or None,
            stride=args.stride, max_per_file=args.max_per_file, group_every=args.group_every,
        ):
            frames.append(label_frame(fr, teacher=teacher, student=student,
                                      r_pair_max=args.r_pair_max,
                                      box_forces=not args.no_box_forces,
                                      box_from_file=args.box_from_file))
            last = frames[-1]
            n = last["mols"].shape[0]
            print(f"{stem} frame {fr.index}: teacher E_int/N {last['teacher_e_int'] / n:+.3f}"
                  f"  pairs {len(last['pairs'])}", flush=True)
        if frames:
            save_labels(out, frames, meta=dict(meta, source=str(path)))
        print(f"wrote {out} ({len(frames)} frames, {time.time() - t0:.0f}s)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def _label_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out += sorted(p.glob("labels_*.npz")) if p.is_dir() else [p]
    if not out:
        raise SystemExit("no label files found")
    return out


def _features(frames, ff, switch, *, with_forces, cache: Path | None):
    from mmml.distill.box_cohesion import frame_features_from_labels
    from mmml.models.mm_nonbonded_tune import load_features, save_features

    if cache is not None and cache.exists():
        data = load_features(cache)
        if len(data.G) == len(frames) and data.has_forces >= with_forces:
            return data
    data = frame_features_from_labels(frames, ff, switch, with_forces=with_forces, progress=True)
    if cache is not None:
        save_features(cache, data)
    return data


def _evaluate(ff, params, data) -> dict:
    from mmml.models.mm_nonbonded_tune import (
        cohesion_budget,
        energy_rmse_per_mol,
        predict_energy,
        predict_force_rmse,
    )

    e_mm = predict_energy(ff, params, data)
    hyb = (data.e_ml + e_mm) / data.n_mol
    ref = data.e_teacher / data.n_mol
    return {
        "n_frames": int(len(data.G)),
        "E_int_rmse_kcal_per_mol": energy_rmse_per_mol(ff, params, data),
        "E_int_mae_kcal_per_mol": float(np.mean(np.abs(hyb - ref))),
        "E_int_mean_err_kcal_per_mol": float(np.mean(hyb - ref)),
        "F_int_rmse_kcal_per_A": predict_force_rmse(ff, params, data),
        "cohesion_budget": cohesion_budget(ff, params, data),
    }


def _variant(ff, name, prior, train, valid, alldata, n_boot, boot_seed) -> dict:
    from mmml.models.mm_nonbonded_tune import (
        TuneParams,
        bootstrap_fit,
        fit_parameters,
        summarize_params,
    )

    t0 = time.time()
    best = fit_parameters(ff, train, prior)
    boots = bootstrap_fit(ff, train, prior, n_boot=n_boot, seed=boot_seed) if n_boot else []
    cg = TuneParams.cgenff(ff.n_types)
    out = {
        "name": name,
        "prior": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(prior).items()},
        "params": best.params.to_dict(ff.type_names),
        "params_bootstrap": summarize_params(ff, boots or [best]),
        "at_bound": best.at_bound,
        "optimizer": {"success": best.success, "message": best.message, "loss": best.loss},
        "fit_seconds": time.time() - t0,
        "cgenff": {"train": _evaluate(ff, cg, train), "valid": _evaluate(ff, cg, valid) if valid else None,
                   "all": _evaluate(ff, cg, alldata)},
        "tuned": {"train": _evaluate(ff, best.params, train),
                  "valid": _evaluate(ff, best.params, valid) if valid else None,
                  "all": _evaluate(ff, best.params, alldata)},
        "_params_obj": best.params,
    }
    if boots:
        out["valid_rmse_bootstrap"] = [
            _evaluate(ff, b.params, valid if valid else train)["E_int_rmse_kcal_per_mol"] for b in boots
        ]
    return out


def _dimer_check(args, ff, switch, results) -> dict:
    """Two-body check where the hybrid is MM-only: teacher dimer E_int vs w_MM E_MM."""
    from mmml.models.mm_nonbonded_tune import (
        EV_TO_KCAL,
        TuneParams,
        dimer_features,
        mm_coefficients,
    )

    d = np.load(args.dimers, allow_pickle=True)
    a = ff.n_atoms
    sel = np.flatnonzero((d["kind"] == 1) & (d["N"] == 2 * a))
    r_com = np.asarray(d["r_com"])[sel] if "r_com" in d.files else None
    R = np.asarray(d["R"])[sel, : 2 * a]
    if r_com is None:
        r_com = np.linalg.norm(R[:, a:].mean(1) - R[:, :a].mean(1), axis=1)
    keep = r_com >= switch.mm_switch_on
    sel, R, r_com = sel[keep], R[keep], r_com[keep]
    if len(sel) > args.dimer_max:
        pick = np.random.default_rng(0).choice(len(sel), args.dimer_max, replace=False)
        sel, R, r_com = sel[pick], R[pick], r_com[pick]
    e_ref = np.asarray(d["E_int"])[sel] * EV_TO_KCAL
    G = dimer_features(ff, switch, R)
    out = {"n": int(len(sel)), "r_com_min": float(switch.mm_switch_on),
           "teacher_mean": float(np.mean(e_ref)), "r_com": r_com.tolist(),
           "teacher": e_ref.tolist(), "models": {}}
    models = {"cgenff": TuneParams.cgenff(ff.n_types)}
    for r in results:
        models[r["name"]] = r["_params_obj"]
    for name, p in models.items():
        e = G @ np.asarray(mm_coefficients(ff, p))
        out["models"][name] = {
            "rmse": float(np.sqrt(np.mean((e - e_ref) ** 2))),
            "mean_err": float(np.mean(e - e_ref)),
            "pred": e.tolist(),
        }
    return out


def _run_fit(args: argparse.Namespace) -> int:
    import jax

    jax.config.update("jax_enable_x64", True)
    from mmml.distill.box_cohesion import load_labels
    from mmml.models.mm_nonbonded_tune import (
        MonomerNonbonded,
        PriorConfig,
        SwitchConfig,
        lj_sidecar_payload,
    )

    files = _label_files(args.labels)
    frames = []
    for f in files:
        frames += load_labels(f)
    print(f"{len(frames)} frames from {len(files)} label files", flush=True)
    z = np.asarray(frames[0]["z_mol"])
    ff = MonomerNonbonded.from_cgenff(z, np.asarray(frames[0]["mols"])[0])
    r_store = min(float(np.max(f["r_com"])) for f in frames if len(f["r_com"]))
    switch = SwitchConfig(args.mm_switch_on, args.mm_switch_width, args.ml_switch_width)
    if switch.mm_switch_on > r_store + 0.05:
        raise SystemExit(f"mm_switch_on {switch.mm_switch_on} > stored student pairs ({r_store:.2f} A)")

    cache_dir = args.cache_dir or (files[0].parent / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"on{switch.mm_switch_on:g}_w{switch.mm_switch_width:g}_ml{switch.ml_switch_width:g}_n{len(frames)}"
    has_f = all("teacher_f_int" in f for f in frames)
    if not has_f and not args.no_forces:
        print("labels carry no teacher box forces -> energy-only fit", flush=True)
        args.no_forces = True
    data = _features(frames, ff, switch, with_forces=not args.no_forces,
                     cache=cache_dir / f"features_{tag}.npz")

    valid_groups = set(_parse_ints(args.valid_groups))
    is_valid = np.array([g in valid_groups for g in data.group])
    train = data.subset(np.flatnonzero(~is_valid))
    valid = data.subset(np.flatnonzero(is_valid)) if is_valid.any() else None
    print(f"train {len(train.G)} frames / valid {0 if valid is None else len(valid.G)}", flush=True)

    base = PriorConfig(
        tau_eps=args.tau_eps, tau_rmin=args.tau_rmin, tau_charge=args.tau_charge,
        sigma_e=args.sigma_e, sigma_f=args.sigma_f,
        force_weight=0.0 if args.no_forces else args.force_weight,
        fit_rmin=not args.no_rmin, fit_charge=not args.no_charge,
    )
    variants = [
        ("tail-min", replace(base, tie_eps=True, fit_rmin=False)),
        ("tail", base),
    ]
    if args.underlay:
        variants.append(("tail+underlay", replace(base, fit_kappa_elec=True, fit_kappa_disp=True)))
    results = []
    for name, prior in variants:
        print(f"fitting variant {name} ...", flush=True)
        res = _variant(ff, name, prior, train, valid, data, args.n_boot, args.boot_seed)
        print(json.dumps({k: res[k] for k in ("params", "at_bound")}, indent=1), flush=True)
        results.append(res)

    grid = []
    for on, width in _parse_grid(args.handoff_grid):
        sw = SwitchConfig(on, width, args.ml_switch_width)
        if on > r_store + 0.05:
            print(f"skip handoff {on}:{width} (> stored pairs)")
            continue
        t = f"on{on:g}_w{width:g}_ml{args.ml_switch_width:g}_n{len(frames)}"
        d = _features(frames, ff, sw, with_forces=False, cache=cache_dir / f"features_{t}.npz")
        tr = d.subset(np.flatnonzero(~is_valid))
        va = d.subset(np.flatnonzero(is_valid)) if is_valid.any() else None
        gname, gprior = variants[0]
        r = _variant(ff, f"handoff {on:g}:{width:g}", replace(gprior, force_weight=0.0), tr, va, d, 0, 0)
        r["switch"] = sw.to_dict()
        r["variant"] = gname
        grid.append(r)
        print(f"handoff {on}:{width}: valid RMSE cgenff "
              f"{(r['cgenff']['valid'] or r['cgenff']['train'])['E_int_rmse_kcal_per_mol']:.3f} -> "
              f"tuned {(r['tuned']['valid'] or r['tuned']['train'])['E_int_rmse_kcal_per_mol']:.3f}")

    dimer_check = None
    if args.dimers is not None:
        dimer_check = _dimer_check(args, ff, switch, results)

    # Unswitched student pair sum inside 7.5 A (the "ML pairs < 7.5 A" convention
    # of earlier budgets) -- for reconciling with numbers quoted elsewhere.
    ml_lt75 = [float(np.sum(f["student_e_pair"][f["r_com"] < 7.5])) / f["mols"].shape[0]
               for f in frames]
    reference = {
        "student_ml_pairs_unswitched_lt7p5_per_mol": float(np.mean(ml_lt75)),
        "teacher_E_int_per_mol": float(np.mean(data.e_teacher / data.n_mol)),
        "teacher_E_int_per_mol_std_over_frames": float(np.std(data.e_teacher / data.n_mol)),
    }

    payload = {
        "dimer_check": dimer_check,
        "reference_convention": reference,
        "monomer": ff.to_dict(),
        "switch": switch.to_dict(),
        "n_frames": int(len(data.G)),
        "groups": sorted(int(g) for g in np.unique(data.group)),
        "valid_groups": sorted(valid_groups),
        "labels": [str(f) for f in files],
        "label_meta": json.loads(str(np.load(files[0])["meta"])),
        "variants": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        "handoff_grid": [{k: v for k, v in r.items() if not k.startswith("_")} for r in grid],
        "pr_url": args.pr_url,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out_json}")
    if args.sidecar is not None:
        args.sidecar.write_text(json.dumps(lj_sidecar_payload(ff, results[0]["_params_obj"]), indent=2))
        print(f"wrote {args.sidecar}")
    if args.figs_dir is not None:
        from mmml.analysis.mm_nonbonded_tune_plots import make_tune_figures

        for path in make_tune_figures(payload, ff, results, data, is_valid, args.figs_dir):
            print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "label":
        return _run_label(args)
    return _run_fit(args)


if __name__ == "__main__":
    sys.exit(main())
