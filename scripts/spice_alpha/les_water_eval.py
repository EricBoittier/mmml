#!/usr/bin/env python
"""Score EFieldPhysNet checkpoints on periodic LES water sets (forces, energy fluctuations, BEC, cell α).

Each frame is one periodic cell packed as one "molecule" (edges carry lattice-image shifts). Predictions:
  forces  F = −∂E/∂R                                   vs DFT forces (eV/Å)
  energy  E per cell, compared after removing the mean offset (different functional/reference)
  BEC     Z*_i,ab = ∂μ_a/∂R_ib (reverse mode over the cell dipole, molecules kept whole)
  α       ∂μ/∂E (Bohr³ internally, reported in Å³ per cell and per H2O)

--selftest checks the periodic path first: a molecule in a large box must reproduce the non-periodic
E/F/μ/α, and wrapping atoms by lattice vectors must leave E and F unchanged.

Usage: python scripts/spice_alpha/les_water_eval.py OUT_JSON --model NAME CKPT_DIR PARAMS [--model ...]
           --data NAME RAGGED_DIR [--data ...] [--max-frames N] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BOHR3_PER_A3 = 6.748334  # (Å/Bohr)³


def build(ckpt_dir: Path, params_path: Path):
    from mmml.models.efield.training import EFieldPhysNet, coulomb_sign_from_config, load_params

    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = EFieldPhysNet(features=cfg["features"], max_degree=cfg["max_degree"], num_iterations=cfg["num_iterations"],
                          num_basis_functions=cfg["num_basis_functions"], cutoff=cfg["cutoff"], max_atomic_number=55,
                          include_pseudotensors=cfg.get("pseudotensors", True), field_scale=0.001, zbl=False,
                          electrostatics_damping_sigma=4.0, packed=True, charge_activation=cfg.get("charge_activation", "silu"),
                          atomic_dipoles=cfg.get("atomic_dipoles", True), parity_correct_field=cfg.get("parity_correct", False),
                          parity_correct_dipole=cfg.get("parity_correct", False), strict_pseudotensors=cfg.get("strict_pseudotensors", False),
                          coulomb_sign=coulomb_sign_from_config(cfg))
    return model, {"params": load_params(params_path)["params"]}, cfg


def make_fns(model, params, M):
    import jax
    import jax.numpy as jnp
    from mmml.models.efield.packed import _forward, _polar

    def efd(batch):
        def f(pos):
            e, d, _ = _forward(model.apply, params, {**batch, "positions": pos}, M)
            return -jnp.sum(e * batch["mol_mask"]), (e, d)
        (_, (e, d)), F = jax.value_and_grad(f, has_aux=True)(batch["positions"])
        return e, d, F

    def bec(batch):  # (A, 3, 3): [i, a, b] = ∂μ_a(cell 0)/∂R_ib
        J = jax.jacrev(lambda pos: _forward(model.apply, params, {**batch, "positions": pos}, M)[1][0])(batch["positions"])
        return jnp.transpose(J, (1, 0, 2))

    return jax.jit(efd), jax.jit(lambda b: _polar(model.apply, params, b, M, 0.001)), jax.jit(bec)


def spec_for(data, cutoff, frames, M=2):
    from mmml.models.efield.packed import PackSpec, frame_edges, frame_edges_pbc

    off = data["offsets"]
    n_at, n_e = 0, 0
    for f in frames[:20]:
        R = data["R"][off[f]:off[f + 1]]
        ne = len(frame_edges_pbc(R, data["cell"][f], cutoff)[0]) if "cell" in data else len(frame_edges(R, cutoff)[0])
        n_at, n_e = max(n_at, len(R)), max(n_e, ne)
    return PackSpec(M, int(np.ceil((n_at + 1) / 64) * 64), int(np.ceil(n_e * 1.25 / 1024) * 1024), cutoff)


def one_frame_batches(data, frames, spec):
    import jax.numpy as jnp
    from mmml.models.efield.packed import iter_packed_batches

    for b in iter_packed_batches(data, np.asarray(frames), spec):
        assert int(b["mol_mask"].sum()) == 1
        yield {k: jnp.asarray(v) for k, v in b.items()}


def selftest(model, params, cutoff):
    """Periodic path == non-periodic for an isolated molecule; E/F invariant to wrapping."""
    from mmml.data.spice_alpha_ragged import offsets_from_counts

    rng = np.random.default_rng(0)
    # water trimer-ish cluster
    Z = np.array([8, 1, 1, 8, 1, 1, 8, 1, 1], np.int8)
    base = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], np.float64)
    R = np.concatenate([base + o for o in ([0, 0, 0], [2.8, 0.3, 0.2], [1.2, 2.6, -0.4])]) + rng.normal(0, 0.03, (9, 3))

    def mk(R, cell=None):
        d = dict(Z=Z, R=R.astype(np.float32), F=np.zeros_like(R, np.float32), N=np.array([9], np.int16), E=np.zeros(1),
                 D=np.zeros((1, 3), np.float32), Q=np.zeros(1, np.float32), polar=np.zeros((1, 3, 3), np.float32),
                 mol=np.zeros(1, np.int64), subset=np.zeros(1, np.int8))
        if cell is not None:
            d["cell"] = cell[None].astype(np.float32)
        d["offsets"] = offsets_from_counts(d["N"])
        return d

    out = {}
    big = np.eye(3) * (2 * cutoff + 15.0)
    res = {}
    for name, d in (("open", mk(R)), ("pbc_big_box", mk(R, big))):
        spec = spec_for(d, cutoff, [0])
        efd, pol, bec = make_fns(model, params, spec.max_molecules)
        b = next(one_frame_batches(d, [0], spec))
        e, mu, F = efd(b)
        res[name] = (float(e[0]), np.asarray(F[:9]), np.asarray(mu[0]), np.asarray(pol(b)[0]))
    o, p = res["open"], res["pbc_big_box"]
    out["big_box"] = {"dE": abs(o[0] - p[0]), "dF": float(np.abs(o[1] - p[1]).max()),
                      "dmu": float(np.abs(o[2] - p[2]).max()), "dalpha": float(np.abs(o[3] - p[3]).max())}
    # small box (cutoff > L/2 in one direction) with atoms wrapped by random lattice vectors
    cell = np.array([[7.5, 0, 0], [0.4, 8.0, 0], [0.0, 0.3, 9.0]])
    R2 = R + 0.5
    wrapped = R2 + rng.integers(-2, 3, (9, 1)) * cell[0] + rng.integers(-1, 2, (9, 1)) * cell[2]
    vals = []
    for RR in (R2, wrapped):
        d = mk(RR, cell)
        spec = spec_for(d, cutoff, [0])
        efd, _, _ = make_fns(model, params, spec.max_molecules)
        e, _, F = efd(next(one_frame_batches(d, [0], spec)))
        vals.append((float(e[0]), np.asarray(F[:9])))
    out["wrap_invariance"] = {"dE": abs(vals[0][0] - vals[1][0]), "dF": float(np.abs(vals[0][1] - vals[1][1]).max()),
                              "E_box_minus_open": vals[0][0] - o[0]}
    return out


def evaluate(model, params, cfg, data, frames, want_bec, want_alpha):
    off = data["offsets"]
    spec = spec_for(data, cfg["cutoff"], frames)
    efd, pol, bec = make_fns(model, params, spec.max_molecules)
    E, F, Fr, B, Br, A, Ar, Zall = [], [], [], [], [], [], [], []
    t0 = time.time()
    for k, (f, b) in enumerate(zip(frames, one_frame_batches(data, frames, spec))):
        n = int(off[f + 1] - off[f])
        e, _, Fp = efd(b)
        E.append(float(e[0])); F.append(np.asarray(Fp[:n])); Fr.append(data["F"][off[f]:off[f + 1]])
        Zall.append(data["Z"][off[f]:off[f + 1]])
        if want_bec:
            B.append(np.asarray(bec(b)[:n])); Br.append(data["bec"][off[f]:off[f + 1]])
        if want_alpha:
            A.append(np.asarray(pol(b)[0])); Ar.append(data["polar"][f])
        if k == 0:
            print(f"    first frame {time.time() - t0:.1f}s (compile), spec {spec}", flush=True)
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    F, Fr, Z = np.concatenate(F), np.concatenate(Fr), np.concatenate(Zall)
    Eref = data["E"][frames]
    Ep, n_w = np.asarray(E), data["N"][frames] / 3.0
    dE = (Ep - Ep.mean()) - (Eref - Eref.mean())
    r = {"n_frames": len(frames), "time_s": time.time() - t0,
         "forces_rmse": rm(F - Fr), "forces_mae": float(np.mean(np.abs(F - Fr))),
         "forces_rmse_O": rm((F - Fr)[Z == 8]), "forces_rmse_H": rm((F - Fr)[Z == 1]),
         "forces_ref_std": float(Fr.std()),
         "energy_centered_rmse_meV_per_h2o": rm(dE / n_w) * 1000, "energy_ref_std_meV_per_h2o": float((Eref / n_w).std() * 1000),
         "energy_pearson": float(np.corrcoef(Ep, Eref)[0, 1]) if len(frames) > 2 else None}
    if want_bec:
        B, Br = np.concatenate(B), np.concatenate(Br)
        for tag, ref in (("as_file", Br), ("transposed", Br.transpose(0, 2, 1))):
            r[f"bec_rmse_{tag}"] = rm(B - ref)
        for z, s in ((8, "O"), (1, "H")):
            m = Z == z
            r[f"bec_rmse_{s}"] = rm(B[m] - Br[m])
            r[f"bec_mean_{s}_pred"] = B[m].mean(0).round(4).tolist()
            r[f"bec_mean_{s}_ref"] = Br[m].mean(0).round(4).tolist()
            r[f"bec_iso_{s}_pred_ref"] = [float(np.trace(B[m], axis1=1, axis2=2).mean() / 3),
                                          float(np.trace(Br[m], axis1=1, axis2=2).mean() / 3)]
        r["bec_acoustic_sum_max"] = float(np.abs(B.reshape(len(frames), -1, 3, 3).sum(1)).max())
    if want_alpha:
        A, Ar = np.asarray(A) / BOHR3_PER_A3, np.asarray(Ar) / BOHR3_PER_A3  # Å³ per cell
        nw = n_w[:, None, None]
        r["alpha_rmse_A3_per_h2o"] = rm((A - Ar) / nw)
        r["alpha_diag_rmse_A3_per_h2o"] = rm(np.diagonal((A - Ar) / nw, axis1=1, axis2=2))
        r["alpha_iso_pred_ref_A3_per_h2o"] = [float((np.trace(A, axis1=1, axis2=2) / 3 / n_w).mean()),
                                              float((np.trace(Ar, axis1=1, axis2=2) / 3 / n_w).mean())]
    preds = {"frames": np.asarray(frames), "E": Ep, "E_ref": Eref, "F": F, "F_ref": Fr, "Z": Z,
             **({"bec": B, "bec_ref": Br} if want_bec else {}), **({"alpha_A3": A, "alpha_ref_A3": Ar} if want_alpha else {})}
    return r, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "CKPT_DIR", "PARAMS"), required=True)
    ap.add_argument("--data", nargs=2, action="append", metavar=("NAME", "RAGGED_DIR"), default=[])
    ap.add_argument("--split", nargs=2, action="append", metavar=("DATA_NAME", "SPLIT_NPZ"), default=[],
                    help="score only the test frames of this split for DATA_NAME")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    from mmml.data.spice_alpha_ragged import load_ragged

    splits = {k: np.load(v)["test"] for k, v in args.split}
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    for mname, ckpt, prm in args.model:
        model, params, cfg = build(Path(ckpt), Path(prm))
        res.setdefault(mname, {})
        if args.selftest:
            st = selftest(model, params, cfg["cutoff"])
            print(mname, "selftest", json.dumps(st), flush=True)
            res[mname]["selftest"] = st
        for dname, ddir in args.data:
            data = load_ragged(sorted(Path(ddir).glob("*.npz")))
            frames = splits.get(dname, np.arange(len(data["N"])))
            if args.max_frames:
                frames = frames[:args.max_frames]
            print(f"  {mname} on {dname}: {len(frames)} frames", flush=True)
            r, preds = evaluate(model, params, cfg, data, frames, "bec" in data, bool(np.abs(data["polar"]).max() > 0))
            res[mname][dname] = r
            np.savez_compressed(args.out.parent / f"{args.out.stem}_{mname}_{dname}.npz", **preds)
            print("   ", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            args.out.write_text(json.dumps(res, indent=1))
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
