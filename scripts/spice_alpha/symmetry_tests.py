#!/usr/bin/env python
"""Symmetry, conservation and energy-derivative tests for a packed SPICE-α checkpoint.

For each frame and external field E (input units: Ef_input * 0.001 = field in
atomic units), evaluates the model on transformed copies of the geometry and
compares with the transformed reference prediction:

  rotation Q       R -> R Q^T, E -> Q E    expect  U inv., F -> QF, mu -> Q mu, alpha -> Q alpha Q^T
  inversion P      R -> -R,    E -> -E     expect  U inv., F -> -F, mu -> -mu, alpha -> alpha
  mirror s (z)     R -> R s,   E -> s E    expect  U inv., F -> sF, mu -> s mu, alpha -> s alpha s
  mirror, E fixed  R -> R s,   E -> E      (not a symmetry when E != 0; sensitivity only)
  translation, atom permutation             expect invariance / permuted outputs

Hidden features: per layer, scalars x[:,0,0] should be invariant under all of
the above; pseudoscalars x[:,1,0] invariant under rotations and sign-flipped
under inversion/mirror. Reported as the parity-odd and parity-even parts of the
pseudoscalars, ||(p(x) -/+ p(Px))/2|| / ||p(x)||.

Conservation / energy derivatives (no transformation): sum F, sum r x F, sum q,
alpha asymmetry, mu vs -dU/dE and alpha vs -d2U/dE2 (energy U in eV, E in V/Å),
and linear response mu(E) - mu(0) vs alpha(0) E.

Usage (GPU node):
  python scripts/spice_alpha/symmetry_tests.py CKPT_DIR PARAMS_JSON FRAMES_JSON OUT_NPZ
FRAMES_JSON: {"frames": [ragged frame indices], ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AU_FIELD_V_PER_A = 51.42206747632595
FIELDS_INPUT = (0.0, 1.0, 5.0, 10.0)  # Ef_input; x 0.001 au = 0, 0.05, 0.26, 0.51 V/Å


def rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("params", type=Path)
    ap.add_argument("frames_json", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-rot", type=int, default=3)
    ap.add_argument("--parity-fix", action="store_true",
                    help="evaluate the same weights with the parity-correct field embedding")
    ap.add_argument("--parity-fix-dipole", action="store_true", help="read atomic dipoles from the polar slot")
    ap.add_argument("--random-init", type=int, default=None, help="ignore PARAMS_JSON; random weights with this seed")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", False)
    from karml.data.spice_alpha_ragged import load_ragged
    from karml.models.efield.packed import PackSpec, iter_packed_batches
    from karml.models.efield.training import EFieldPhysNet, load_params

    cfg = json.loads((args.ckpt_dir / "config.json").read_text())
    shards = sorted(Path(cfg["ragged_dir"]).glob("*.npz"))
    data = load_ragged(shards)
    off = data["offsets"]
    frames = json.loads(args.frames_json.read_text())["frames"]
    pseudo = cfg.get("pseudotensors", True)
    model = EFieldPhysNet(features=cfg["features"], max_degree=cfg["max_degree"],
                          num_iterations=cfg["num_iterations"], num_basis_functions=cfg["num_basis_functions"],
                          cutoff=cfg["cutoff"], max_atomic_number=55, include_pseudotensors=pseudo,
                          field_scale=0.001, zbl=False, electrostatics_damping_sigma=4.0, packed=True,
                          charge_activation=cfg.get("charge_activation", "silu"),
                          atomic_dipoles=cfg.get("atomic_dipoles", True),
                          parity_correct_field=args.parity_fix or cfg.get("parity_correct", False),
                          parity_correct_dipole=args.parity_fix_dipole or cfg.get("parity_correct", False),
                          strict_pseudotensors=cfg.get("strict_pseudotensors", False))
    A = 128
    lrc = cfg.get("coulomb_cutoff") is not None
    spec = PackSpec(2, A, 16384, cfg["cutoff"], coulomb_cutoff=cfg.get("coulomb_cutoff"),
                    max_coulomb_edges=12100 if lrc else 0)
    conv = 0.001 * AU_FIELD_V_PER_A  # Ef_input -> V/Å

    def apply(pos, ef, b, capture=False):
        Ef = jnp.zeros((2, 3), pos.dtype).at[0].set(ef)
        kw = dict(atomic_numbers=b["atomic_numbers"], positions=pos, Ef=Ef, dst_idx_flat=b["dst_idx_flat"],
                  src_idx_flat=b["src_idx_flat"], batch_segments=b["batch_segments"], batch_size=2,
                  coulomb_dst_idx_flat=b.get("coulomb_dst_idx_flat"),
                  coulomb_src_idx_flat=b.get("coulomb_src_idx_flat"), mutable=["intermediates"])
        if capture:
            kw["capture_intermediates"] = True
        (energy, dipole), state = model.apply(params, **kw)
        return energy[0], dipole[0], state

    @jax.jit
    def evaluate(b, ef):
        pos = b["positions"]
        (u, (mu, state)), g = jax.value_and_grad(lambda p: (lambda r: (r[0], (r[1], r[2])))(apply(p, ef, b)),
                                                 has_aux=True)(pos)
        dmu = jax.jacfwd(lambda e: apply(pos, e, b)[1])(ef)              # (3,3) dmu_i/dEf_j
        du = jax.grad(lambda e: apply(pos, e, b)[0])(ef)
        d2u = jax.hessian(lambda e: apply(pos, e, b)[0])(ef)
        inter = apply(pos, ef, b, capture=True)[2]["intermediates"]
        feats = {}
        for name, v in inter.items():
            if isinstance(v, dict) and "__call__" in v:
                y = v["__call__"][0]
                if y.ndim == 4 and y.shape[0] == A:
                    feats[name] = y[:, :, 0, :]                            # l = 0: (A, P, F)
        return {"U": u, "F": -g, "mu": mu, "alpha": dmu / conv, "mu_E": -du / conv, "alpha_E": -d2u / conv ** 2,
                "q": state["intermediates"]["atomic_charges"][-1], "feats": feats}

    def batch_for(Z, R):
        d = {"Z": Z, "R": R.astype(np.float32), "F": np.zeros_like(R, np.float32), "E": np.zeros(1),
             "D": np.zeros((1, 3), np.float32), "polar": np.zeros((1, 3, 3), np.float32),
             "subset": np.zeros(1, np.int8), "offsets": np.array([0, len(Z)])}
        return {k: jnp.asarray(v) for k, v in next(iter_packed_batches(d, np.array([0]), spec)).items()}

    if args.random_init is None:
        params = {"params": load_params(args.params)["params"]}
    else:
        f0 = frames[0]
        b0 = batch_for(data["Z"][off[f0]:off[f0 + 1]].astype(np.int32), data["R"][off[f0]:off[f0 + 1]])
        params = {"params": model.init(jax.random.PRNGKey(args.random_init), atomic_numbers=b0["atomic_numbers"],
                                       positions=b0["positions"], Ef=jnp.zeros((2, 3)),
                                       dst_idx_flat=b0["dst_idx_flat"], src_idx_flat=b0["src_idx_flat"],
                                       batch_segments=b0["batch_segments"], batch_size=2,
                                       coulomb_dst_idx_flat=b0.get("coulomb_dst_idx_flat"),
                                       coulomb_src_idx_flat=b0.get("coulomb_src_idx_flat"))["params"]}
        # Output heads are zero-initialised; give every all-zero array small random values so the
        # charges, atomic dipoles and energy are non-trivial functions of the features.
        leaves, tree = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(jax.random.PRNGKey(args.random_init + 1), len(leaves))
        leaves = [jnp.where(jnp.all(x == 0), 0.1 * jax.random.normal(k, x.shape, x.dtype), x) for x, k in zip(leaves, keys)]
        params = jax.tree_util.tree_unflatten(tree, leaves)

    def run(Z, R, ef):
        out = jax.device_get(evaluate(batch_for(Z, R), jnp.asarray(ef, jnp.float32)))
        n = len(Z)
        out["F"] = out["F"][:n]; out["q"] = out["q"][:n]
        out["feats"] = {k: v[:n] for k, v in out["feats"].items()}
        return out

    rng = np.random.default_rng(0)
    S = np.diag([1.0, 1.0, -1.0])
    rec = []
    layer_names = None
    for fi, f in enumerate(frames):
        Z = data["Z"][off[f]:off[f + 1]].astype(np.int32)
        R0 = data["R"][off[f]:off[f + 1]].astype(np.float64)
        c = R0.mean(0)
        R = R0 - c
        dirs = rng.normal(size=3); dirs /= np.linalg.norm(dirs)
        Qs = [rot(rng) for _ in range(args.n_rot)]
        perm = rng.permutation(len(Z))
        for fin in FIELDS_INPUT:
            ef = fin * dirs
            base = run(Z, R, ef)
            if layer_names is None:
                layer_names = sorted(base["feats"])
            r = {"frame": f, "field": fin, "n": len(Z),
                 "U": float(base["U"]), "mu": base["mu"], "alpha": base["alpha"],
                 "sumF": float(np.linalg.norm(base["F"].sum(0))),
                 "Fnorm": float(np.sqrt((base["F"] ** 2).sum(1).mean())),
                 "torque": float(np.linalg.norm(np.cross(R, base["F"]).sum(0))),
                 "sumq": float(base["q"].sum()),
                 "alpha_asym": float(np.linalg.norm(base["alpha"] - base["alpha"].T) / np.linalg.norm(base["alpha"])),
                 "mu_E": base["mu_E"], "alpha_E": base["alpha_E"]}
            tests = {}
            # transform name -> (R', ef', expected maps for F/mu/alpha, pseudoscalar sign)
            T = {f"rot{i}": (R @ Q.T, Q @ ef, Q, +1) for i, Q in enumerate(Qs)}
            T["inversion"] = (-R, -ef, -np.eye(3), -1)
            T["mirror"] = (R @ S, S @ ef, S, -1)
            if fin:
                T["mirror_Efixed"] = (R @ S, ef, S, -1)
            T["translate"] = (R + np.array([3.1, -2.2, 5.4]), ef, np.eye(3), +1)
            for name, (Rt, eft, M, sgn) in T.items():
                o = run(Z, Rt, eft)
                t = {"dU": float(o["U"] - base["U"]),
                     "dF": float(np.linalg.norm(o["F"] - base["F"] @ M.T) / max(np.linalg.norm(base["F"]), 1e-9)),
                     "dmu": float(np.linalg.norm(o["mu"] - M @ base["mu"]) / max(np.linalg.norm(base["mu"]), 1e-9)),
                     "dalpha": float(np.linalg.norm(o["alpha"] - M @ base["alpha"] @ M.T) / np.linalg.norm(base["alpha"])),
                     "dq": float(np.abs(o["q"] - base["q"]).max())}
                for ln in layer_names:
                    x0, x1 = base["feats"][ln], o["feats"][ln]
                    s0, s1 = x0[:, 0], x1[:, 0]
                    t[f"scal/{ln}"] = float(np.linalg.norm(s1 - s0) / max(np.linalg.norm(s0), 1e-12))
                    if x0.shape[1] == 2:
                        p0, p1 = x0[:, 1], x1[:, 1]
                        nrm = max(np.linalg.norm(p0), 1e-12)
                        # sgn=-1: odd part (p0 - p1)/2 is the correct, enantiomer-sensing part; even part is contamination
                        t[f"ps_odd/{ln}"] = float(np.linalg.norm((p0 - p1) / 2) / nrm)
                        t[f"ps_even/{ln}"] = float(np.linalg.norm((p0 + p1) / 2) / nrm)
                        t[f"ps_err/{ln}"] = float(np.linalg.norm(p1 - sgn * p0) / nrm)
                tests[name] = t
            o = run(Z[perm], R[perm], ef)
            tests["permute"] = {"dU": float(o["U"] - base["U"]),
                                "dF": float(np.linalg.norm(o["F"] - base["F"][perm]) / max(np.linalg.norm(base["F"]), 1e-9)),
                                "dmu": float(np.linalg.norm(o["mu"] - base["mu"]) / max(np.linalg.norm(base["mu"]), 1e-9)),
                                "dalpha": float(np.linalg.norm(o["alpha"] - base["alpha"]) / np.linalg.norm(base["alpha"])),
                                "dq": float(np.abs(o["q"] - base["q"][perm]).max())}
            r["tests"] = tests
            if fin == 0.0:
                zero = base
            else:
                lin = base["mu"] - zero["mu"]
                pred = zero["alpha"] @ (ef * conv)
                r["linresp"] = float(np.linalg.norm(lin - pred) / max(np.linalg.norm(lin), 1e-12))
                r["dmu_field"] = float(np.linalg.norm(lin))
            rec.append(r)
        if fi % 10 == 0:
            print(f"{fi + 1}/{len(frames)} frames", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"ckpt": str(args.ckpt_dir), "params": str(args.params), "pseudotensors": pseudo,
                                    "parity_fix": bool(args.parity_fix), "parity_fix_dipole": bool(args.parity_fix_dipole),
                                    "random_init": args.random_init,
                                    "layers": layer_names, "fields_input": FIELDS_INPUT, "records": rec},
                                   default=lambda o: np.asarray(o).tolist()))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
