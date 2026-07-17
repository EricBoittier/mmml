#!/usr/bin/env python3
"""Validate hybrid dimer rays against the MD calculator and against GFN2-xTB.

Two different questions, deliberately kept apart:

* **CHARMM** (``--with-charmm``) -- the deployed calculator, same weights. This
  tests the SHORTCUT: the orientation scan uses ``hybrid_forward`` because it is
  batched and CHARMM-free, licensed by a parity gate that only ever ran on 13
  dataset structures in 3 regimes. These rays reach orientations no dataset
  structure occupies, so the licence needs re-earning here.
* **GFN2-xTB** (``--with-xtb``) -- an INDEPENDENT potential. This tests the
  PREMISE: "a rigid scan admits one minimum" is an assumption. If xTB shows the
  same double wells, the minima are real and the ML model is right. Only if xTB
  is smooth where the model is not are they genuinely spurious.

xTB is semi-empirical, not the MP2 the model was trained on, so treat it as
evidence about the SHAPE of the curve (how many minima, roughly where), not as
a reference for well depths.

Ray indexing is imported from scan_dimer_orientations so ray N here is ray N
there.

    python scripts/validate_dimer_rays.py --checkpoint CKPT --data D.npz \\
        --resid ACO --rays 0,7,13 --with-xtb --with-charmm --out validate_ACO
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.scan_dimer_orientations import (  # noqa: E402
    fibonacci_sphere,
    quat_to_matrix,
    super_fibonacci,
)

EV_TO_KCAL = 23.0605


def _psf_permutation(ds_Z, ds_q, psf_Z, psf_q):
    used, perm = set(), []
    for z, q in zip(psf_Z, psf_q):
        for j in range(len(ds_Z)):
            if j in used or int(ds_Z[j]) != int(z) or abs(float(ds_q[j]) - float(q)) > 1e-6:
                continue
            perm.append(j)
            used.add(j)
            break
        else:
            raise ValueError(f"no dataset atom matches PSF atom (Z={z}, q={q})")
    return np.array(perm)


def main() -> int:
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--resid", default="ACO")
    p.add_argument("--rays", default="0,1,2", help="comma-separated ray indices")
    p.add_argument("--n-directions", type=int, default=10)
    p.add_argument("--n-orientations", type=int, default=24)
    p.add_argument("--r-min", type=float, default=3.0)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--n-r", type=int, default=36)
    p.add_argument("--ml-switch-width", type=float, default=DEFAULT_ML_SWITCH_WIDTH)
    p.add_argument("--mm-switch-on", type=float, default=DEFAULT_MM_SWITCH_ON)
    p.add_argument("--mm-switch-width", type=float, default=DEFAULT_MM_SWITCH_WIDTH)
    p.add_argument("--with-charmm", action="store_true")
    p.add_argument("--with-xtb", action="store_true")
    p.add_argument("--out", default="validate_rays")
    args = p.parse_args()

    import jax.numpy as jnp

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.models.hybrid_energy import hybrid_forward

    raw = dict(np.load(args.data, allow_pickle=True))
    res = np.array([str(x) for x in raw["res_name"]])
    k = int(np.where(res == args.resid)[0][0])
    n_mono = int(raw["N"][k])
    pad = int(np.asarray(raw["Z"]).shape[1])
    Z1 = np.asarray(raw["Z"][k])[:n_mono]
    R1 = np.asarray(raw["R"][k])[:n_mono]
    R1 = R1 - R1.mean(axis=0)
    t1 = np.asarray(raw["cgenff_type_idx"][k])[:n_mono]
    q1 = np.asarray(raw["cgenff_charge"][k])[:n_mono]
    n_at = 2 * n_mono

    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = np.linspace(args.r_min, args.r_max, args.n_r)
    rays = [int(x) for x in args.rays.split(",")]

    _, params, model = _load_physnet_checkpoint(Path(args.checkpoint), pad)
    sig = jnp.asarray(raw["cgenff_master_sigmas"])
    eps = jnp.asarray(raw["cgenff_master_epsilons"])
    KW = dict(mm_switch_on=args.mm_switch_on, mm_switch_width=args.mm_switch_width,
              ml_switch_width=args.ml_switch_width)

    def geom(ray: int, r: float):
        di, qi = ray // len(quats), ray % len(quats)
        Rb = R1 @ quat_to_matrix(quats[qi]).T
        R = np.zeros((pad, 3))
        R[:n_mono] = R1 - 0.5 * r * dirs[di]
        R[n_mono:n_at] = Rb + 0.5 * r * dirs[di]
        return R

    def ml_batch1(R):
        Z = np.zeros(pad, dtype=np.int32); Z[:n_mono] = Z1; Z[n_mono:n_at] = Z1
        T = np.full(pad, -1, np.int32); T[:n_mono] = t1; T[n_mono:n_at] = t1
        Q = np.zeros(pad); Q[:n_mono] = q1; Q[n_mono:n_at] = q1
        M = np.full(pad, -1, np.int32); M[:n_mono] = 0; M[n_mono:n_at] = 1
        am = (Z > 0).astype(np.float32)
        i = np.arange(pad); dst, src = np.meshgrid(i, i, indexing="ij")
        dst, src = dst.reshape(-1), src.reshape(-1)
        keep = (dst != src) & (am[dst] > 0) & (am[src] > 0)
        b = {"R": jnp.asarray(R), "Z": jnp.asarray(Z), "atom_mask": jnp.asarray(am),
             "batch_mask": jnp.asarray(keep.astype(np.float32)),
             "dst_idx": jnp.asarray(dst), "src_idx": jnp.asarray(src),
             "batch_segments": jnp.zeros(pad, dtype=jnp.int32),
             "mol_id": jnp.asarray(M[None, :]), "cgenff_type_idx": jnp.asarray(T[None, :]),
             "cgenff_charge": jnp.asarray(Q[None, :])}
        out = hybrid_forward(model.apply, params, b, 1, sig, eps, **KW)
        return float(np.asarray(out["energy"]).reshape(-1)[0])

    # --- CHARMM (deployed calculator, PSF order) ---------------------------
    sc_fn = update_fn_factory = cutoff_params = perm = None
    if args.with_charmm:
        import pycharmm

        from mmml.interfaces.pycharmmInterface import setupRes
        from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
        from mmml.interfaces.pycharmmInterface.import_pycharmm import (
            pycharmm_quiet,
            reset_block,
        )
        from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
        from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

        pycharmm_quiet(); reset_block()
        setupRes.main(args.resid)
        pycharmm.read.sequence_string(f"{args.resid} {args.resid}")
        pycharmm.gen.new_segment(seg_name=args.resid, setup_ic=True)
        pycharmm.ic.prm_fill(replace_all=True)
        psf_q = np.asarray(pycharmm.psf.get_charges())[:n_mono]
        psf_Z = np.asarray(get_Z_from_psf())[:n_mono]
        # The calculator indexes MM by PSF position; the scan geometry is in
        # dataset order. For acetone those differ (O first vs fourth).
        perm = _psf_permutation(Z1, q1, psf_Z, psf_q)
        cutoff_params = CutoffParameters(
            ml_switch_width=args.ml_switch_width, mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width)
        factory = setup_calculator(
            ATOMS_PER_MONOMER=[n_mono, n_mono], N_MONOMERS=2,
            ml_switch_width=args.ml_switch_width, mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width, complementary_handoff=True,
            doML=True, doMM=True, doML_dimer=True,
            model_restart_path=args.checkpoint, MAX_ATOMS_PER_SYSTEM=n_at,
            ml_energy_conversion_factor=1, ml_force_conversion_factor=1)
        Zc = np.concatenate([Z1[perm], Z1[perm]])
        _c, sc_fn, update_fn_factory = factory(
            atomic_numbers=Zc,
            atomic_positions=np.concatenate([R1[perm], R1[perm] + np.array([8.0, 0, 0])]),
            n_monomers=2, cutoff_params=cutoff_params,
            doML=True, doMM=True, doML_dimer=True, backprop=False)

    def charmm_e(R):
        Rp = np.concatenate([R[:n_mono][perm], R[n_mono:n_at][perm]])
        Zc = np.concatenate([Z1[perm], Z1[perm]])
        idx = mask = None
        if update_fn_factory is not None:
            u = update_fn_factory(Rp, cutoff_params)
            if u is not None:
                idx, mask = u(Rp)
        o = sc_fn(jnp.asarray(Rp), jnp.asarray(Zc), 2, cutoff_params,
                  doML=True, doMM=True, doML_dimer=True,
                  mm_pair_idx=idx, mm_pair_mask=mask)
        return float(np.asarray(o.energy).reshape(-1)[0])

    # --- GFN2-xTB (independent potential) ----------------------------------
    e_mono_xtb = None
    if args.with_xtb:
        from ase import Atoms
        from tblite.ase import TBLite

        m = Atoms(numbers=Z1, positions=R1)
        m.calc = TBLite(method="GFN2-xTB", verbosity=0)
        e_mono_xtb = float(m.get_potential_energy())
        print(f"GFN2-xTB rigid monomer E = {e_mono_xtb:.4f} eV "
              f"(constant: monomers are rigid, so E_int = E_dimer - 2*E_mono)")

    def xtb_e(R):
        from ase import Atoms

        d = Atoms(numbers=np.concatenate([Z1, Z1]),
                  positions=np.concatenate([R[:n_mono], R[n_mono:n_at]]))
        d.calc = TBLite(method="GFN2-xTB", verbosity=0)
        return float(d.get_potential_energy()) - 2.0 * e_mono_xtb

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"rays_{args.resid}.csv"
    t0 = time.time()
    with csv.open("w") as fh:
        fh.write("ray,direction,orientation,r_com,E_ml,E_charmm,E_xtb\n")
        for ray in rays:
            ml, ch, xt = [], [], []
            for r in rs:
                R = geom(ray, float(r))
                ml.append(ml_batch1(R))
                ch.append(charmm_e(R) if args.with_charmm else float("nan"))
                xt.append(xtb_e(R) if args.with_xtb else float("nan"))
            ml = np.array(ml); ch = np.array(ch); xt = np.array(xt)
            # interaction energies: subtract each curve's own separated limit
            ml -= ml[-1]
            if args.with_charmm:
                ch -= ch[-1]
            for i, r in enumerate(rs):
                fh.write(f"{ray},{ray // len(quats)},{ray % len(quats)},{r:.4f},"
                         f"{ml[i]:.8g},{ch[i]:.8g},{xt[i]:.8g}\n")
            msg = f"ray {ray:4d}: ML min {ml.min() * EV_TO_KCAL:7.2f}"
            if args.with_charmm:
                msg += (f" | CHARMM min {ch.min() * EV_TO_KCAL:7.2f}"
                        f" | max|ML-CHARMM| {np.abs(ml - ch).max():.2e} eV")
            if args.with_xtb:
                msg += f" | xTB min {xt.min() * EV_TO_KCAL:7.2f} kcal/mol"
            print(msg, flush=True)

    print(f"\n{len(rays)} rays x {len(rs)} points in {time.time() - t0:.0f}s -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
