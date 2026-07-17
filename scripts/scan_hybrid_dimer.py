#!/usr/bin/env python3
"""1D dimer scan through the hybrid ML/MM calculator, with per-term decomposition.

Sweeps the centre-to-centre separation of two rigid monomers and reports what
each term of the hybrid potential contributes. This is the picture of the
handoff: ML owns close range, MM owns the tail, and the taper blends them --
so the scan shows directly whether that blend is smooth or has a step in it.

Atom ORDER matters and is easy to get wrong. The MD calculator assigns MM types
and charges by PSF index, while the dataset carries them in each structure's own
order (graph isomorphism at prep time). For acetone those differ -- the PSF puts
O first, the dataset fourth -- so monomers are reindexed into PSF order here.
DCM's two orders coincide, which is why only ACO ever showed the discrepancy.
See scripts/check_hybrid_train_md_parity.py.

Run on a node with CHARMM (build_mm_energy_forces_fn indexes param.get_atc()):

    python scripts/scan_hybrid_dimer.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz \
        --mm-switch-on 6.0 --ml-switch-width 1.5 --mm-switch-width 5.0 \
        --out dimer_scans

Defaults for the handoff come from the shared constants, so a scan run with no
flags reflects what MD would actually do today.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _psf_permutation(ds_Z, ds_q, psf_Z, psf_q):
    """Permutation putting dataset-ordered atoms into PSF order (match on Z, charge)."""
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


def _setup_charmm_psf(resid: str, n_monomers: int) -> int:
    """Generate a CHARMM PSF for ``n_monomers`` copies of ``resid``."""
    import pycharmm

    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        pycharmm_quiet,
        reset_block,
    )

    pycharmm_quiet()
    reset_block()
    atoms = setupRes.main(resid)
    pycharmm.read.sequence_string(" ".join([resid] * n_monomers))
    pycharmm.gen.new_segment(seg_name=resid, setup_ic=True)
    pycharmm.ic.prm_fill(replace_all=True)
    return len(atoms)


def _psf_ordered_monomer(data, resid: str, n_real: int):
    """A real monomer geometry from the dataset, reindexed into PSF order."""
    import pycharmm

    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    res = np.array([str(x) for x in data["res_name"]])
    idx = np.where(res == resid)[0]
    if len(idx) == 0:
        raise ValueError(f"no {resid} monomer in the dataset")
    i = int(idx[0])

    psf_q = np.asarray(pycharmm.psf.get_charges())[:n_real]
    psf_Z = np.asarray(get_Z_from_psf())[:n_real]
    ds_Z = np.asarray(data["Z"])[i][:n_real]
    ds_q = np.asarray(data["cgenff_charge"])[i][:n_real]
    perm = _psf_permutation(ds_Z, ds_q, psf_Z, psf_q)
    R = np.asarray(data["R"])[i][:n_real][perm]
    Z = ds_Z[perm]
    return Z, R - R.mean(axis=0)  # centred


def _terms(out) -> dict:
    def g(name):
        v = getattr(out, name, None)
        if v is None:
            return float("nan")
        arr = np.asarray(v)
        return float(arr.sum()) if arr.size else float("nan")

    return {
        "E_total": g("energy"),
        "internal_E": g("internal_E"),
        "ml_2b_E": g("ml_2b_E"),
        "mm_E": g("mm_E"),
        "mm_vdw_E": g("mm_vdw_E"),
        "mm_elec_E": g("mm_elec_E"),
        "wall_E": g("wall_E"),
    }


def main() -> int:
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="NPZ with monomers + CGenFF fields")
    p.add_argument("--resids", default="DCM,ACO", help="comma-separated residues to scan")
    p.add_argument("--r-min", type=float, default=2.5)
    p.add_argument("--r-max", type=float, default=12.0)
    p.add_argument("--n-points", type=int, default=60)
    p.add_argument("--ml-switch-width", type=float, default=DEFAULT_ML_SWITCH_WIDTH)
    p.add_argument("--mm-switch-on", type=float, default=DEFAULT_MM_SWITCH_ON)
    p.add_argument("--mm-switch-width", type=float, default=DEFAULT_MM_SWITCH_WIDTH)
    p.add_argument("--axis", default="1,0,0")
    p.add_argument("--out", default="dimer_scans")
    args = p.parse_args()

    import jax.numpy as jnp
    from ase import Atoms

    from mmml.analysis.dimer_scans import build_rigid_dimer
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    axis = np.array([float(x) for x in args.axis.split(",")])
    data = dict(np.load(args.data, allow_pickle=True))

    cutoff_params = CutoffParameters(
        ml_switch_width=args.ml_switch_width,
        mm_switch_on=args.mm_switch_on,
        mm_switch_width=args.mm_switch_width,
    )
    print(
        f"handoff: ML fully on 0-{args.mm_switch_on - args.ml_switch_width:g}, "
        f"handoff {args.mm_switch_on - args.ml_switch_width:g}-{args.mm_switch_on:g}, "
        f"MM tail {args.mm_switch_on:g}-{args.mm_switch_on + args.mm_switch_width:g} A"
    )

    results = {}
    for resid in [r.strip() for r in args.resids.split(",") if r.strip()]:
        n_mono = _setup_charmm_psf(resid, 2)
        Z1, R1 = _psf_ordered_monomer(data, resid, n_mono)
        mono = Atoms(numbers=Z1, positions=R1)

        factory = setup_calculator(
            ATOMS_PER_MONOMER=[n_mono, n_mono],
            N_MONOMERS=2,
            ml_switch_width=args.ml_switch_width,
            mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width,
            complementary_handoff=True,
            doML=True,
            doMM=True,
            doML_dimer=True,
            model_restart_path=args.checkpoint,
            MAX_ATOMS_PER_SYSTEM=2 * n_mono,
            ml_energy_conversion_factor=1,
            ml_force_conversion_factor=1,
        )
        _calc, sc_fn, update_fn_factory = factory(
            atomic_numbers=np.concatenate([Z1, Z1]),
            atomic_positions=np.concatenate([R1, R1 + np.array([8.0, 0, 0])]),
            n_monomers=2,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
        )

        rows = []
        for r in np.linspace(args.r_min, args.r_max, args.n_points):
            dimer, _ = build_rigid_dimer(
                mono, mono, distance_angstrom=float(r), axis=axis, center="centroid"
            )
            R = np.asarray(dimer.get_positions())
            Z = np.asarray(dimer.get_atomic_numbers())
            mm_pair_idx = mm_pair_mask = None
            if update_fn_factory is not None:
                ufn = update_fn_factory(R, cutoff_params)
                if ufn is not None:
                    mm_pair_idx, mm_pair_mask = ufn(R)
            out = sc_fn(
                jnp.asarray(R), jnp.asarray(Z), 2, cutoff_params,
                doML=True, doMM=True, doML_dimer=True,
                mm_pair_idx=mm_pair_idx, mm_pair_mask=mm_pair_mask,
            )
            t = _terms(out)
            t["r_com"] = float(r)
            a = R[:n_mono]
            b = R[n_mono:]
            t["min_contact"] = float(np.linalg.norm(a[:, None] - b[None, :], axis=-1).min())
            rows.append(t)

        keys = ["r_com", "min_contact", "E_total", "internal_E", "ml_2b_E",
                "mm_E", "mm_vdw_E", "mm_elec_E", "wall_E"]
        csv = out_dir / f"scan_{resid}.csv"
        with csv.open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in rows:
                fh.write(",".join(f"{row[k]:.8g}" for k in keys) + "\n")

        # Interaction curve: subtract the well-separated limit so the monomers'
        # internal energy (which dwarfs the interaction) drops out.
        E = np.array([r["E_total"] for r in rows])
        e_int = E - E[-1]
        results[resid] = (np.array([r["r_com"] for r in rows]), e_int, rows)

        i_min = int(np.argmin(e_int))
        print(f"\n=== {resid}-{resid} ({n_mono} atoms/monomer) -> {csv}")
        print(f"  minimum: r_com={rows[i_min]['r_com']:.2f} A  "
              f"E_int={e_int[i_min]:.4f} eV ({e_int[i_min] * 23.0605:.2f} kcal/mol)  "
              f"closest contact={rows[i_min]['min_contact']:.2f} A")
        print(f"{'r_com':>7} {'E_int':>10} {'ml_2b':>10} {'mm_E':>10} {'wall':>9}")
        for row, ei in list(zip(rows, e_int))[:: max(1, len(rows) // 12)]:
            print(f"{row['r_com']:7.2f} {ei:10.4f} {row['ml_2b_E']:10.4f} "
                  f"{row['mm_E']:10.4f} {row['wall_E']:9.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2), squeeze=False)
        for ax, (resid, (r, e_int, rows)) in zip(axes[0], results.items()):
            ax.axhline(0.0, color="0.8", lw=0.8)
            ax.plot(r, e_int, "-", color="#1f77b4", lw=2, label="E_total - E(inf)")
            ax.plot(r, [x["mm_E"] for x in rows], "--", color="#d62728", lw=1.2, label="MM")
            ax.plot(r, [x["ml_2b_E"] for x in rows], ":", color="#2ca02c", lw=1.4, label="ML 2-body")
            on = args.mm_switch_on
            ax.axvspan(on - args.ml_switch_width, on, color="0.9", zorder=0, label="handoff")
            ax.set_title(f"{resid}-{resid}")
            ax.set_xlabel("COM separation (Å)")
            ax.set_ylabel("energy (eV)")
            lo = float(np.nanmin(e_int))
            ax.set_ylim(min(lo * 1.5, -0.05), max(0.15, abs(lo)))
            ax.legend(fontsize=8)
        fig.tight_layout()
        png = out_dir / "dimer_scans.png"
        fig.savefig(png, dpi=150)
        print(f"\nplot -> {png}")
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
