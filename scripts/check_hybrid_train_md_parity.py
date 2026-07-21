#!/usr/bin/env python3
"""E2E gate: hybrid TRAINING energy must equal the MD calculator's energy.

The unit tests pin the pieces independently -- CGenFF parameters
(``test_cgenff_lj_parity``), the MM formula (``test_cgenff_mm_energy``), the
switching (``test_ml_switch_scale``) and the assembly (``test_hybrid_energy``).
This is the only check that the *assembled* number a model is trained on equals
the number it is deployed with.

It cannot be a CI unit test: the MD path's MM (``mm_energy_forces``) indexes
``pycharmm.param.get_atc()``, so it needs a live CHARMM session.  Run it on the
cluster after training:

    # Mode A (fixed CGenFF charges) — default, always run first
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz

    # Mode B (latent) — checkpoint trained with --mm-charge-mode latent
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz \
        --mm-charge-mode latent

    # Mode C (fixed+latent) — --mm-charge-mode fixed_plus_latent or --mm-charge-correction
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz \
        --mm-charge-correction

IMPORTANT: point ``--checkpoint`` at a *frozen* checkpoint, not a run that is
still training.  ``_load_physnet_checkpoint`` resolves ``get_last()`` once at
startup while ``setup_calculator`` re-resolves it per structure, so against a
live run this compares two different epochs and reports a spurious ~0.1 eV
mismatch that scales with molecule size.

Known open discrepancy: ACO+ACO fails by up to 2.4e-2 eV wherever E_MM != 0
(handoff/MM-tail), while its ML-only case and *all* DCM cases pass at <=3e-4 eV.
So the ML assembly is confirmed and the gap is in the MM term for acetone only.
Atom ordering was tested and RULED OUT: the dataset's ACO order differs from the
PSF's (O first in the PSF, fourth in the dataset), but permuting into PSF order
moved E_md by only ~2e-5.  Next suspect is the ACO CGenFF LJ parameters
(dataset master tables vs the PSF's).

Compares, for real DCM+DCM and ACO+ACO dimers spanning the ML-only / handoff /
MM-tail regimes:

    training path : hybrid_forward(model.apply, params, batch)
    MD path       : setup_calculator(...)(Z, R, n_monomers=2)

See ``docs/hybrid-mm-charges.md`` for the fixed / latent / fixed+latent taxonomy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _pick_dimers(data, n_per_regime=2):
    """Real dimers spanning ML-only (<6.5), handoff (6.5-8) and MM tail (8-13)."""
    mol_id = np.asarray(data["mol_id"])
    R = np.asarray(data["R"])
    res = np.asarray(data["res_name"])
    picks = []
    for name in sorted({str(r) for r in res if "," in str(r)}):
        idx = np.where(res == name)[0]
        r_com = []
        for i in idx:
            m0, m1 = mol_id[i] == 0, mol_id[i] == 1
            r_com.append(np.linalg.norm(R[i][m0].mean(0) - R[i][m1].mean(0)))
        r_com = np.asarray(r_com)
        for lo, hi, label in ((0, 6.5, "ML-only"), (6.5, 8.0, "handoff"), (8.0, 13.0, "MM-tail")):
            sel = idx[(r_com >= lo) & (r_com < hi)][:n_per_regime]
            picks += [(int(i), name, label) for i in sel]
    return picks


def _psf_permutation(ds_Z, ds_q, psf_Z, psf_q):
    """Permutation putting dataset-ordered atoms into PSF order.

    Matched on (Z, charge). Atoms identical in both (e.g. the six methyl H) are
    interchangeable for the MM term, so any consistent choice among them is fine.
    """
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


def _to_psf_order(data, i, n_real):
    """Reindex structure ``i``'s real atoms into PSF order, per monomer.

    The dataset carries CGenFF types/charges matched to each structure's OWN
    atom order (graph isomorphism at prep time); for ACO that is NOT the PSF
    order -- the PSF lists O first, the dataset fourth. Training is
    self-consistent in dataset order and MD is self-consistent in PSF order,
    but handing dataset-ordered coordinates to a PSF-ordered CHARMM system
    assigns the right charges to the wrong atoms. DCM's two orders coincide,
    which is exactly why only ACO failed parity (d_mm ~1e-9 for DCM vs ~2.4e-2
    for ACO).
    """
    import pycharmm

    from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

    psf_q = np.asarray(pycharmm.psf.get_charges())[:n_real]
    psf_Z = np.asarray(get_Z_from_psf())[:n_real]
    mol = np.asarray(data["mol_id"])[i][:n_real]
    ds_Z = np.asarray(data["Z"])[i][:n_real]
    ds_q = np.asarray(data["cgenff_charge"])[i][:n_real]

    perm = np.empty(n_real, dtype=int)
    for m in sorted({int(x) for x in mol if x >= 0}):
        sel = np.where(mol == m)[0]
        lo, hi = m * len(sel), (m + 1) * len(sel)
        perm[lo:hi] = sel[_psf_permutation(ds_Z[sel], ds_q[sel], psf_Z[lo:hi], psf_q[lo:hi])]
    return perm


def _add_wall_probe(data, picks, target_sep):
    """Append a dimer squeezed to ``target_sep`` closest inter-monomer contact.

    Every real dimer sits above the 1.0 A wall onset, so the wall term is zero
    on all of them and they would pass whether the MD wall were wired correctly,
    wired wrong, or absent. This makes the term live so the gate can fail.
    """
    mol = np.asarray(data["mol_id"])
    res = np.array([str(x) for x in data["res_name"]])
    taken = {int(t[0]) for t in picks}
    cand = [
        k for k in range(len(res))
        if "," in res[k] and (mol[k] == 1).any() and k not in taken
    ]
    if not cand:
        return picks
    i = cand[0]
    R = np.asarray(data["R"]).copy()
    a, b = mol[i] == 0, mol[i] == 1
    axis = R[i][b].mean(0) - R[i][a].mean(0)
    axis = axis / np.linalg.norm(axis)
    for _ in range(500):
        d = np.linalg.norm(R[i][a][:, None] - R[i][b][None, :], axis=-1).min()
        if d <= target_sep:
            break
        R[i][b] -= axis * max((d - target_sep) * 0.5, 1e-3)
    data["R"] = R
    sep = np.linalg.norm(R[i][a][:, None] - R[i][b][None, :], axis=-1).min()
    print(f"[wall-probe] structure {i} ({res[i]}) squeezed to closest contact {sep:.3f} A")
    return list(picks) + [(i, res[i], "WALL")]


def _md_parts(res) -> dict:
    """Pull the MD calculator's per-term decomposition out of its ModelOutput.

    Comparing only totals cannot say WHICH term disagrees; ModelOutput already
    carries the pieces, so read them directly rather than inferring.
    """

    def g(name):
        v = getattr(res, name, None)
        if v is None:
            return float("nan")
        arr = np.asarray(v)
        return float(arr.sum()) if arr.size else float("nan")

    return {
        # Training folds the wall into e_mm (it lives inside _emm); MD reports it
        # separately as wall_E. Fold it back in so the columns compare like for like.
        "mm_E": g("mm_E") + (g("wall_E") if not np.isnan(g("wall_E")) else 0.0),
        "wall_E": g("wall_E"),
        "mm_vdw": g("mm_vdw_E"),
        "mm_elec": g("mm_elec_E"),
        "internal": g("internal_E"),
        "ml_2b": g("ml_2b_E"),
        "com_dist": g("com_dist"),
    }


def _train_com_dist(data, i) -> float:
    """Centroid separation, the switching input on the training side."""
    mol = np.asarray(data["mol_id"])[i]
    R = np.asarray(data["R"])[i]
    if not (mol == 1).any():
        return float("nan")
    return float(np.linalg.norm(R[mol == 0].mean(0) - R[mol == 1].mean(0)))


def _setup_charmm_psf(resid: str, n_monomers: int) -> int:
    """Generate a CHARMM PSF for ``n_monomers`` copies of ``resid``.

    build_mm_energy_forces_fn indexes pycharmm.param.get_atc(), so the MD side
    needs a live PSF for this exact system -- there is no CHARMM-free shortcut.
    Returns the atom count per monomer.
    """
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


def _batch_from_structure(data, i):
    """Single-structure batch in the layout prepare_batches_jit produces."""
    import jax.numpy as jnp

    Z = np.asarray(data["Z"])[i]
    R = np.asarray(data["R"])[i]
    n = Z.shape[0]
    atom_mask = (Z > 0).astype(np.float32)
    idx = np.arange(n)
    dst, src = np.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src) & (atom_mask[dst] > 0) & (atom_mask[src] > 0)
    return {
        "R": jnp.asarray(R),
        "Z": jnp.asarray(Z),
        "atom_mask": jnp.asarray(atom_mask),
        "batch_mask": jnp.asarray(keep.astype(np.float32)),
        "dst_idx": jnp.asarray(dst),
        "src_idx": jnp.asarray(src),
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
        "mol_id": jnp.asarray(np.asarray(data["mol_id"])[i][None, :]),
        "cgenff_type_idx": jnp.asarray(np.asarray(data["cgenff_type_idx"])[i][None, :]),
        "cgenff_charge": jnp.asarray(np.asarray(data["cgenff_charge"])[i][None, :]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Trained PhysNet checkpoint dir")
    p.add_argument("--data", required=True, help="NPZ with CGenFF fields (out_combined_dedup)")
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    p.add_argument("--mm-switch-on", type=float, default=8.0)
    p.add_argument("--mm-switch-width", type=float, default=5.0)
    p.add_argument(
        "--mm-charge-mode",
        choices=["fixed", "latent", "fixed_plus_latent"],
        default=None,
        help="MM Coulomb charge mode (default: fixed). Must match training.",
    )
    p.add_argument(
        "--mm-charge-correction",
        action="store_true",
        help=(
            "Alias for --mm-charge-mode fixed_plus_latent. "
            "Must match how the checkpoint was trained. Default is Mode A (fixed)."
        ),
    )
    p.add_argument("--tol", type=float, default=1e-3, help="abs energy tolerance (eV)")
    p.add_argument(
        "--force-tol",
        type=float,
        default=1e-2,
        help="max abs per-component force tolerance (eV/Angstrom)",
    )
    p.add_argument(
        "--wall-probe",
        type=float,
        default=0.0,
        help=(
            "If >0, append a synthetic close-contact dimer squeezed to this "
            "closest atom-atom separation (Angstrom), so the short-range wall is "
            "non-zero and actually exercised. The real dimers all sit where the "
            "wall is exactly 0 and cannot detect a mis-wired MD wall."
        ),
    )
    args = p.parse_args()

    import jax.numpy as jnp

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.models.hybrid_energy import hybrid_forward
    from mmml.models.mm_charge_mode import (
        mm_charge_mode_needs_q_ml,
        resolve_hybrid_mm_charge_mode,
    )

    mode = resolve_hybrid_mm_charge_mode(
        mm_charge_mode=args.mm_charge_mode,
        charge_correction=bool(args.mm_charge_correction),
    )

    data = dict(np.load(args.data, allow_pickle=True))
    sigmas = jnp.asarray(data["cgenff_master_sigmas"])
    epsilons = jnp.asarray(data["cgenff_master_epsilons"])
    natoms = int(np.asarray(data["Z"]).shape[1])

    _, params, model = _load_physnet_checkpoint(Path(args.checkpoint), natoms)
    if mm_charge_mode_needs_q_ml(mode) and not getattr(model, "charges", False):
        print(
            f"--mm-charge-mode {mode.value} needs a checkpoint built with charges=True",
            file=sys.stderr,
        )
        return 1

    picks = _pick_dimers(data)
    if args.wall_probe:
        picks = _add_wall_probe(data, picks, args.wall_probe)
    if not picks:
        print("no dimers found in the dataset", file=sys.stderr)
        return 1

    cutoff_params = CutoffParameters(
        ml_switch_width=args.ml_switch_width,
        mm_switch_on=args.mm_switch_on,
        mm_switch_width=args.mm_switch_width,
    )
    print(f"mm_charge_mode={mode.value}")

    print(
        f"{'idx':>5} {'species':>8} {'regime':>8} {'dE_tot':>10} | "
        f"{'e_mm(tr)':>10} {'mm_E(md)':>10} {'d_mm':>10} | "
        f"{'s_ml':>6} {'rcom(tr)':>9} | {'dF_max':>9}"
    )
    worst = 0.0
    worst_f = 0.0
    bad = 0
    current_species = None
    for i, name, regime in sorted(picks, key=lambda t: t[1]):
        resid = name.split(",")[0].strip()
        if resid != current_species:
            _setup_charmm_psf(resid, 2)
            current_species = resid
        Z = np.asarray(data["Z"])[i]
        R = np.asarray(data["R"])[i]
        n_real = int(np.asarray(data["N"])[i])
        per_mono = n_real // 2

        # The MD side indexes MM types/charges by PSF position, so it must be
        # handed PSF-ordered coordinates. The training side reads `data`
        # directly and is unaffected (the model is permutation-equivariant).
        _perm = _to_psf_order(data, i, n_real)
        Z = Z[:n_real][_perm]
        R = R[:n_real][_perm]

        # --- training path -------------------------------------------------
        batch = _batch_from_structure(data, i)
        out = hybrid_forward(
            model.apply, params, batch, 1, sigmas, epsilons,
            mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width,
            ml_switch_width=args.ml_switch_width,
            mm_charge_mode=mode.value,
        )
        e_train = float(np.asarray(out["energy"]).reshape(-1)[0])
        # Dataset order; reindexed to PSF order below to match the MD side.
        f_train = np.asarray(out["forces"]).reshape(-1, 3)[:n_real][_perm]
        e_mm_train = float(np.asarray(out["e_mm"]).reshape(-1)[0])
        s_train = float(np.asarray(out["ml_scale"]).reshape(-1)[0])
        rcom_train = _train_com_dist(data, i)

        # --- MD path -------------------------------------------------------
        factory = setup_calculator(
            ATOMS_PER_MONOMER=[per_mono, per_mono],
            N_MONOMERS=2,
            ml_switch_width=args.ml_switch_width,
            mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width,
            complementary_handoff=True,
            doML=True,
            doMM=True,
            doML_dimer=True,
            model_restart_path=args.checkpoint,
            MAX_ATOMS_PER_SYSTEM=n_real,
            ml_energy_conversion_factor=1,
            ml_force_conversion_factor=1,
            mm_charge_mode=mode.value,
        )
        _calc, sc_fn, update_fn_factory = factory(
            atomic_numbers=Z[:n_real],
            atomic_positions=R[:n_real],
            n_monomers=2,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
        )
        R_jax = jnp.asarray(R[:n_real])
        Z_jax = jnp.asarray(Z[:n_real])
        mm_pair_idx = mm_pair_mask = None
        if update_fn_factory is not None:
            update_fn = update_fn_factory(R[:n_real], cutoff_params)
            if update_fn is not None:
                mm_pair_idx, mm_pair_mask = update_fn(R[:n_real])
        out_md = sc_fn(
            R_jax,
            Z_jax,
            2,
            cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
        )
        e_md = float(np.asarray(out_md.energy).reshape(-1)[0])
        md = _md_parts(out_md)
        f_md = np.asarray(out_md.forces).reshape(-1, 3)[:n_real]

        diff = abs(e_train - e_md)
        # Forces are what dynamics integrates, and the ds/dR product-rule term in
        # the handoff appears ONLY here -- an energy-only gate cannot see it.
        fdiff = float(np.abs(f_train - f_md).max())
        worst = max(worst, diff)
        worst_f = max(worst_f, fdiff)
        # The wall probe carries a ~7 eV / ~24 eV/A term, where float32 noise
        # alone exceeds tolerances tuned for ~1 eV/A forces: judge it relatively.
        if regime == "WALL":
            fscale = max(float(np.abs(f_md).max()), 1.0)
            ok = (diff <= max(args.tol, 1e-3 * abs(e_md))) and (
                fdiff <= args.force_tol * fscale
            )
        else:
            ok = (diff <= args.tol) and (fdiff <= args.force_tol)
        bad += (not ok)
        d_mm = e_mm_train - md["mm_E"]
        print(
            f"{i:>5} {name:>8} {regime:>8} {diff:>10.2e} | "
            f"{e_mm_train:>10.5f} {md['mm_E']:>10.5f} {d_mm:>10.2e} | "
            f"{s_train:>6.3f} {rcom_train:>9.3f} | {fdiff:>9.2e}"
            f"  {'OK' if ok else 'FAIL'}"
        )

    print(f"\nworst |E_train - E_md| = {worst:.3e} eV (tol {args.tol})")
    print(f"worst |F_train - F_md| = {worst_f:.3e} eV/A (tol {args.force_tol})")
    if bad:
        print(f"PARITY FAILED on {bad}/{len(picks)} structures", file=sys.stderr)
        return 1
    print(f"PARITY OK on {len(picks)} structures: training == MD calculator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
